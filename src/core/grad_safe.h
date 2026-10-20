#ifndef __SD_CORE_GRAD_SAFE_H__
#define __SD_CORE_GRAD_SAFE_H__

#include "ggml.h"

// Inplace ops abort ggml's backward walk. The normal sampler keeps them.
// flux2-dno sets the flag on the thread that builds its graph.
namespace sd {

inline bool& differentiable_ops_enabled() {
    static thread_local bool enabled = false;
    return enabled;
}

struct DifferentiableOpsScope {
    bool previous;
    bool previous_exact;
    explicit DifferentiableOpsScope(bool enabled)
        : previous(differentiable_ops_enabled()),
          previous_exact(ggml_set_exact_quant_matmul(enabled)) {
        differentiable_ops_enabled() = enabled;
    }
    ~DifferentiableOpsScope() {
        differentiable_ops_enabled() = previous;
        ggml_set_exact_quant_matmul(previous_exact);
    }
    DifferentiableOpsScope(const DifferentiableOpsScope&)            = delete;
    DifferentiableOpsScope& operator=(const DifferentiableOpsScope&) = delete;
};

inline const float* persistent_float(float value) {
    static float slots[8];
    static int count = 0;
    for (int i = 0; i < count; ++i) {
        if (slots[i] == value) {
            return &slots[i];
        }
    }
    GGML_ASSERT(count < 8);
    slots[count] = value;
    return &slots[count++];
}

inline ggml_tensor* maybe_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return differentiable_ops_enabled() ? ggml_add(ctx, a, b) : ggml_add_inplace(ctx, a, b);
}

inline ggml_tensor* maybe_mul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return differentiable_ops_enabled() ? ggml_mul(ctx, a, b) : ggml_mul_inplace(ctx, a, b);
}

inline ggml_tensor* maybe_silu(ggml_context* ctx, ggml_tensor* x) {
    return differentiable_ops_enabled() ? ggml_silu(ctx, x) : ggml_silu_inplace(ctx, x);
}

inline ggml_tensor* maybe_scale(ggml_context* ctx, ggml_tensor* x, float factor) {
    return differentiable_ops_enabled() ? ggml_scale(ctx, x, factor) : ggml_scale_inplace(ctx, x, factor);
}

inline ggml_tensor* maybe_softmax(ggml_context* ctx, ggml_tensor* x) {
    return differentiable_ops_enabled() ? ggml_soft_max(ctx, x) : ggml_soft_max_inplace(ctx, x);
}

inline ggml_tensor* maybe_gelu(ggml_context* ctx, ggml_tensor* x) {
    return differentiable_ops_enabled() ? ggml_gelu(ctx, x) : ggml_gelu_inplace(ctx, x);
}

inline ggml_tensor* maybe_gelu_quick(ggml_context* ctx, ggml_tensor* x) {
    return differentiable_ops_enabled() ? ggml_gelu_quick(ctx, x) : ggml_gelu_quick_inplace(ctx, x);
}

// Quantized weights stay quantized. A backward step still reads every weight,
// so materializing f32 copies would keep the whole model expanded at once.
// Vulkan has no quantized out-prod; that op falls back to the CPU.
inline ggml_tensor* cast_weight_f32(ggml_context* ctx, ggml_tensor* weight) {
    GGML_UNUSED(ctx);
    return weight;
}

template <typename SetData>
inline ggml_tensor* differentiable_layer_norm(ggml_context* ctx,
                                              ggml_tensor* x,
                                              ggml_tensor* weight,
                                              ggml_tensor* bias,
                                              float eps,
                                              SetData set_data) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    const float inv_n       = 1.0f / static_cast<float>(x->ne[0]);
    ggml_tensor* mean       = ggml_scale(ctx, ggml_sum_rows(ctx, x), inv_n);
    ggml_tensor* centered   = ggml_sub(ctx, x, mean);
    ggml_tensor* variance   = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, centered)), inv_n);
    ggml_tensor* eps_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    set_data(eps_tensor, persistent_float(eps));
    ggml_tensor* y = ggml_div(ctx, centered, ggml_sqrt(ctx, ggml_add(ctx, variance, eps_tensor)));
    if (weight != nullptr) {
        y = ggml_mul(ctx, y, weight);
        if (bias != nullptr) {
            y = ggml_add(ctx, y, bias);
        }
    }
    return y;
}

template <typename SetData>
inline ggml_tensor* differentiable_group_norm(ggml_context* ctx,
                                              ggml_tensor* x,
                                              ggml_tensor* weight,
                                              ggml_tensor* bias,
                                              int groups,
                                              float eps,
                                              SetData set_data) {
    GGML_ASSERT(groups > 0);
    GGML_ASSERT(x->ne[2] % groups == 0);
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    const int64_t width  = x->ne[0];
    const int64_t height = x->ne[1];
    const int64_t count  = width * height * (x->ne[2] / groups);
    const int64_t rows   = groups * x->ne[3];
    ggml_tensor* flat    = ggml_reshape_2d(ctx, x, count, rows);
    const float inv_n    = 1.0f / static_cast<float>(count);
    ggml_tensor* mean    = ggml_scale(ctx, ggml_sum_rows(ctx, flat), inv_n);
    ggml_tensor* centered = ggml_sub(ctx, flat, mean);
    ggml_tensor* variance = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, centered)), inv_n);
    ggml_tensor* eps_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    set_data(eps_tensor, persistent_float(eps));
    ggml_tensor* y = ggml_div(ctx, centered, ggml_sqrt(ctx, ggml_add(ctx, variance, eps_tensor)));
    y              = ggml_reshape_4d(ctx, y, width, height, x->ne[2], x->ne[3]);
    if (weight != nullptr && bias != nullptr) {
        weight = ggml_reshape_4d(ctx, weight, 1, 1, weight->ne[0], 1);
        bias   = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
        y      = ggml_add(ctx, ggml_mul(ctx, y, weight), bias);
    }
    return y;
}

}  // namespace sd

#endif
