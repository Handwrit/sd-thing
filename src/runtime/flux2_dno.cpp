#include "runtime/flux2_dno.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ggml-cpu.h"

#include "conditioning/conditioner.hpp"
#include "core/grad_safe.h"
#include "core/rng.hpp"
#include "core/util.h"
#include "model/diffusion/flux.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "pipeline/diffusion_engine.h"
#include "runtime/denoiser.hpp"

namespace sd {
namespace flux2_dno {
namespace {

constexpr int kPair = 2;

struct LatentStats {
    std::vector<float> mean;
    std::vector<float> stddev;
    float scale = 1.f;
};

void log_graph_time(const char* phase, std::chrono::steady_clock::time_point started) {
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    LOG_INFO("flux2-dno %s %.2f s", phase, seconds);
}

double dot(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

double norm(const float* v, int64_t n) {
    return std::sqrt(dot(v, v, n));
}

float log_probability_bound(double bound) {
    if (bound >= 1.0) {
        return 0.f;
    }
    return static_cast<float>(std::log(std::max(bound, 1e-30)));
}

// Lemma 1 is an upper bound. Capping it at 1 keeps a gradient in the
// low-probability region, which is the region the penalty is there to leave.
void pair_penalty(const float* z, int n, double* value, float* grad) {
    if (n < kPair || (n % kPair) != 0) {
        return;
    }
    const int m = n / kPair;
    const double k = static_cast<double>(kPair);
    double mean_x = 0.0;
    double mean_y = 0.0;
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (int i = 0; i < m; ++i) {
        const double x = z[2 * i];
        const double y = z[2 * i + 1];
        mean_x += x;
        mean_y += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    mean_x /= m;
    mean_y /= m;
    sxx = sxx / m - 1.0;
    syy = syy / m - 1.0;
    sxy /= m;

    const double m1 = std::hypot(mean_x, mean_y);
    const double p1_bound = 2.0 * std::exp(-m * m1 * m1 / (2.0 * k));
    if (value != nullptr) {
        *value += log_probability_bound(p1_bound);
    }
    if (grad != nullptr && p1_bound < 1.0 && m1 > 0.0) {
        const float pull = static_cast<float>(-mean_x / k);
        const float pully = static_cast<float>(-mean_y / k);
        for (int i = 0; i < m; ++i) {
            grad[2 * i] += pull;
            grad[2 * i + 1] += pully;
        }
    }

    const double half_diff = 0.5 * (sxx - syy);
    const double disc = std::hypot(half_diff, sxy);
    const double trace = sxx + syy;
    const double lambda_hi = 0.5 * trace + disc;
    const double lambda_lo = 0.5 * trace - disc;
    const double abs_hi = std::fabs(lambda_hi);
    const double abs_lo = std::fabs(lambda_lo);
    const bool use_hi = abs_hi >= abs_lo;
    const double lambda = use_hi ? lambda_hi : lambda_lo;
    const double m2 = std::max(abs_hi, abs_lo);
    const double skew = std::sqrt(k / static_cast<double>(m));
    const double inside = std::sqrt(1.0 + m2) - 1.0 - skew;
    const double p2_bound = inside > 0.0 ? 2.0 * std::exp(-0.5 * m * inside * inside) : 2.0;
    if (value != nullptr) {
        *value += log_probability_bound(p2_bound);
    }
    if (grad == nullptr || inside <= 0.0 || p2_bound >= 1.0 || m2 <= 0.0) {
        return;
    }

    double vx = sxy;
    double vy = lambda - sxx;
    if (std::fabs(vx) + std::fabs(vy) < 1e-12) {
        vx = lambda - syy;
        vy = sxy;
    }
    const double vnorm = std::hypot(vx, vy);
    if (vnorm < 1e-12) {
        vx = 1.0;
        vy = 0.0;
    } else {
        vx /= vnorm;
        vy /= vnorm;
    }
    const double sign = lambda >= 0.0 ? 1.0 : -1.0;
    const double gxx = sign * vx * vx;
    const double gyy = sign * vy * vy;
    const double gxy = sign * 2.0 * vx * vy;
    const double dlog_dm2 = (-m * inside) * (0.5 / std::sqrt(1.0 + m2));
    for (int i = 0; i < m; ++i) {
        const double x = z[2 * i];
        const double y = z[2 * i + 1];
        const double dm2_dx = (gxx * 2.0 * x + gxy * y) / m;
        const double dm2_dy = (gyy * 2.0 * y + gxy * x) / m;
        grad[2 * i] += static_cast<float>(dlog_dm2 * dm2_dx);
        grad[2 * i + 1] += static_cast<float>(dlog_dm2 * dm2_dy);
    }
}

std::vector<std::vector<int>> make_permutations(int n, int count, uint64_t seed) {
    std::vector<std::vector<int>> perms;
    perms.reserve(static_cast<size_t>(count));
    std::mt19937_64 rng(seed);
    std::vector<int> identity(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        identity[static_cast<size_t>(i)] = i;
    }
    for (int p = 0; p < count; ++p) {
        auto perm = identity;
        for (int i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<int> pick(0, i);
            std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(pick(rng))]);
        }
        perms.push_back(std::move(perm));
    }
    return perms;
}

double penalty_value(const float* z, int n, const std::vector<std::vector<int>>& perms) {
    if (perms.empty() || n < kPair) {
        return 0.0;
    }
    std::vector<float> shuffled(static_cast<size_t>(n));
    double total = 0.0;
    for (const auto& perm : perms) {
        for (int i = 0; i < n; ++i) {
            shuffled[static_cast<size_t>(i)] = z[perm[static_cast<size_t>(i)]];
        }
        double value = 0.0;
        pair_penalty(shuffled.data(), n, &value, nullptr);
        total += value;
    }
    return total / static_cast<double>(perms.size());
}

void penalty_gradient(const float* z, int n, const std::vector<std::vector<int>>& perms, float* grad) {
    std::fill(grad, grad + n, 0.f);
    if (perms.empty() || n < kPair) {
        return;
    }
    std::vector<float> shuffled(static_cast<size_t>(n));
    std::vector<float> local(static_cast<size_t>(n));
    const float scale = 1.f / static_cast<float>(perms.size());
    for (const auto& perm : perms) {
        std::fill(local.begin(), local.end(), 0.f);
        for (int i = 0; i < n; ++i) {
            shuffled[static_cast<size_t>(i)] = z[perm[static_cast<size_t>(i)]];
        }
        pair_penalty(shuffled.data(), n, nullptr, local.data());
        for (int i = 0; i < n; ++i) {
            grad[perm[static_cast<size_t>(i)]] += local[static_cast<size_t>(i)] * scale;
        }
    }
}

void geodesic_step(float* z, const float* grad, int64_t n, float angle) {
    const double radius2 = dot(z, z, n);
    if (radius2 <= 0.0) {
        return;
    }
    const double along = dot(grad, z, n) / radius2;
    std::vector<double> tangent(static_cast<size_t>(n));
    double tangent2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        tangent[static_cast<size_t>(i)] = static_cast<double>(grad[i]) - along * static_cast<double>(z[i]);
        tangent2 += tangent[static_cast<size_t>(i)] * tangent[static_cast<size_t>(i)];
    }
    if (tangent2 <= 1e-20) {
        return;
    }
    const double radius = std::sqrt(radius2);
    const double inv_tangent = 1.0 / std::sqrt(tangent2);
    const double cosine = std::cos(static_cast<double>(angle));
    const double sine = std::sin(static_cast<double>(angle));
    double next2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double value = cosine * static_cast<double>(z[i]) + sine * tangent[static_cast<size_t>(i)] * inv_tangent * radius;
        z[i] = static_cast<float>(value);
        next2 += value * value;
    }
    const double renorm = radius / std::sqrt(next2);
    for (int64_t i = 0; i < n; ++i) {
        z[i] = static_cast<float>(static_cast<double>(z[i]) * renorm);
    }
}

Tensor<float> copy_f32(ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        LOG_ERROR("flux2-dno expected a contiguous f32 graph tensor");
        return {};
    }
    const int64_t n = ggml_nelements(tensor);
    Tensor<float> out({tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}, std::vector<float>(static_cast<size_t>(n)));
    ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
    return out;
}

struct GraphScope {
    GGMLRunner* runner;
    DifferentiableOpsScope ops;
    explicit GraphScope(GGMLRunner* runner)
        : runner(runner), ops(true) {
        runner->set_graph_grads_enabled(true);
        runner->set_flash_attention_enabled(false);
        runner->set_conv2d_direct_enabled(false);
    }
    ~GraphScope() {
        runner->set_graph_grads_enabled(false);
    }
    GraphScope(const GraphScope&)            = delete;
    GraphScope& operator=(const GraphScope&) = delete;
};

struct EulerInputs {
    Flux::FluxRunner* flux = nullptr;
    int n_threads          = 1;
    const Tensor<float>* x = nullptr;
    float sigma            = 1.f;
    float sigma_next       = 0.f;
    const Tensor<float>* context  = nullptr;
    const Tensor<float>* guidance = nullptr;
    const Tensor<float>* adjoint  = nullptr;
};

struct EulerOutputs {
    Tensor<float> next;
    Tensor<float> grad;
};

bool euler_graph(const EulerInputs& in, EulerOutputs& out) {
    GraphScope scope(in.flux);
    Tensor<float> timesteps({1}, std::vector<float>{in.sigma});
    Tensor<float> guidance = in.guidance != nullptr ? *in.guidance : Tensor<float>{};
    ggml_tensor* next_tensor = nullptr;
    ggml_tensor* grad_tensor = nullptr;
    bool read_ok = false;
    auto get_graph = [&]() -> ggml_cgraph* {
        ggml_cgraph* graph = in.flux->build_graph(*in.x, timesteps, *in.context, {}, {}, guidance);
        if (graph == nullptr) {
            return nullptr;
        }
        ggml_context* ctx = in.flux->active_compute_ctx();
        ggml_tensor* pred = ggml_graph_get_tensor(graph, "dno_pred");
        ggml_tensor* x = ggml_graph_get_tensor(graph, "dno_x");
        if (pred == nullptr || x == nullptr) {
            LOG_ERROR("flux2-dno could not find the denoiser input or prediction");
            return nullptr;
        }
        if (pred->type != GGML_TYPE_F32) {
            pred = ggml_cast(ctx, pred, GGML_TYPE_F32);
        }
        const float dt = in.sigma_next - in.sigma;
        ggml_tensor* x_next = ggml_add(ctx, x, ggml_scale(ctx, pred, dt));
        ggml_set_name(x_next, "dno_x_next");
        if (in.adjoint != nullptr) {
            ggml_tensor* adjoint = in.flux->make_input(*in.adjoint);
            ggml_tensor* loss = ggml_sum(ctx, ggml_mul(ctx, x_next, adjoint));
            ggml_set_name(loss, "dno_loss");
            ggml_set_loss(loss);
            ggml_build_forward_expand(graph, loss);
            in.flux->append_backward(graph);
            grad_tensor = ggml_graph_get_grad(graph, x);
            if (grad_tensor == nullptr) {
                LOG_ERROR("flux2-dno backward did not produce an input gradient");
                return nullptr;
            }
            if (!ggml_is_contiguous(grad_tensor) || grad_tensor->type != GGML_TYPE_F32) {
                grad_tensor = ggml_cast(ctx, ggml_cont(ctx, grad_tensor), GGML_TYPE_F32);
                ggml_build_forward_expand(graph, grad_tensor);
            }
            ggml_set_output(grad_tensor);
        } else {
            ggml_build_forward_expand(graph, x_next);
        }
        if (!ggml_is_contiguous(x_next)) {
            x_next = ggml_cont(ctx, x_next);
            ggml_build_forward_expand(graph, x_next);
        }
        ggml_set_output(x_next);
        next_tensor = x_next;
        return graph;
    };
    auto read_outputs = [&]() {
        out.next = copy_f32(next_tensor);
        if (in.adjoint != nullptr) {
            out.grad = copy_f32(grad_tensor);
            read_ok = !out.next.empty() && !out.grad.empty();
        } else {
            read_ok = !out.next.empty();
        }
        return read_ok;
    };
    in.flux->GGMLRunner::compute(get_graph, in.n_threads, true, true, read_outputs);
    return read_ok;
}

Tensor<float> constant_tensor(const std::vector<int64_t>& shape, float value) {
    return Tensor<float>::full(shape, value);
}

bool vae_graph(AutoEncoderKL* vae,
               int n_threads,
               const Tensor<float>& latent,
               const Tensor<float>* adjoint,
               Tensor<float>& rgb,
               Tensor<float>& grad) {
    GraphScope scope(vae);
    ggml_tensor* rgb_tensor = nullptr;
    ggml_tensor* grad_tensor = nullptr;
    bool read_ok = false;
    auto get_graph = [&]() -> ggml_cgraph* {
        ggml_cgraph* graph = vae->build_graph(latent, true);
        if (graph == nullptr) {
            return nullptr;
        }
        ggml_context* ctx = vae->active_compute_ctx();
        ggml_tensor* decoded = ggml_graph_get_tensor(graph, "dno_rgb");
        ggml_tensor* z = ggml_graph_get_tensor(graph, "dno_z");
        if (decoded == nullptr || z == nullptr) {
            LOG_ERROR("flux2-dno could not find the VAE input or decode");
            return nullptr;
        }
        if (decoded->type != GGML_TYPE_F32) {
            decoded = ggml_cast(ctx, decoded, GGML_TYPE_F32);
        }
        if (adjoint != nullptr) {
            ggml_tensor* adjoint_tensor = vae->make_input(*adjoint);
            ggml_tensor* loss = ggml_sum(ctx, ggml_mul(ctx, decoded, adjoint_tensor));
            ggml_set_name(loss, "dno_vae_loss");
            ggml_set_loss(loss);
            ggml_build_forward_expand(graph, loss);
            vae->append_backward(graph);
            grad_tensor = ggml_graph_get_grad(graph, z);
            if (grad_tensor == nullptr) {
                LOG_ERROR("flux2-dno VAE backward did not produce a latent gradient");
                return nullptr;
            }
            if (!ggml_is_contiguous(grad_tensor) || grad_tensor->type != GGML_TYPE_F32) {
                grad_tensor = ggml_cast(ctx, ggml_cont(ctx, grad_tensor), GGML_TYPE_F32);
                ggml_build_forward_expand(graph, grad_tensor);
            }
            ggml_set_output(grad_tensor);
        } else {
            ggml_build_forward_expand(graph, decoded);
        }
        if (!ggml_is_contiguous(decoded)) {
            decoded = ggml_cont(ctx, decoded);
            ggml_build_forward_expand(graph, decoded);
        }
        ggml_set_output(decoded);
        rgb_tensor = decoded;
        return graph;
    };
    auto read_outputs = [&]() {
        rgb = copy_f32(rgb_tensor);
        if (adjoint != nullptr) {
            grad = copy_f32(grad_tensor);
            read_ok = !rgb.empty() && !grad.empty();
        } else {
            read_ok = !rgb.empty();
        }
        return read_ok;
    };
    vae->compute(get_graph, n_threads, true, true, read_outputs);
    return read_ok;
}

Tensor<float> diffusion_to_vae(const Tensor<float>& latent, const LatentStats& stats) {
    Tensor<float> converted = latent;
    const auto& shape = converted.shape();
    for (int64_t n = 0; n < shape[3]; ++n) {
        for (int64_t c = 0; c < shape[2]; ++c) {
            const float scale = stats.stddev[static_cast<size_t>(c)] / stats.scale;
            const float shift = stats.mean[static_cast<size_t>(c)];
            for (int64_t h = 0; h < shape[1]; ++h) {
                for (int64_t w = 0; w < shape[0]; ++w) {
                    float& value = converted.index(w, h, c, n);
                    value = value * scale + shift;
                }
            }
        }
    }
    return converted;
}

void vae_grad_to_diffusion(Tensor<float>& grad, const LatentStats& stats) {
    const auto& shape = grad.shape();
    for (int64_t n = 0; n < shape[3]; ++n) {
        for (int64_t c = 0; c < shape[2]; ++c) {
            const float scale = stats.stddev[static_cast<size_t>(c)] / stats.scale;
            for (int64_t h = 0; h < shape[1]; ++h) {
                for (int64_t w = 0; w < shape[0]; ++w) {
                    grad.index(w, h, c, n) *= scale;
                }
            }
        }
    }
}

float mean_pixel(const Tensor<float>& rgb) {
    double sum = 0.0;
    const int64_t n = rgb.numel();
    const float* data = rgb.data();
    for (int64_t i = 0; i < n; ++i) {
        sum += (static_cast<double>(data[i]) + 1.0) * 0.5;
    }
    return n > 0 ? static_cast<float>(sum / static_cast<double>(n)) : 0.f;
}

struct Rollout {
    Flux::FluxRunner* flux = nullptr;
    AutoEncoderKL* vae     = nullptr;
    LatentStats stats;
    int n_threads = 1;
    std::vector<float> sigmas;
    Tensor<float> context;
    Tensor<float> guidance;
    bool through_vae = false;
};

bool forward_chain(const Rollout& rollout, const Tensor<float>& start, std::vector<Tensor<float>>& states, Tensor<float>* rgb) {
    states.clear();
    states.push_back(start);
    for (size_t step = 0; step + 1 < rollout.sigmas.size(); ++step) {
        EulerInputs in;
        in.flux = rollout.flux;
        in.n_threads = rollout.n_threads;
        in.x = &states.back();
        in.sigma = rollout.sigmas[step];
        in.sigma_next = rollout.sigmas[step + 1];
        in.context = &rollout.context;
        in.guidance = rollout.guidance.empty() ? nullptr : &rollout.guidance;
        EulerOutputs out;
        LOG_INFO("flux2-dno forward step %zu/%zu", step + 1, rollout.sigmas.size() - 1);
        const auto started = std::chrono::steady_clock::now();
        if (!euler_graph(in, out)) {
            return false;
        }
        char phase[64];
        std::snprintf(phase, sizeof(phase), "forward step %zu", step + 1);
        log_graph_time(phase, started);
        states.push_back(std::move(out.next));
    }
    if (rgb == nullptr) {
        return true;
    }
    if (!rollout.through_vae) {
        return true;
    }
    Tensor<float> unused;
    const auto started = std::chrono::steady_clock::now();
    const bool ok = vae_graph(rollout.vae, rollout.n_threads, diffusion_to_vae(states.back(), rollout.stats), nullptr, *rgb, unused);
    log_graph_time("vae forward", started);
    return ok;
}

bool backward_chain(const Rollout& rollout, const std::vector<Tensor<float>>& states, Tensor<float>& grad) {
    if (rollout.through_vae) {
        Tensor<float> rgb;
        Tensor<float> vae_latent = diffusion_to_vae(states.back(), rollout.stats);
        Tensor<float> unused;
        const auto shape_started = std::chrono::steady_clock::now();
        if (!vae_graph(rollout.vae, rollout.n_threads, vae_latent, nullptr, rgb, unused)) {
            return false;
        }
        log_graph_time("vae forward", shape_started);
        const float adjoint_value = 0.5f / static_cast<float>(std::max<int64_t>(1, rgb.numel()));
        Tensor<float> adjoint = constant_tensor(rgb.shape(), adjoint_value);
        Tensor<float> vae_grad;
        LOG_INFO("flux2-dno VAE backward");
        const auto vae_started = std::chrono::steady_clock::now();
        if (!vae_graph(rollout.vae, rollout.n_threads, vae_latent, &adjoint, rgb, vae_grad)) {
            return false;
        }
        log_graph_time("vae backward", vae_started);
        vae_grad_to_diffusion(vae_grad, rollout.stats);
        grad = std::move(vae_grad);
    } else {
        grad = states.back() * 2.f;
    }
    for (size_t step = rollout.sigmas.size() - 1; step-- > 0;) {
        EulerInputs in;
        in.flux = rollout.flux;
        in.n_threads = rollout.n_threads;
        in.x = &states[step];
        in.sigma = rollout.sigmas[step];
        in.sigma_next = rollout.sigmas[step + 1];
        in.context = &rollout.context;
        in.guidance = rollout.guidance.empty() ? nullptr : &rollout.guidance;
        in.adjoint = &grad;
        EulerOutputs out;
        LOG_INFO("flux2-dno backward step %zu", step + 1);
        const auto started = std::chrono::steady_clock::now();
        if (!euler_graph(in, out)) {
            return false;
        }
        char phase[64];
        std::snprintf(phase, sizeof(phase), "backward step %zu", step + 1);
        log_graph_time(phase, started);
        double max_replay = 0.0;
        double grad_sq = 0.0;
        double skip_sq = 0.0;
        double network_sq = 0.0;
        const float* replay = out.next.data();
        const float* target = states[step + 1].data();
        const float* got = out.grad.data();
        for (int64_t i = 0; i < out.grad.numel(); ++i) {
            max_replay = std::max(max_replay, std::fabs(static_cast<double>(replay[i]) - target[i]));
            const double skip = 2.0 * static_cast<double>(target[i]);
            const double delta = static_cast<double>(got[i]) - skip;
            grad_sq += static_cast<double>(got[i]) * got[i];
            skip_sq += skip * skip;
            network_sq += delta * delta;
        }
        LOG_INFO("flux2-dno replay max abs %.6g grad norm %.6g skip norm %.6g network norm %.6g",
                 max_replay, std::sqrt(grad_sq), std::sqrt(skip_sq), std::sqrt(network_sq));
        grad = std::move(out.grad);
    }
    return true;
}

double scalar_loss(const Rollout& rollout, const Tensor<float>& start) {
    std::vector<Tensor<float>> states;
    Tensor<float> rgb;
    if (!forward_chain(rollout, start, states, rollout.through_vae ? &rgb : nullptr)) {
        return NAN;
    }
    if (rollout.through_vae) {
        return mean_pixel(rgb);
    }
    double sum = 0.0;
    const float* data = states.back().data();
    for (int64_t i = 0; i < states.back().numel(); ++i) {
        const double value = data[i];
        sum += value * value;
    }
    return sum;
}

bool finite_difference_check(const Rollout& rollout, const Tensor<float>& start, const Tensor<float>& analytic, const Options& options) {
    const int64_t n = start.numel();
    const int points = std::max(1, std::min(options.check_points, static_cast<int>(n)));
    double grad_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        grad_sq += static_cast<double>(analytic.data()[i]) * analytic.data()[i];
    }
    // Float32 central differences on a loss of a few thousand do not resolve a
    // component far below the gradient's rms. Keep the relative test, and only
    // lift the denominator up to 5% of that rms.
    const float floor = std::max(1e-4f, 0.05f * static_cast<float>(std::sqrt(grad_sq / static_cast<double>(n))));
    float worst = 0.f;
    for (int point = 0; point < points; ++point) {
        const int64_t index = (n * point) / points;
        Tensor<float> probe = start;
        probe.data()[index] += options.check_eps;
        const double plus = scalar_loss(rollout, probe);
        probe.data()[index] = start.data()[index] - options.check_eps;
        const double minus = scalar_loss(rollout, probe);
        if (!std::isfinite(plus) || !std::isfinite(minus)) {
            LOG_ERROR("flux2-dno finite difference failed");
            return false;
        }
        const float fd = static_cast<float>((plus - minus) / (2.0 * static_cast<double>(options.check_eps)));
        const float got = analytic.data()[index];
        const float denom = std::max(std::max(std::fabs(fd), std::fabs(got)), floor);
        const float rel = std::fabs(fd - got) / denom;
        worst = std::max(worst, rel);
        LOG_INFO("flux2-dno grad[%lld] analytic %.6g finite-difference %.6g relative %.4g loss %.8g %.8g",
                 static_cast<long long>(index), got, fd, rel, plus, minus);
    }
    if (worst > options.check_tol) {
        LOG_ERROR("flux2-dno gradient check failed, worst relative error %.4g", worst);
        return false;
    }
    LOG_INFO("flux2-dno gradient check passed, worst relative error %.4g", worst);
    return true;
}

Tensor<float> unit_interval(const Tensor<float>& rgb) {
    Tensor<float> scaled = rgb;
    for (int64_t i = 0; i < scaled.numel(); ++i) {
        const float value = (scaled.data()[i] + 1.f) * 0.5f;
        scaled.data()[i] = std::min(1.f, std::max(0.f, value));
    }
    return scaled;
}

struct Tape {
    ggml_context* ctx = nullptr;
    explicit Tape(size_t bytes) {
        ggml_init_params params = {};
        params.mem_size = bytes;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        ctx = ggml_init(params);
    }
    ~Tape() {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;
};

bool check_tape(const char* name, ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* x, ggml_tensor* y, int max_points = -1) {
    ggml_set_param(x);
    ggml_tensor* loss = ggml_sum(ctx, ggml_sqr(ctx, y));
    ggml_set_loss(loss);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_tensor* grad = ggml_graph_get_grad(graph, x);
    if (grad == nullptr) {
        LOG_ERROR("flux2-dno primitive %s has no input gradient", name);
        return false;
    }
    if (!ggml_is_contiguous(grad) || grad->type != GGML_TYPE_F32) {
        grad = ggml_cont(ctx, grad);
        if (grad->type != GGML_TYPE_F32) {
            grad = ggml_cast(ctx, grad, GGML_TYPE_F32);
        }
        ggml_build_forward_expand(graph, grad);
    }
    ggml_graph_reset(graph);
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("flux2-dno primitive %s compute failed", name);
        return false;
    }
    const int64_t n = ggml_nelements(x);
    std::vector<float> analytic(static_cast<size_t>(n));
    memcpy(analytic.data(), grad->data, sizeof(float) * static_cast<size_t>(n));
    const float eps = 1e-3f;
    float worst = 0.f;
    float* values = static_cast<float*>(x->data);
    const int64_t ncheck = max_points > 0 && max_points < n ? max_points : n;
    for (int64_t point = 0; point < ncheck; ++point) {
        const int64_t i = ncheck == n ? point : (n * point) / ncheck;
        const float saved = values[i];
        values[i] = saved + eps;
        ggml_graph_reset(graph);
        if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) {
            LOG_ERROR("flux2-dno primitive %s finite difference failed", name);
            return false;
        }
        const float plus = static_cast<float*>(loss->data)[0];
        values[i] = saved - eps;
        ggml_graph_reset(graph);
        if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) {
            LOG_ERROR("flux2-dno primitive %s finite difference failed", name);
            return false;
        }
        const float minus = static_cast<float*>(loss->data)[0];
        values[i] = saved;
        const float fd = (plus - minus) / (2.f * eps);
        const float denom = std::max(std::max(std::fabs(fd), std::fabs(analytic[static_cast<size_t>(i)])), 1e-4f);
        const float rel = std::fabs(fd - analytic[static_cast<size_t>(i)]) / denom;
        if (i < 6 || rel > 0.005f) {
            LOG_INFO("flux2-dno primitive %s [%lld] analytic %.6g finite-difference %.6g relative %.4g",
                     name, static_cast<long long>(i), analytic[static_cast<size_t>(i)], fd, rel);
        }
        worst = std::max(worst, rel);
    }
    if (worst > 3e-2f) {
        LOG_ERROR("flux2-dno primitive %s failed, worst relative error %.4g", name, worst);
        return false;
    }
    LOG_INFO("flux2-dno primitive %s passed, worst relative error %.4g", name, worst);
    return true;
}

bool check_matmul_tape() {
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor* w = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 4, 3);
    float* xv = static_cast<float*>(x->data);
    float* wv = static_cast<float*>(w->data);
    for (int i = 0; i < 4; ++i) {
        xv[i] = 0.25f * static_cast<float>(i - 1);
    }
    for (int i = 0; i < 12; ++i) {
        wv[i] = 0.1f * static_cast<float>(i - 5);
    }
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 64, true);
    return check_tape("matmul", tape.ctx, graph, x, ggml_mul_mat(tape.ctx, w, x));
}

bool check_split_view_tape() {
    Tape tape(16 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_1d(tape.ctx, GGML_TYPE_F32, 8);
    float* xv = static_cast<float*>(x->data);
    for (int i = 0; i < 8; ++i) {
        xv[i] = 0.3f * static_cast<float>(i - 3);
    }
    ggml_tensor* second = ggml_view_1d(tape.ctx, x, 4, 4 * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 64, true);
    return check_tape("split-view", tape.ctx, graph, x, second);
}

bool check_quantized_matmul_tape() {
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t k = 256;
    constexpr int64_t rows = 4;
    ggml_quantize_init(GGML_TYPE_Q2_K);
    std::vector<float> weights(static_cast<size_t>(k * rows));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = 0.02f * static_cast<float>(static_cast<int>(i % 17) - 8);
    }
    ggml_tensor* w = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_Q2_K, k, rows);
    ggml_quantize_chunk(GGML_TYPE_Q2_K, weights.data(), w->data, 0, rows, k, nullptr);
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, k, 1);
    float* xv = static_cast<float*>(x->data);
    for (int64_t i = 0; i < k; ++i) {
        xv[i] = 0.01f * static_cast<float>(i % 11 - 5);
    }
    const bool previous_exact = ggml_set_exact_quant_matmul(true);
    ggml_tensor* y = ggml_mul_mat(tape.ctx, w, x);
    ggml_set_exact_quant_matmul(previous_exact);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 128, true);
    return check_tape("q2-matmul", tape.ctx, graph, x, y);
}

bool check_bf16_matmul_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t k = 64;
    constexpr int64_t rows = 8;
    constexpr int64_t tokens = 4;
    std::vector<float> weights(static_cast<size_t>(k * rows));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = 0.15f * static_cast<float>(static_cast<int>(i % 9) - 4);
    }
    ggml_tensor* w = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_BF16, k, rows);
    std::vector<ggml_bf16_t> packed(weights.size());
    ggml_fp32_to_bf16_row(weights.data(), packed.data(), static_cast<int64_t>(weights.size()));
    memcpy(w->data, packed.data(), packed.size() * sizeof(ggml_bf16_t));
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, k, tokens);
    float* xv = static_cast<float*>(x->data);
    for (int64_t i = 0; i < k * tokens; ++i) {
        xv[i] = 0.2f * static_cast<float>(static_cast<int>(i % 7) - 3);
    }
    ggml_tensor* y = ggml_mul_mat(tape.ctx, w, x);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("bf16-matmul", tape.ctx, graph, x, y);
}

bool check_euler_skip_tape() {
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor* w = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 4, 4);
    float* xv = static_cast<float*>(x->data);
    float* wv = static_cast<float*>(w->data);
    for (int i = 0; i < 4; ++i) {
        xv[i] = 0.4f * static_cast<float>(i - 2);
    }
    for (int i = 0; i < 16; ++i) {
        wv[i] = 0.05f * static_cast<float>(i - 7);
    }
    ggml_tensor* pred = ggml_mul_mat(tape.ctx, w, x);
    ggml_tensor* next = ggml_add(tape.ctx, x, ggml_scale(tape.ctx, pred, -1.f));
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 128, true);
    return check_tape("euler-skip", tape.ctx, graph, x, next);
}

bool check_rms_tape() {
    Tape tape(16 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor* w = ggml_new_tensor_1d(tape.ctx, GGML_TYPE_F32, 8);
    float* xv = static_cast<float*>(x->data);
    float* wv = static_cast<float*>(w->data);
    for (int i = 0; i < 32; ++i) {
        xv[i] = 0.15f * static_cast<float>((i % 7) - 3);
    }
    for (int i = 0; i < 8; ++i) {
        wv[i] = 0.2f * static_cast<float>(i + 1);
    }
    ggml_tensor* y = ggml_mul(tape.ctx, ggml_rms_norm(tape.ctx, x, 1e-6f), w);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("rms-norm", tape.ctx, graph, x, y);
}

bool check_both_sides_tape() {
    Tape tape(16 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* a = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor* b = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 4, 2);
    float* av = static_cast<float*>(a->data);
    float* bv = static_cast<float*>(b->data);
    for (int i = 0; i < 12; ++i) {
        av[i] = 0.1f * static_cast<float>(i - 4);
    }
    for (int i = 0; i < 8; ++i) {
        bv[i] = 0.2f * static_cast<float>(i - 2);
    }
    ggml_tensor* y = ggml_mul_mat(tape.ctx, a, b);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 128, true);
    if (!check_tape("matmul-src0", tape.ctx, graph, a, y)) {
        return false;
    }
    graph = ggml_new_graph_custom(tape.ctx, 128, true);
    return check_tape("matmul-src1", tape.ctx, graph, b, y);
}

bool check_attention_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(64 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t channels = 16;
    constexpr int64_t tokens = 4;
    constexpr int64_t heads = 2;
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, channels, tokens);
    ggml_tensor* wq = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, channels, channels);
    ggml_tensor* wk = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, channels, channels);
    ggml_tensor* wv = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, channels, channels);
    float* xv = static_cast<float*>(x->data);
    float* wqv = static_cast<float*>(wq->data);
    float* wkv = static_cast<float*>(wk->data);
    float* wvv = static_cast<float*>(wv->data);
    for (int i = 0; i < channels * tokens; ++i) {
        xv[i] = 0.05f * static_cast<float>((i % 9) - 4);
    }
    for (int i = 0; i < channels * channels; ++i) {
        wqv[i] = 0.03f * static_cast<float>((i % 11) - 5);
        wkv[i] = 0.04f * static_cast<float>((i % 7) - 3);
        wvv[i] = 0.02f * static_cast<float>((i % 5) - 2);
    }
    ggml_tensor* q = ggml_mul_mat(tape.ctx, wq, x);
    ggml_tensor* k = ggml_mul_mat(tape.ctx, wk, x);
    ggml_tensor* v = ggml_mul_mat(tape.ctx, wv, x);
    ggml_tensor* y = ggml_ext_attention_ext(tape.ctx, nullptr, q, k, v, heads, nullptr, false, false, 1.f);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 1024, true);
    return check_tape("attention", tape.ctx, graph, x, y);
}

bool check_rope_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(64 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t d_head = 128;
    constexpr int64_t n_head = 2;
    constexpr int64_t tokens = 6;
    ggml_tensor* x = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, n_head, tokens, 1);
    ggml_tensor* pe = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, 2, 2, d_head / 2, tokens);
    float* xv = static_cast<float*>(x->data);
    float* pv = static_cast<float*>(pe->data);
    for (int i = 0; i < d_head * n_head * tokens; ++i) {
        xv[i] = 0.1f * static_cast<float>((i % 8) - 3);
    }
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t f = 0; f < d_head / 2; ++f) {
            const float angle = 0.3f * static_cast<float>(t + 1) / static_cast<float>(f + 1);
            const int64_t base = (f * tokens + t) * 4;
            pv[base + 0] = std::cos(angle);
            pv[base + 1] = -std::sin(angle);
            pv[base + 2] = std::sin(angle);
            pv[base + 3] = std::cos(angle);
        }
    }
    ggml_tensor* y = Rope::apply_rope(tape.ctx, x, pe, true);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 4096, true);
    return check_tape("rope", tape.ctx, graph, x, y);
}

bool check_strided_view_tape() {
    Tape tape(16 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 2, 4);
    float* xv = static_cast<float*>(x->data);
    for (int i = 0; i < 8; ++i) {
        xv[i] = 0.2f * static_cast<float>(i - 2);
    }
    ggml_tensor* parent = ggml_view_4d(tape.ctx, x, 2, 2, 1, 1, 2 * x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor* child = ggml_view_4d(tape.ctx, parent, 2, 1, 1, 1, parent->nb[1], parent->nb[2], parent->nb[3], parent->nb[1]);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 64, true);
    return check_tape("strided-view", tape.ctx, graph, x, child);
}

}  // namespace

bool check_swiglu_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 16, 4);
    float* xv = static_cast<float*>(x->data);
    for (int i = 0; i < 64; ++i) {
        xv[i] = 0.2f * static_cast<float>((i % 9) - 4);
    }
    ggml_tensor* y = ggml_ext_silu_act(tape.ctx, x, false);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("swiglu", tape.ctx, graph, x, y);
}

bool check_patch_tape() {
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, 4, 4, 128, 1);
    float* xv = static_cast<float*>(x->data);
    for (int i = 0; i < 4 * 4 * 128; ++i) {
        xv[i] = 0.1f * static_cast<float>((i % 11) - 5);
    }
    ggml_tensor* tokens = DiT::patchify(tape.ctx, x, 1, 1, true);
    ggml_tensor* y = DiT::unpatchify(tape.ctx, tokens, 4, 4, 1, 1, true);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("patch", tape.ctx, graph, x, y);
}

void fill_f32(ggml_tensor* tensor, float scale, int period, int center) {
    float* values = static_cast<float*>(tensor->data);
    const int64_t n = ggml_nelements(tensor);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = scale * static_cast<float>(static_cast<int>(i % period) - center);
    }
}

void fill_params(GGMLBlock& block) {
    std::map<std::string, ggml_tensor*> tensors;
    block.get_param_tensors(tensors);
    int which = 0;
    for (auto& item : tensors) {
        fill_f32(item.second, 0.05f + 0.01f * static_cast<float>(which % 4), 9, 4);
        if (item.first.find("norm") != std::string::npos || item.first.find("scale") != std::string::npos) {
            float* values = static_cast<float*>(item.second->data);
            const int64_t n = ggml_nelements(item.second);
            for (int64_t i = 0; i < n; ++i) {
                values[i] = 0.8f + 0.05f * static_cast<float>(i % 5);
            }
        }
        which++;
    }
}

void fill_rotation(ggml_tensor* pe) {
    float* values = static_cast<float*>(pe->data);
    const int64_t bands = pe->ne[2];
    const int64_t tokens = pe->ne[3];
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t band = 0; band < bands; ++band) {
            const float angle = 0.2f * static_cast<float>(token + 1) / static_cast<float>(band + 1);
            float* cell = values + band * (pe->nb[2] / sizeof(float)) + token * (pe->nb[3] / sizeof(float));
            cell[0] = std::cos(angle);
            cell[1] = -std::sin(angle);
            cell[2] = std::sin(angle);
            cell[3] = std::cos(angle);
        }
    }
}

GGMLRunnerContext local_runner(ggml_context* ctx) {
    GGMLRunnerContext runner;
    runner.ggml_ctx = ctx;
    runner.flash_attn_enabled = false;
    runner.set_backend_tensor_data = [](ggml_tensor* tensor, const void* data) {
        memcpy(tensor->data, data, ggml_nbytes(tensor));
    };
    return runner;
}

bool check_layer_norm_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, 16, 4, 1);
    ggml_tensor* w = ggml_new_tensor_1d(tape.ctx, GGML_TYPE_F32, 16);
    fill_f32(x, 0.2f, 9, 4);
    for (int i = 0; i < 16; ++i) {
        static_cast<float*>(w->data)[i] = 0.25f * static_cast<float>(i + 1);
    }
    GGMLRunnerContext runner = local_runner(tape.ctx);
    ggml_tensor* y = ggml_mul(tape.ctx, differentiable_layer_norm(tape.ctx, x, nullptr, nullptr, 1e-6f, runner.set_backend_tensor_data), w);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("layer-norm", tape.ctx, graph, x, y);
}

bool check_slice_linear_tape() {
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t hidden = 16;
    constexpr int64_t n_txt = 2;
    constexpr int64_t n_img = 4;
    ggml_tensor* parent = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_txt + n_img, 1);
    ggml_tensor* weight = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, hidden, hidden);
    fill_f32(parent, 0.15f, 9, 4);
    fill_f32(weight, 0.05f, 7, 3);
    ggml_tensor* img = ggml_view_3d(tape.ctx, parent, hidden, n_img, 1, parent->nb[1], parent->nb[2], n_txt * parent->nb[1]);
    ggml_tensor* y = ggml_mul_mat(tape.ctx, weight, img);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("slice-linear", tape.ctx, graph, parent, y);
}

bool check_concat_seq_tape() {
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t d_head = 8;
    constexpr int64_t heads = 2;
    constexpr int64_t n_txt = 2;
    constexpr int64_t n_img = 4;
    ggml_tensor* txt = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_txt, 1);
    ggml_tensor* img = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_img, 1);
    fill_f32(txt, 0.1f, 7, 3);
    fill_f32(img, 0.2f, 9, 4);
    ggml_tensor* joined = ggml_concat(tape.ctx, txt, img, 2);
    ggml_tensor* y = ggml_view_4d(tape.ctx, joined, d_head, heads, n_img, 1, joined->nb[1], joined->nb[2], joined->nb[3], n_txt * joined->nb[2]);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("concat-seq", tape.ctx, graph, img, y);
}

bool check_modulate_tape() {
    Tape tape(32 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, 16, 4, 1);
    ggml_tensor* shift = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 16, 1);
    ggml_tensor* scale = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, 16, 1);
    fill_f32(x, 0.2f, 9, 4);
    fill_f32(shift, 0.1f, 5, 2);
    fill_f32(scale, 0.15f, 7, 3);
    ggml_tensor* y = Flux::modulate(tape.ctx, x, shift, scale);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 256, true);
    return check_tape("modulate", tape.ctx, graph, x, y);
}

bool check_qkv_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(64 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t hidden = 16;
    constexpr int64_t heads = 2;
    constexpr int64_t tokens = 4;
    Flux::SelfAttention attention(hidden, heads, false, false);
    attention.init(tape.ctx);
    fill_params(attention);
    ggml_tensor* x = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, tokens, 1);
    ggml_tensor* pe = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, 2, 2, hidden / heads / 2, tokens);
    fill_f32(x, 0.15f, 9, 4);
    fill_rotation(pe);
    GGMLRunnerContext runner = local_runner(tape.ctx);
    ggml_tensor* y = attention.forward(&runner, x, pe, nullptr);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 8192, true);
    return check_tape("flux-qkv", tape.ctx, graph, x, y);
}

bool check_flux_join_tape() {
    DifferentiableOpsScope ops(true);
    bool ok = true;
    for (int mode = 0; mode < 3; ++mode) {
        Tape tape(64 * 1024 * 1024);
        if (tape.ctx == nullptr) {
            return false;
        }
        constexpr int64_t hidden = 16;
        constexpr int64_t heads = 2;
        constexpr int64_t n_img = 4;
        constexpr int64_t n_txt = 2;
        Flux::SelfAttention img_attn(hidden, heads, false, false);
        Flux::SelfAttention txt_attn(hidden, heads, false, false);
        img_attn.init(tape.ctx);
        txt_attn.init(tape.ctx);
        fill_params(img_attn);
        fill_params(txt_attn);
        ggml_tensor* img = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_img, 1);
        ggml_tensor* txt = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_txt, 1);
        ggml_tensor* pe = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, 2, 2, hidden / heads / 2, n_txt + n_img);
        fill_f32(img, 0.12f, 11, 5);
        fill_f32(txt, 0.08f, 7, 3);
        fill_rotation(pe);
        GGMLRunnerContext runner = local_runner(tape.ctx);
        std::vector<ggml_tensor*> img_qkv = img_attn.pre_attention(&runner, img);
        std::vector<ggml_tensor*> txt_qkv = txt_attn.pre_attention(&runner, txt);
        const char* name = "flux-join";
        ggml_tensor* param = img;
        ggml_tensor* q = ggml_concat(tape.ctx, txt_qkv[0], img_qkv[0], 2);
        ggml_tensor* y = nullptr;
        if (mode == 0) {
            const int64_t d_head = hidden / heads;
            ggml_tensor* img_v = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_img, 1);
            ggml_tensor* txt_q = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_txt, 1);
            ggml_tensor* txt_k = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_txt, 1);
            ggml_tensor* txt_v = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_txt, 1);
            ggml_tensor* img_q = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_img, 1);
            ggml_tensor* img_k = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, d_head, heads, n_img, 1);
            fill_f32(img_v, 0.13f, 11, 5);
            fill_f32(txt_q, 0.08f, 7, 3);
            fill_f32(txt_k, 0.07f, 5, 1);
            fill_f32(txt_v, 0.06f, 9, 4);
            fill_f32(img_q, 0.12f, 11, 5);
            fill_f32(img_k, 0.09f, 5, 2);
            ggml_tensor* attn = Rope::attention(&runner,
                                                ggml_concat(tape.ctx, txt_q, img_q, 2),
                                                ggml_concat(tape.ctx, txt_k, img_k, 2),
                                                ggml_concat(tape.ctx, txt_v, img_v, 2),
                                                pe,
                                                nullptr);
            y = ggml_view_3d(tape.ctx, attn, attn->ne[0], n_img, attn->ne[2], attn->nb[1], attn->nb[2], n_txt * attn->nb[1]);
            param = img_v;
            name = "join-v";
        } else if (mode == 1) {
            ggml_tensor* seq = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_txt + n_img, 1);
            fill_f32(seq, 0.12f, 11, 5);
            ggml_tensor* attn = img_attn.forward(&runner, seq, pe, nullptr);
            y = ggml_view_3d(tape.ctx, attn, attn->ne[0], n_img, attn->ne[2], attn->nb[1], attn->nb[2], n_txt * attn->nb[1]);
            param = seq;
            name = "attn-tail";
        } else {
            ggml_tensor* k = ggml_concat(tape.ctx, txt_qkv[1], img_qkv[1], 2);
            ggml_tensor* v = ggml_concat(tape.ctx, txt_qkv[2], img_qkv[2], 2);
            ggml_tensor* attn = Rope::attention(&runner, q, k, v, pe, nullptr);
            y = ggml_view_3d(tape.ctx, attn, attn->ne[0], n_img, attn->ne[2], attn->nb[1], attn->nb[2], n_txt * attn->nb[1]);
            name = "join-attn";
        }
        ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 16384, true);
        ok = check_tape(name, tape.ctx, graph, param, y) && ok;
    }
    return ok;
}

bool check_flux_double_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(128 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t hidden = 16;
    constexpr int64_t heads = 2;
    constexpr int64_t n_img = 4;
    constexpr int64_t n_txt = 2;
    Flux::DoubleStreamBlock block(hidden, heads, 2.f, 0, false, false, true, false, false, true);
    Flux::Modulation img_mod(hidden, true, false);
    Flux::Modulation txt_mod(hidden, true, false);
    block.init(tape.ctx);
    img_mod.init(tape.ctx);
    txt_mod.init(tape.ctx);
    fill_params(block);
    fill_params(img_mod);
    fill_params(txt_mod);
    ggml_tensor* img = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_img, 1);
    ggml_tensor* txt = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_txt, 1);
    ggml_tensor* vec = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor* pe = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, 2, 2, hidden / heads / 2, n_txt + n_img);
    fill_f32(img, 0.12f, 9, 4);
    fill_f32(txt, 0.08f, 7, 3);
    fill_f32(vec, 0.05f, 5, 2);
    fill_rotation(pe);
    GGMLRunnerContext runner = local_runner(tape.ctx);
    auto img_mods = img_mod.forward(&runner, vec);
    auto txt_mods = txt_mod.forward(&runner, vec);
    Flux::StreamForward out = block.forward(&runner, img, txt, vec, pe, nullptr, img_mods, txt_mods);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 32768, true);
    return check_tape("flux-double", tape.ctx, graph, img, out.img);
}

bool check_flux_single_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(128 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t hidden = 16;
    constexpr int64_t heads = 2;
    constexpr int64_t n_img = 4;
    constexpr int64_t n_txt = 2;
    Flux::SingleStreamBlock block(hidden, heads, 2.f, 0, 0.f, false, true, false, false, true);
    Flux::Modulation modulation(hidden, false, false);
    block.init(tape.ctx);
    modulation.init(tape.ctx);
    fill_params(block);
    fill_params(modulation);
    ggml_tensor* img = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_img, 1);
    ggml_tensor* txt = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, hidden, n_txt, 1);
    ggml_tensor* vec = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor* pe = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, 2, 2, hidden / heads / 2, n_txt + n_img);
    fill_f32(img, 0.12f, 9, 4);
    fill_f32(txt, 0.08f, 7, 3);
    fill_f32(vec, 0.05f, 5, 2);
    fill_rotation(pe);
    GGMLRunnerContext runner = local_runner(tape.ctx);
    ggml_tensor* joined = ggml_concat(tape.ctx, txt, img, 1);
    joined = block.forward(&runner, joined, vec, pe, nullptr, modulation.forward(&runner, vec));
    ggml_tensor* img_out = ggml_view_3d(tape.ctx,
                                        joined,
                                        joined->ne[0],
                                        n_img,
                                        joined->ne[2],
                                        joined->nb[1],
                                        joined->nb[2],
                                        n_txt * joined->nb[1]);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 32768, true);
    return check_tape("flux-single", tape.ctx, graph, img, img_out);
}

bool check_flux_step_tape() {
    DifferentiableOpsScope ops(true);
    Tape tape(256 * 1024 * 1024);
    if (tape.ctx == nullptr) {
        return false;
    }
    constexpr int64_t width = 2;
    constexpr int64_t height = 2;
    constexpr int64_t channels = 8;
    constexpr int64_t hidden = 16;
    constexpr int64_t heads = 2;
    constexpr int64_t n_txt = 2;
    constexpr int64_t txt_dim = 8;
    constexpr int64_t n_img = width * height;
    Linear img_in(channels, hidden, false);
    Linear txt_in(txt_dim, hidden, false);
    Flux::Modulation img_mod(hidden, true, false);
    Flux::Modulation txt_mod(hidden, true, false);
    Flux::Modulation single_mod(hidden, false, false);
    Flux::DoubleStreamBlock double_block(hidden, heads, 2.f, 0, false, false, true, false, false, true);
    Flux::SingleStreamBlock single_block(hidden, heads, 2.f, 0, 0.f, false, true, false, false, true);
    Flux::LastLayer last(hidden, 1, channels, false, false);
    img_in.init(tape.ctx);
    txt_in.init(tape.ctx);
    img_mod.init(tape.ctx);
    txt_mod.init(tape.ctx);
    single_mod.init(tape.ctx);
    double_block.init(tape.ctx);
    single_block.init(tape.ctx);
    last.init(tape.ctx);
    fill_params(img_in);
    fill_params(txt_in);
    fill_params(img_mod);
    fill_params(txt_mod);
    fill_params(single_mod);
    fill_params(double_block);
    fill_params(single_block);
    fill_params(last);
    ggml_tensor* x = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, width, height, channels, 1);
    ggml_tensor* txt = ggml_new_tensor_3d(tape.ctx, GGML_TYPE_F32, txt_dim, n_txt, 1);
    ggml_tensor* vec = ggml_new_tensor_2d(tape.ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor* pe = ggml_new_tensor_4d(tape.ctx, GGML_TYPE_F32, 2, 2, hidden / heads / 2, n_txt + n_img);
    fill_f32(x, 0.1f, 11, 5);
    fill_f32(txt, 0.08f, 7, 3);
    fill_f32(vec, 0.05f, 5, 2);
    fill_rotation(pe);
    GGMLRunnerContext runner = local_runner(tape.ctx);
    ggml_tensor* img = DiT::patchify(tape.ctx, x, 1, 1, true);
    img = img_in.forward(&runner, img);
    ggml_tensor* text = txt_in.forward(&runner, txt);
    Flux::StreamForward streamed = double_block.forward(&runner,
                                                        img,
                                                        text,
                                                        vec,
                                                        pe,
                                                        nullptr,
                                                        img_mod.forward(&runner, vec),
                                                        txt_mod.forward(&runner, vec));
    ggml_tensor* joined = ggml_concat(tape.ctx, streamed.txt, streamed.img, 1);
    joined = single_block.forward(&runner, joined, vec, pe, nullptr, single_mod.forward(&runner, vec));
    img = ggml_view_3d(tape.ctx, joined, joined->ne[0], n_img, joined->ne[2], joined->nb[1], joined->nb[2], n_txt * joined->nb[1]);
    img = last.forward(&runner, img, vec);
    ggml_tensor* y = DiT::unpatchify(tape.ctx, img, height, width, 1, 1, true);
    ggml_cgraph* graph = ggml_new_graph_custom(tape.ctx, 65536, true);
    return check_tape("flux-step", tape.ctx, graph, x, y);
}

bool primitive_grad_matches_finite_difference() {
    const bool base = check_matmul_tape() && check_quantized_matmul_tape() && check_bf16_matmul_tape() && check_euler_skip_tape() &&
                       check_split_view_tape() && check_strided_view_tape();
    const bool rms = check_rms_tape();
    const bool both = check_both_sides_tape();
    const bool attention = check_attention_tape();
    const bool rope = check_rope_tape();
    const bool swiglu = check_swiglu_tape();
    const bool patch = check_patch_tape();
    const bool layer = check_layer_norm_tape();
    const bool slice = check_slice_linear_tape();
    const bool concat_seq = check_concat_seq_tape();
    const bool modulate = check_modulate_tape();
    const bool qkv = check_qkv_tape();
    const bool join = check_flux_join_tape();
    const bool flux_double = check_flux_double_tape();
    const bool flux_single = check_flux_single_tape();
    const bool flux_step = check_flux_step_tape();
    return base && rms && both && attention && rope && swiglu && patch && layer && slice && concat_seq && modulate && qkv &&
           join && flux_double && flux_single && flux_step;
}

bool penalty_grad_matches_finite_difference() {
    constexpr int n = 8;
    std::vector<float> z(n, 4.f);
    z[1] = -3.f;
    z[4] = 5.f;
    const auto perms = make_permutations(n, 3, 7);
    std::vector<float> grad(n, 0.f);
    penalty_gradient(z.data(), n, perms, grad.data());
    const float eps = 1e-3f;
    float worst = 0.f;
    for (int i = 0; i < n; ++i) {
        const float saved = z[static_cast<size_t>(i)];
        z[static_cast<size_t>(i)] = saved + eps;
        const double plus = penalty_value(z.data(), n, perms);
        z[static_cast<size_t>(i)] = saved - eps;
        const double minus = penalty_value(z.data(), n, perms);
        z[static_cast<size_t>(i)] = saved;
        const float fd = static_cast<float>((plus - minus) / (2.0 * eps));
        const float denom = std::max(std::max(std::fabs(fd), std::fabs(grad[static_cast<size_t>(i)])), 1e-4f);
        worst = std::max(worst, std::fabs(fd - grad[static_cast<size_t>(i)]) / denom);
    }
    if (worst > 1e-2f) {
        LOG_ERROR("flux2-dno penalty gradient check failed, worst relative error %.4g", worst);
        return false;
    }
    LOG_INFO("flux2-dno penalty gradient check passed, worst relative error %.4g", worst);
    return true;
}

struct OutProdDiff {
    float max_abs = 0.f;
    double rel_rms = 0.0;
    double cosine = 0.0;
};

OutProdDiff compare_out_prod(const float* got, const float* ref, int64_t n) {
    OutProdDiff diff;
    double dot = 0.0;
    double got_sq = 0.0;
    double ref_sq = 0.0;
    double err_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double g = got[i];
        const double r = ref[i];
        const double err = g - r;
        diff.max_abs = std::max(diff.max_abs, static_cast<float>(std::fabs(err)));
        dot += g * r;
        got_sq += g * g;
        ref_sq += r * r;
        err_sq += err * err;
    }
    const double denom = std::sqrt(got_sq * ref_sq);
    diff.cosine = denom > 0.0 ? dot / denom : 0.0;
    const double ref_rms = std::sqrt(ref_sq / static_cast<double>(std::max<int64_t>(n, 1)));
    diff.rel_rms = ref_rms > 0.0 ? std::sqrt(err_sq / static_cast<double>(n)) / ref_rms : 0.0;
    return diff;
}

void fused_q2_out_prod(const void* weight, int64_t k, int64_t m, int64_t tokens, const float* grad, float* dst, int n_threads) {
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q2_K);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q2_K, k);
    const size_t block_bytes = ggml_row_size(GGML_TYPE_Q2_K, 256);
    const int64_t n_blocks = k / 256;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(n_threads));
    for (int ith = 0; ith < n_threads; ++ith) {
        const int64_t blocks_per = (n_blocks + n_threads - 1) / n_threads;
        const int64_t block0 = static_cast<int64_t>(ith) * blocks_per;
        const int64_t block1 = std::min(block0 + blocks_per, n_blocks);
        threads.emplace_back([=]() {
            if (block0 >= block1) {
                return;
            }
            const int64_t k0 = block0 * 256;
            const int64_t k_len = (block1 - block0) * 256;
            constexpr int64_t kRows = 64;
            std::vector<float> tile(static_cast<size_t>(kRows * k_len));
            for (int64_t token = 0; token < tokens; ++token) {
                std::fill(dst + token * k + k0, dst + token * k + k0 + k_len, 0.f);
            }
            for (int64_t row0 = 0; row0 < m; row0 += kRows) {
                const int64_t rows = std::min(kRows, m - row0);
                for (int64_t row = 0; row < rows; ++row) {
                    const char* slice = static_cast<const char*>(weight) + (row0 + row) * row_bytes + block0 * block_bytes;
                    traits->to_float(slice, tile.data() + row * k_len, k_len);
                }
                for (int64_t token = 0; token < tokens; ++token) {
                    float* out = dst + token * k + k0;
                    for (int64_t row = 0; row < rows; ++row) {
                        const float scale = grad[(row0 + row) + token * m];
                        const float* decoded = tile.data() + row * k_len;
                        for (int64_t col = 0; col < k_len; ++col) {
                            out[col] += decoded[col] * scale;
                        }
                    }
                }
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

bool time_out_prod(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* dst, int n_threads, const float* reference, const char* name) {
    const int64_t n = ggml_nelements(dst);
    std::vector<float> got(static_cast<size_t>(n));
    const auto warmup_start = std::chrono::steady_clock::now();
    if (ggml_graph_compute_with_ctx(ctx, graph, n_threads) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("flux2-dno out-prod bench %s failed", name);
        return false;
    }
    const double warmup = std::chrono::duration<double>(std::chrono::steady_clock::now() - warmup_start).count();
    const auto timed_start = std::chrono::steady_clock::now();
    if (ggml_graph_compute_with_ctx(ctx, graph, n_threads) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("flux2-dno out-prod bench %s failed", name);
        return false;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - timed_start).count();
    memcpy(got.data(), dst->data, sizeof(float) * static_cast<size_t>(n));
    const OutProdDiff diff = compare_out_prod(got.data(), reference, n);
    LOG_INFO("flux2-dno out-prod %s warmup %.2f s repeat %.2f s max abs %.4g rel rms %.4g cosine %.8f",
             name, warmup, seconds, diff.max_abs, diff.rel_rms, diff.cosine);
    return true;
}

bool bench_out_prod_case(int64_t k, int64_t m, int64_t tokens, int n_threads, bool report_time) {
    if (k % 256 != 0 || tokens <= 0 || m <= 0) {
        return false;
    }
    ggml_quantize_init(GGML_TYPE_Q2_K);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q2_K, k);
    const size_t bytes = row_bytes * static_cast<size_t>(m) +
                         sizeof(float) * static_cast<size_t>(k * m + m * tokens + 2 * k * tokens) +
                         64ull * 1024ull * 1024ull;
    ggml_init_params params = {};
    params.mem_size = bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        LOG_ERROR("flux2-dno out-prod bench could not allocate %.2f MB", bytes / (1024.0 * 1024.0));
        return false;
    }
    struct Release {
        ggml_context* ctx;
        ~Release() { ggml_free(ctx); }
    } release{ctx};

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_K, k, m);
    ggml_tensor* grad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, tokens);
    std::vector<float> row(static_cast<size_t>(k));
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t c = 0; c < k; ++c) {
            row[static_cast<size_t>(c)] = 0.01f * static_cast<float>(static_cast<int>((r * 17 + c * 3) % 100) - 50);
        }
        ggml_quantize_chunk(GGML_TYPE_Q2_K, row.data(), static_cast<char*>(weight->data) + r * static_cast<int64_t>(row_bytes), 0, 1, k, nullptr);
    }
    float* grad_data = static_cast<float*>(grad->data);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t r = 0; r < m; ++r) {
            grad_data[r + token * m] = 0.02f * static_cast<float>(static_cast<int>((r * 5 + token * 9) % 50) - 25);
        }
    }

    ggml_tensor* cast_dst = ggml_out_prod(ctx, ggml_cast(ctx, weight, GGML_TYPE_F32), ggml_transpose(ctx, grad));
    ggml_tensor* quant_dst = ggml_out_prod(ctx, weight, ggml_transpose(ctx, grad));
    ggml_cgraph* cast_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph* quant_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(cast_graph, cast_dst);
    ggml_build_forward_expand(quant_graph, quant_dst);

    const int64_t n = k * tokens;
    std::vector<float> reference(static_cast<size_t>(n));
    if (ggml_graph_compute_with_ctx(ctx, cast_graph, n_threads) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("flux2-dno out-prod bench cast failed");
        return false;
    }
    memcpy(reference.data(), cast_dst->data, sizeof(float) * static_cast<size_t>(n));

    std::vector<float> fused(static_cast<size_t>(n));
    fused_q2_out_prod(weight->data, k, m, tokens, grad_data, fused.data(), n_threads);
    const OutProdDiff fused_diff = compare_out_prod(fused.data(), reference.data(), n);
    if (!report_time) {
        LOG_INFO("flux2-dno out-prod small K=%lld M=%lld T=%lld max abs %.4g rel rms %.4g",
                 static_cast<long long>(k), static_cast<long long>(m), static_cast<long long>(tokens),
                 fused_diff.max_abs, fused_diff.rel_rms);
        return fused_diff.max_abs < 1e-4f;
    }

    const auto fused_timed = std::chrono::steady_clock::now();
    fused_q2_out_prod(weight->data, k, m, tokens, grad_data, fused.data(), n_threads);
    const double fused_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fused_timed).count();
    LOG_INFO("flux2-dno out-prod fused %.2f s max abs %.4g rel rms %.4g cosine %.8f",
             fused_s, fused_diff.max_abs, fused_diff.rel_rms, fused_diff.cosine);
    if (!time_out_prod(ctx, cast_graph, cast_dst, n_threads, reference.data(), "cast") ||
        !time_out_prod(ctx, quant_graph, quant_dst, n_threads, reference.data(), "quant")) {
        return false;
    }
    return fused_diff.rel_rms < 1e-3 && fused_diff.cosine > 0.9999;
}

bool bench_out_prod() {
    constexpr int kThreads = 12;
    LOG_INFO("flux2-dno out-prod bench, %d threads", kThreads);
    if (!bench_out_prod_case(512, 8, 4, kThreads, false)) {
        LOG_ERROR("flux2-dno out-prod bench small case disagreed with the cast");
        return false;
    }
    const bool ok = bench_out_prod_case(6144, 55296, 528, kThreads, true);
    if (!ok) {
        LOG_ERROR("flux2-dno out-prod bench disagreed with the cast");
    }
    return ok;
}

bool run(StableDiffusionGGML* sd, const Options& options, const ImageWriter& write_image) {
    if (!penalty_grad_matches_finite_difference()) {
        return false;
    }
    if (options.check_penalty) {
        return true;
    }
    if (sd == nullptr || !sd_version_is_flux2(sd->version)) {
        LOG_ERROR("flux2-dno requires a Flux 2 dev model");
        return false;
    }
    if (options.steps < 1 || options.width % 16 != 0 || options.height % 16 != 0) {
        LOG_ERROR("flux2-dno needs a positive step count and dimensions divisible by 16");
        return false;
    }
    auto flux = std::dynamic_pointer_cast<Flux::FluxRunner>(sd->diffusion_model);
    auto vae = std::dynamic_pointer_cast<AutoEncoderKL>(sd->first_stage_model);
    if (!flux || !vae || !sd->cond_stage_model) {
        LOG_ERROR("flux2-dno could not find the Flux 2 denoiser or VAE");
        return false;
    }
    flux->set_flash_attention_enabled(false);
    vae->set_flash_attention_enabled(false);
    flux->set_conv2d_direct_enabled(false);
    vae->set_conv2d_direct_enabled(false);

    ConditionerParams condition_params;
    condition_params.text = options.prompt;
    condition_params.width = options.width;
    condition_params.height = options.height;
    SDCondition condition = sd->cond_stage_model->get_learned_condition(sd->n_threads, condition_params);
    sd->cond_stage_model->runner_end();
    if (condition.c_crossattn.empty()) {
        LOG_ERROR("flux2-dno failed to encode the prompt");
        return false;
    }

    Rollout rollout;
    rollout.flux = flux.get();
    rollout.vae = vae.get();
    rollout.n_threads = sd->n_threads;
    rollout.context = condition.c_crossattn;
    const int64_t text_tokens = rollout.context.shape().size() > 1 ? rollout.context.shape()[1] : 0;
    const int64_t image_tokens = static_cast<int64_t>(options.width / 16) * static_cast<int64_t>(options.height / 16);
    LOG_INFO("flux2-dno context shape %s, text tokens %lld, image tokens %lld, single-stream tokens %lld",
             tensor_shape_to_string(rollout.context.shape()).c_str(),
             static_cast<long long>(text_tokens),
             static_cast<long long>(image_tokens),
             static_cast<long long>(text_tokens + image_tokens));
    rollout.through_vae = !options.check_grad || options.through_vae;
    if (flux->config.guidance_embed) {
        rollout.guidance = Tensor<float>({1}, std::vector<float>{options.guidance});
    }
    const Tensor<float> probe = sd->generate_init_latent(options.width, options.height, 1, false);
    auto stats_pair = vae->get_latents_mean_std(probe, 2);
    rollout.stats.mean.assign(stats_pair.first.data(), stats_pair.first.data() + stats_pair.first.numel());
    rollout.stats.stddev.assign(stats_pair.second.data(), stats_pair.second.data() + stats_pair.second.numel());
    rollout.stats.scale = vae->scale_factor;

    Flux2Scheduler scheduler(sd->get_image_seq_len(options.height, options.width));
    rollout.sigmas = scheduler.get_sigmas(static_cast<uint32_t>(options.steps), 0.f, 1.f, {});
    if (rollout.sigmas.size() != static_cast<size_t>(options.steps) + 1) {
        LOG_ERROR("flux2-dno scheduler returned an unexpected sigma list");
        return false;
    }
    LOG_INFO("flux2-dno sigmas %.4f -> %.4f over %d Euler steps, cfg 1",
             rollout.sigmas.front(), rollout.sigmas.back(), options.steps);

    auto rng = std::make_shared<STDDefaultRNG>();
    rng->manual_seed(options.seed);
    Tensor<float> noise = Tensor<float>::randn(probe.shape(), rng);
    const auto perms = make_permutations(static_cast<int>(noise.numel()), std::max(1, options.permutations), options.seed + 17);

    if (options.check_grad) {
        rollout.through_vae = options.through_vae;
        std::vector<Tensor<float>> states;
        if (!forward_chain(rollout, noise, states, nullptr)) {
            return false;
        }
        Tensor<float> analytic;
        if (!backward_chain(rollout, states, analytic)) {
            return false;
        }
        return finite_difference_check(rollout, noise, analytic, options);
    }

    rollout.through_vae = true;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        std::vector<Tensor<float>> states;
        Tensor<float> rgb;
        if (!forward_chain(rollout, noise, states, &rgb)) {
            return false;
        }
        if (write_image && !write_image(iteration, unit_interval(rgb))) {
            return false;
        }
        Tensor<float> grad;
        if (!backward_chain(rollout, states, grad)) {
            return false;
        }
        if (options.gamma != 0.f) {
            std::vector<float> penalty(static_cast<size_t>(noise.numel()), 0.f);
            penalty_gradient(noise.data(), static_cast<int>(noise.numel()), perms, penalty.data());
            for (int64_t i = 0; i < noise.numel(); ++i) {
                grad.data()[i] += options.gamma * penalty[static_cast<size_t>(i)];
            }
        }
        LOG_INFO("flux2-dno iteration %d mean pixel %.5f", iteration, mean_pixel(rgb));
        geodesic_step(noise.data(), grad.data(), noise.numel(), options.angle);
    }
    return true;
}

}  // namespace flux2_dno
}  // namespace sd
