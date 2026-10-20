#include "runtime/zit_dno.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "conditioning/conditioner.hpp"
#include "core/grad_safe.h"
#include "core/rng.hpp"
#include "ggml-cpu.h"
#include "model/diffusion/z_image.hpp"
#include "model/vae/vae.hpp"
#include "pipeline/diffusion_engine.h"
#include "runtime/denoiser.hpp"

namespace sd {
namespace zit_dno {

Tensor<float> copy_f32(ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        LOG_ERROR("zit-dno expected a contiguous f32 graph tensor");
        return {};
    }
    const int64_t n = ggml_nelements(tensor);
    Tensor<float> out({tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}, std::vector<float>(static_cast<size_t>(n)));
    ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
    return out;
}

double sum_squares(const Tensor<float>& tensor) {
    double sum = 0.0;
    const float* data = tensor.data();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        const double value = data[i];
        sum += value * value;
    }
    return sum;
}

double dot(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

float log_probability_bound(double bound) {
    if (bound >= 1.0) {
        return 0.f;
    }
    return static_cast<float>(std::log(std::max(bound, 1e-30)));
}

// Host Tang penalty and geodesic step, same math as flux2-dno.
constexpr int kPair = 2;

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

// shortcut: a view aliases its parent, so only contiguous adjoints are cuts; one block must still fit beside the resident weights.
void mark_activation_grad_cuts(ggml_cgraph* graph) {
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (!sd::ggml_graph_cut::is_graph_cut_tensor(node)) {
            continue;
        }
        const char* payload = node->name + std::strlen(sd::ggml_graph_cut::GGML_RUNNER_CUT_PREFIX);
        if (std::strncmp(payload, "bwd.", 4) == 0) {
            continue;
        }
        std::string body(payload);
        if (!body.empty() && body.back() == '|') {
            body.pop_back();
        }
        const size_t sep = body.find('|');
        if (sep == std::string::npos || body.substr(sep + 1) == "t_emb") {
            continue;
        }
        ggml_tensor* grad = ggml_graph_get_grad(graph, node);
        if (grad == nullptr || grad->view_src != nullptr || !ggml_is_contiguous(grad) ||
            sd::ggml_graph_cut::is_graph_cut_tensor(grad)) {
            continue;
        }
        std::string group = "bwd." + body;
        std::replace(group.begin(), group.end(), '|', '.');
        sd::ggml_graph_cut::mark_graph_cut(grad, group, "g");
        if (!sd::ggml_graph_cut::is_graph_cut_tensor(grad)) {
            LOG_ERROR("zit-dno could not name an activation adjoint cut");
        }
    }
}

struct GraphScope {
    GGMLRunner* runner;
    DifferentiableOpsScope ops;
    explicit GraphScope(GGMLRunner* runner)
        : runner(runner), ops(true) {
        runner->set_graph_grads_enabled(true);
        runner->set_flash_attention_enabled(false);
    }
    ~GraphScope() {
        runner->set_graph_grads_enabled(false);
    }
    GraphScope(const GraphScope&)            = delete;
    GraphScope& operator=(const GraphScope&) = delete;
};

// x_next = x + pred * (sigma_next - sigma). The sampler applies c_out = -sigma first,
// which cancels the /sigma in the Euler update, so the raw model output is the velocity.
bool flow_step(ZImage::ZImageRunner* model,
               int n_threads,
               const Tensor<float>& x,
               const Tensor<float>& context,
               const std::vector<float>& timesteps,
               float dt,
               const Tensor<float>* adjoint,
               Tensor<float>& next,
               Tensor<float>& grad) {
    GraphScope scope(model);
    // make_input keeps this pointer until compute copies it. A temporary would be gone by then.
    const Tensor<float> step_time({static_cast<int64_t>(timesteps.size())}, timesteps);
    ggml_tensor* next_tensor = nullptr;
    ggml_tensor* grad_tensor = nullptr;
    bool read_ok             = false;
    auto get_graph = [&]() -> ggml_cgraph* {
        ggml_cgraph* graph = model->build_graph(x, step_time, context);
        if (graph == nullptr) {
            return nullptr;
        }
        ggml_context* ctx  = model->active_compute_ctx();
        ggml_tensor* pred  = ggml_graph_get_tensor(graph, "dno_pred");
        ggml_tensor* input = ggml_graph_get_tensor(graph, "dno_x");
        if (pred == nullptr || input == nullptr) {
            LOG_ERROR("zit-dno could not find the denoiser input or prediction");
            return nullptr;
        }
        if (pred->type != GGML_TYPE_F32) {
            pred = ggml_cast(ctx, pred, GGML_TYPE_F32);
        }
        ggml_tensor* x_next = ggml_add(ctx, input, ggml_scale(ctx, pred, dt));
        ggml_set_name(x_next, "dno_x_next");
        if (adjoint != nullptr) {
            ggml_tensor* loss = ggml_sum(ctx, ggml_mul(ctx, x_next, model->make_input(*adjoint)));
            ggml_set_name(loss, "dno_loss");
            ggml_set_loss(loss);
            ggml_build_forward_expand(graph, loss);
            model->append_backward(graph);
            // Layer adjoints become cuts so a tape that does not fit runs one block at a time.
            mark_activation_grad_cuts(graph);
            grad_tensor = ggml_graph_get_grad(graph, input);
            if (grad_tensor == nullptr) {
                LOG_ERROR("zit-dno backward did not produce an input gradient");
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
        next = copy_f32(next_tensor);
        if (adjoint != nullptr) {
            grad    = copy_f32(grad_tensor);
            read_ok = !next.empty() && !grad.empty();
        } else {
            read_ok = !next.empty();
        }
        return read_ok;
    };
    model->GGMLRunner::compute(get_graph, n_threads, true, true, read_outputs);
    return read_ok;
}

bool rollout(ZImage::ZImageRunner* model,
             StableDiffusionGGML* sd,
             int n_threads,
             const Tensor<float>& start,
             const Tensor<float>& context,
             const std::vector<float>& sigmas,
             int n_steps,
             std::vector<Tensor<float>>& states) {
    states.clear();
    states.push_back(start);
    for (int step = 0; step < n_steps; ++step) {
        const std::vector<float> timesteps = sd->prepare_sample_timesteps(sigmas[static_cast<size_t>(step)], 0);
        Tensor<float> next;
        Tensor<float> unused;
        if (!flow_step(model,
                       n_threads,
                       states.back(),
                       context,
                       timesteps,
                       sigmas[static_cast<size_t>(step + 1)] - sigmas[static_cast<size_t>(step)],
                       nullptr,
                       next,
                       unused)) {
            return false;
        }
        states.push_back(std::move(next));
    }
    return true;
}

bool backward_latent(ZImage::ZImageRunner* model,
                    StableDiffusionGGML* sd,
                    int n_threads,
                    const Tensor<float>& context,
                    const std::vector<float>& sigmas,
                    int n_steps,
                    int backward_steps,
                    const std::vector<Tensor<float>>& states,
                    Tensor<float>& grad,
                    bool stabilize,
                    double* adjoint_scale) {
    Tensor<float> adjoint = states.back();
    float* adjoint_data   = adjoint.data();
    for (int64_t i = 0; i < adjoint.numel(); ++i) {
        adjoint_data[i] *= 2.f;
    }
    // One step already reaches about 1e17. The map is linear in the adjoint,
    // so the next step overflows float32 products. Dividing a large adjoint
    // by its peak keeps the direction and leaves headroom. adjoint_scale is
    // that divisor, so a host penalty can be added back in the original units.
    double scale = 1.0;
    const int first = n_steps - backward_steps;
    for (int step = n_steps - 1; step >= first; --step) {
        if (stabilize) {
            float* data          = adjoint.data();
            const int64_t count  = adjoint.numel();
            float peak           = 0.f;
            for (int64_t i = 0; i < count; ++i) {
                peak = std::max(peak, std::fabs(data[i]));
            }
            if (peak > 1.0e4f) {
                const float inv = 1.f / peak;
                for (int64_t i = 0; i < count; ++i) {
                    data[i] *= inv;
                }
                scale *= static_cast<double>(peak);
            }
        }
        const std::vector<float> timesteps = sd->prepare_sample_timesteps(sigmas[static_cast<size_t>(step)], 0);
        Tensor<float> next;
        Tensor<float> step_grad;
        if (!flow_step(model,
                       n_threads,
                       states[static_cast<size_t>(step)],
                       context,
                       timesteps,
                       sigmas[static_cast<size_t>(step + 1)] - sigmas[static_cast<size_t>(step)],
                       &adjoint,
                       next,
                       step_grad)) {
            return false;
        }
        const float* step_grad_data = step_grad.data();
        for (int64_t i = 0; i < step_grad.numel(); ++i) {
            if (!std::isfinite(step_grad_data[i])) {
                LOG_ERROR("zit-dno backward step %d gradient is not finite", step);
                return false;
            }
        }
        adjoint = std::move(step_grad);
    }
    if (adjoint_scale != nullptr) {
        *adjoint_scale = scale;
    }
    grad = std::move(adjoint);
    return true;
}

bool check_chain(ZImage::ZImageRunner* model,
                 StableDiffusionGGML* sd,
                 int n_threads,
                 const Tensor<float>& start,
                 const Tensor<float>& context,
                 const std::vector<float>& sigmas,
                 int n_steps) {
    // shortcut: more than one CPU thread disagrees with this tape, keep the check on one thread until a multi-thread run matches.
    n_threads = 1;
    LOG_INFO("zit-dno gradient check, %d flow steps, 1 thread, eps 1e-2, tol 0.2", n_steps);
    const auto started = std::chrono::steady_clock::now();
    std::vector<Tensor<float>> states;
    if (!rollout(model, sd, n_threads, start, context, sigmas, n_steps, states)) {
        return false;
    }
    Tensor<float> adjoint;
    if (!backward_latent(model, sd, n_threads, context, sigmas, n_steps, n_steps, states, adjoint, false, nullptr)) {
        return false;
    }
    const double backward_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    LOG_INFO("zit-dno chain %d backward %.2f s, final latent loss %.8g",
             n_steps,
             backward_seconds,
             sum_squares(states.back()));

    const int64_t n     = start.numel();
    const int points    = std::min<int>(4, static_cast<int>(n));
    double grad_sq      = 0.0;
    const float* values = adjoint.data();
    for (int64_t i = 0; i < n; ++i) {
        grad_sq += static_cast<double>(values[i]) * values[i];
    }
    const float floor = std::max(1e-4f, 0.05f * static_cast<float>(std::sqrt(grad_sq / static_cast<double>(n))));
    const float eps   = 1e-2f;
    float worst       = 0.f;
    for (int point = 0; point < points; ++point) {
        const int64_t index = (n * point) / points;
        Tensor<float> probe = start;
        probe.data()[index] += eps;
        std::vector<Tensor<float>> plus_states;
        if (!rollout(model, sd, n_threads, probe, context, sigmas, n_steps, plus_states)) {
            return false;
        }
        const double plus    = sum_squares(plus_states.back());
        probe.data()[index] = start.data()[index] - eps;
        std::vector<Tensor<float>> minus_states;
        if (!rollout(model, sd, n_threads, probe, context, sigmas, n_steps, minus_states)) {
            return false;
        }
        const double minus = sum_squares(minus_states.back());
        if (!std::isfinite(plus) || !std::isfinite(minus)) {
            LOG_ERROR("zit-dno finite difference failed");
            return false;
        }
        const float fd    = static_cast<float>((plus - minus) / (2.0 * static_cast<double>(eps)));
        const float got   = adjoint.data()[index];
        const float denom = std::max(std::max(std::fabs(fd), std::fabs(got)), floor);
        const float rel   = std::fabs(fd - got) / denom;
        worst             = std::max(worst, rel);
        LOG_INFO("zit-dno steps %d grad[%lld] analytic %.6g finite-difference %.6g relative %.4g",
                 n_steps,
                 static_cast<long long>(index),
                 got,
                 fd,
                 rel);
    }
    if (worst > 0.2f) {
        LOG_ERROR("zit-dno gradient check failed, worst relative error %.4g", worst);
        return false;
    }
    LOG_INFO("zit-dno gradient check passed, %d steps, worst relative error %.4g", n_steps, worst);
    return true;
}

bool iterate_once(ZImage::ZImageRunner* model,
                  StableDiffusionGGML* sd,
                  const Tensor<float>& context,
                  const std::vector<float>& sigmas,
                  Tensor<float>& noise,
                  uint64_t seed,
                  int backward_steps,
                  int reforward_steps,
                  int n_threads,
                  const ImageWriter& write_image) {
    const int n_steps = static_cast<int>(sigmas.size()) - 1;
    if (backward_steps <= 0) {
        backward_steps = n_steps;
    }
    if (reforward_steps <= 0) {
        reforward_steps = backward_steps;
    }
    if (backward_steps > n_steps) {
        LOG_ERROR("zit-dno backward steps %d exceed the %d flow steps", backward_steps, n_steps);
        return false;
    }
    if (reforward_steps > backward_steps) {
        LOG_ERROR("zit-dno reforward steps %d exceed the %d backward steps", reforward_steps, backward_steps);
        return false;
    }
    const int resume = n_steps - backward_steps;
    std::vector<Tensor<float>> states;
    if (!rollout(model, sd, n_threads, noise, context, sigmas, n_steps, states)) {
        return false;
    }
    if (!sd->first_stage_model) {
        LOG_ERROR("zit-dno needs the VAE to write the image");
        return false;
    }
    sd->first_stage_model->set_flash_attention_enabled(false);
    sd->n_threads = n_threads;
    const Tensor<float> rgb  = sd->decode_first_stage(states.back());
    if (rgb.empty()) {
        LOG_ERROR("zit-dno VAE decode failed");
        return false;
    }
    if (!write_image || !write_image(0, rgb)) {
        return false;
    }

    Tensor<float> grad;
    double adjoint_scale = 1.0;
    if (!backward_latent(model, sd, n_threads, context, sigmas, n_steps, backward_steps, states, grad, true, &adjoint_scale)) {
        return false;
    }
    Tensor<float> moved = states[static_cast<size_t>(resume)];
    constexpr float kGamma = 0.01f;
    constexpr float kAngle = 0.03f;
    constexpr int kPerms   = 100;
    const auto perms = make_permutations(static_cast<int>(moved.numel()), kPerms, seed + 17);
    std::vector<float> penalty(static_cast<size_t>(moved.numel()), 0.f);
    penalty_gradient(moved.data(), static_cast<int>(moved.numel()), perms, penalty.data());
    const double penalty_gain = static_cast<double>(kGamma) / adjoint_scale;
    for (int64_t i = 0; i < moved.numel(); ++i) {
        grad.data()[i] += static_cast<float>(penalty_gain * static_cast<double>(penalty[static_cast<size_t>(i)]));
    }

    const double radius_before = std::sqrt(dot(moved.data(), moved.data(), moved.numel()));
    std::vector<float> before(moved.data(), moved.data() + moved.numel());
    geodesic_step(moved.data(), grad.data(), moved.numel(), kAngle);
    const double radius_after = std::sqrt(dot(moved.data(), moved.data(), moved.numel()));
    if (radius_before <= 0.0 || std::fabs(radius_after - radius_before) / radius_before > 1e-4) {
        LOG_ERROR("zit-dno geodesic step changed the latent radius");
        return false;
    }
    double shifted = 0.0;
    for (int64_t i = 0; i < moved.numel(); ++i) {
        shifted = std::max(shifted, std::fabs(static_cast<double>(moved.data()[i]) - before[static_cast<size_t>(i)]));
    }
    LOG_INFO("zit-dno iteration 0 loss %.8g sigma %.4f backward %d reforward %d step max abs %.6g radius %.6g adjoint scale %.6g",
             sum_squares(states.back()),
             sigmas[static_cast<size_t>(resume)],
             backward_steps,
             reforward_steps,
             shifted,
             radius_after,
             adjoint_scale);
    if (!(shifted > 0.0)) {
        return false;
    }

    const std::vector<float> tail(sigmas.begin() + resume, sigmas.begin() + resume + reforward_steps + 1);
    std::vector<Tensor<float>> moved_states;
    if (!rollout(model, sd, n_threads, moved, context, tail, reforward_steps, moved_states)) {
        return false;
    }
    const Tensor<float> moved_rgb = sd->decode_first_stage(moved_states.back());
    if (moved_rgb.empty()) {
        LOG_ERROR("zit-dno VAE decode failed");
        return false;
    }
    LOG_INFO("zit-dno iteration 1 loss %.8g", sum_squares(moved_states.back()));
    return write_image(1, moved_rgb);
}

bool run(StableDiffusionGGML* sd, const Options& options, const ImageWriter& write_image) {
    if (sd == nullptr || !sd_version_is_z_image(sd->version)) {
        LOG_ERROR("zit-dno requires a Z-Image model");
        return false;
    }
    if (options.steps < 1 || options.width % 16 != 0 || options.height % 16 != 0) {
        LOG_ERROR("zit-dno needs a positive step count and dimensions divisible by 16");
        return false;
    }
    if (options.prompt.empty()) {
        LOG_ERROR("zit-dno needs a prompt");
        return false;
    }
    auto zimage = std::dynamic_pointer_cast<ZImage::ZImageRunner>(sd->diffusion_model);
    if (!zimage || !sd->cond_stage_model || !sd->denoiser) {
        LOG_ERROR("zit-dno could not find the Z-Image denoiser");
        return false;
    }
    zimage->set_flash_attention_enabled(false);

    ConditionerParams condition_params;
    condition_params.text   = options.prompt;
    condition_params.width  = options.width;
    condition_params.height = options.height;
    SDCondition condition = sd->cond_stage_model->get_learned_condition(sd->n_threads, condition_params);
    sd->cond_stage_model->runner_end();
    if (condition.c_crossattn.empty()) {
        LOG_ERROR("zit-dno failed to encode the prompt");
        return false;
    }
    const int64_t text_tokens = condition.c_crossattn.shape().size() > 1 ? condition.c_crossattn.shape()[1] : 0;
    LOG_INFO("zit-dno context shape %s, text tokens %lld",
             tensor_shape_to_string(condition.c_crossattn.shape()).c_str(),
             static_cast<long long>(text_tokens));

    sd->set_flow_shift(INFINITY);
    const Tensor<float> latent = sd->generate_init_latent(options.width, options.height, 1, false);
    if (latent.empty()) {
        LOG_ERROR("zit-dno failed to build the initial latent");
        return false;
    }
    const int64_t image_tokens = (latent.shape()[0] / zimage->config.patch_size) * (latent.shape()[1] / zimage->config.patch_size);
    LOG_INFO("zit-dno latent shape %s, image tokens %lld, joined tokens %lld",
             tensor_shape_to_string(latent.shape()).c_str(),
             static_cast<long long>(image_tokens),
             static_cast<long long>(text_tokens + image_tokens));

    auto rng = std::make_shared<STDDefaultRNG>();
    rng->manual_seed(options.seed);
    Tensor<float> noise = Tensor<float>::randn(latent.shape(), rng);
    const std::vector<float> sigmas = sd->denoiser->get_sigmas(static_cast<uint32_t>(options.steps),
                                                               sd->get_image_seq_len(options.height, options.width),
                                                               DISCRETE_SCHEDULER,
                                                               sd->version);
    if (sigmas.size() < 2) {
        LOG_ERROR("zit-dno scheduler returned no step");
        return false;
    }
    const std::vector<float> timesteps = sd->prepare_sample_timesteps(sigmas.front(), 0);
    LOG_INFO("zit-dno sigma %.4f timestep %.1f, schedule length %zu",
             sigmas.front(), timesteps.empty() ? 0.f : timesteps.front(), sigmas.size());

    if (options.check_grad) {
        const int full = static_cast<int>(sigmas.size()) - 1;
        if (!check_chain(zimage.get(), sd, sd->n_threads, noise, condition.c_crossattn, sigmas, 1)) {
            return false;
        }
        if (full >= 2 && !check_chain(zimage.get(), sd, sd->n_threads, noise, condition.c_crossattn, sigmas, 2)) {
            return false;
        }
        if (full > 2 && !check_chain(zimage.get(), sd, sd->n_threads, noise, condition.c_crossattn, sigmas, full)) {
            return false;
        }
        return true;
    }

    // shortcut: more than one CPU thread disagrees with this tape, keep the step on one thread until a multi-thread run matches.
    const int n_threads = options.experimental_full_multi ? std::max(sd->n_threads, 1) : 1;
    LOG_INFO("zit-dno diffusion threads %d", n_threads);
    sd->n_threads = n_threads;
    return iterate_once(zimage.get(), sd, condition.c_crossattn, sigmas, noise, options.seed, options.backward_steps, options.reforward_steps, n_threads, write_image);
}

}  // namespace zit_dno
}  // namespace sd
