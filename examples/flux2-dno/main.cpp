#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "common/common.h"
#include "common/media_io.h"
#undef LOG_DEBUG
#undef LOG_VERBOSE
#undef LOG_INFO
#undef LOG_WARN
#undef LOG_ERROR
#include "pipeline/diffusion_engine.h"
#include "runtime/flux2_dno.hpp"
#include "stable-diffusion.h"

namespace fs = std::filesystem;

void flux2_dno_log(enum sd_log_level_t level, const char* log, void*) {
    log_print(level, log, SD_LOG_DEBUG, false);
}

int main(int argc, const char** argv) {
    sd_set_log_callback(flux2_dno_log, nullptr);
    sd::flux2_dno::Options options;
    SDContextParams ctx_params;
    sd_log_level_t log_level = SD_LOG_INFO;
    int seed                 = 0;
    ArgOptions extra;
    add_log_options(extra, log_level);
    extra.string_options.push_back({"", "--prompt", "prompt", 0, &options.prompt});
    extra.string_options.push_back({"", "--out", "output directory", 0, &options.out_dir});
    extra.int_options.push_back({"", "--steps", "Euler steps", &options.steps});
    extra.int_options.push_back({"", "--iters", "optimization iterations", &options.iterations});
    extra.int_options.push_back({"", "--width", "pixel width, multiple of 16", &options.width});
    extra.int_options.push_back({"", "--height", "pixel height, multiple of 16", &options.height});
    extra.int_options.push_back({"", "--perms", "Tang permutations", &options.permutations});
    extra.int_options.push_back({"", "--seed", "noise seed", &seed});
    extra.float_options.push_back({"", "--angle", "geodesic angle in radians", &options.angle});
    extra.float_options.push_back({"", "--gamma", "Tang penalty weight", &options.gamma});
    extra.float_options.push_back({"", "--guidance", "distilled guidance, if the model has it", &options.guidance});
    extra.float_options.push_back({"", "--check-eps", "finite-difference step", &options.check_eps});
    extra.float_options.push_back({"", "--check-tol", "relative gradient tolerance", &options.check_tol});
    extra.bool_options.push_back({"", "--check-grad", "compare the tape with finite differences and exit", true, &options.check_grad});
    extra.bool_options.push_back({"", "--through-vae", "include Flux 2 VAE decode in --check-grad", true, &options.through_vae});
    extra.bool_options.push_back({"", "--check-penalty", "check the host penalty gradient and exit", true, &options.check_penalty});
    extra.bool_options.push_back({"", "--bench-out-prod", "time one Q2_K activation gradient and exit", true, &options.bench_out_prod});

    std::vector<ArgOptions> option_list = {extra, ctx_params.get_options()};
    if (argc < 2) {
        std::cout << "usage: flux2-dno [options]\n\n";
        option_list[0].print();
        std::cout << "\n";
        option_list[1].print();
        return 1;
    }
    if (!parse_options(argc, argv, option_list)) {
        std::cout << "usage: flux2-dno [options]\n\n";
        option_list[0].print();
        return 1;
    }
    if (options.bench_out_prod) {
        return sd::flux2_dno::bench_out_prod() ? 0 : 1;
    }
    if (options.check_penalty) {
        const bool penalty = sd::flux2_dno::penalty_grad_matches_finite_difference();
        const bool primitive = sd::flux2_dno::primitive_grad_matches_finite_difference();
        return penalty && primitive ? 0 : 1;
    }
    if (!sd::flux2_dno::primitive_grad_matches_finite_difference()) {
        return 1;
    }
    ctx_params.flash_attn = false;
    ctx_params.diffusion_flash_attn = false;
    ctx_params.diffusion_conv_direct = false;
    ctx_params.vae_conv_direct = false;
    options.seed = static_cast<uint64_t>(seed);
    if (!ctx_params.resolve_and_validate(IMG_GEN)) {
        return 1;
    }
    sd_ctx_params_t params = ctx_params.to_sd_ctx_params_t(false);
    params.flash_attn = false;
    params.diffusion_flash_attn = false;

    auto model = std::make_unique<StableDiffusionGGML>();
    if (!model->init(&params)) {
        LOG_ERROR("flux2-dno failed to load the model");
        return 1;
    }
    const bool ok = sd::flux2_dno::run(model.get(), options, [&](int iteration, const sd::Tensor<float>& rgb) {
        fs::create_directories(options.out_dir);
        const std::string path = (fs::path(options.out_dir) / ("iter_" + std::to_string(iteration) + ".png")).string();
        sd_image_t image = tensor_to_sd_image(rgb);
        const bool wrote = write_image_to_file(path, image.data, static_cast<int>(image.width), static_cast<int>(image.height), static_cast<int>(image.channel));
        free(image.data);
        if (!wrote) {
            LOG_ERROR("flux2-dno failed to write %s", path.c_str());
            return false;
        }
        LOG_INFO("flux2-dno wrote %s", path.c_str());
        return true;
    });
    return ok ? 0 : 1;
}
