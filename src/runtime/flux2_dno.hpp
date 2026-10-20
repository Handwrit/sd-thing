#ifndef __SD_RUNTIME_FLUX2_DNO_HPP__
#define __SD_RUNTIME_FLUX2_DNO_HPP__

#include <functional>
#include <string>

#include "core/tensor.hpp"

struct StableDiffusionGGML;

namespace sd {
namespace flux2_dno {

struct Options {
    int steps            = 2;
    int iterations       = 1;
    float angle          = 0.03f;
    float gamma          = 0.01f;
    int width            = 512;
    int height           = 512;
    int permutations     = 100;
    int check_points     = 4;
    float check_eps      = 1e-2f;
    float check_tol      = 0.2f;
    bool check_grad      = false;
    bool through_vae     = false;
    bool check_penalty   = false;
    bool bench_out_prod  = false;
    uint64_t seed        = 0;
    float guidance       = 4.0f;
    std::string prompt;
    std::string out_dir  = "output/flux2-dno";
};

using ImageWriter = std::function<bool(int iteration, const Tensor<float>& rgb)>;

bool penalty_grad_matches_finite_difference();
bool primitive_grad_matches_finite_difference();
bool bench_out_prod();

// Loads nothing. The caller owns a Flux 2 dev context with flash attention off.
bool run(StableDiffusionGGML* sd, const Options& options, const ImageWriter& write_image);

}  // namespace flux2_dno
}  // namespace sd

#endif
