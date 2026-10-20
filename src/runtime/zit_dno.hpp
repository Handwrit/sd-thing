#ifndef __SD_RUNTIME_ZIT_DNO_HPP__
#define __SD_RUNTIME_ZIT_DNO_HPP__

#include <cstdint>
#include <functional>
#include <string>

#include "core/tensor.hpp"

struct StableDiffusionGGML;

namespace sd {
namespace zit_dno {

struct Options {
    int steps            = 8;
    int backward_steps   = 0;
    int reforward_steps  = 0;
    int width            = 512;
    int height           = 512;
    uint64_t seed = 0;
    bool check_grad = false;
    bool experimental_full_multi = false;
    std::string prompt;
};

using ImageWriter = std::function<bool(int iteration, const Tensor<float>& rgb)>;

// One DNO iteration, or the latent gradient check when options.check_grad is set.
// The caller owns a context with flash attention off.
bool run(StableDiffusionGGML* sd, const Options& options, const ImageWriter& write_image);

}  // namespace zit_dno
}  // namespace sd

#endif
