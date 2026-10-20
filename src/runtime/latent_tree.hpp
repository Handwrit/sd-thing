#ifndef __SD_RUNTIME_LATENT_TREE_HPP__
#define __SD_RUNTIME_LATENT_TREE_HPP__

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "stable-diffusion.h"

class StableDiffusionGGML;

namespace sd::tree {

    struct LatentStepRecord {
        sd::Tensor<float> x_before;
        sd::Tensor<float> x_after;
        sd::Tensor<float> clean;
        float sigma_in = 0.f;
        float sigma    = 0.f;
        int step       = 0;
        std::shared_ptr<RNG> rng_before;
        std::shared_ptr<RNG> rng_after;
    };

    struct LatentTreeHooks {
        float dcw_amount = 0.f;
        float sigma_ref  = 0.f;
        std::shared_ptr<RNG> rng;
        std::shared_ptr<RNG> rng_before;
        std::function<void(LatentStepRecord)> on_step;
    };

    // One-level unnormalized Haar. A gain of 1 inverts back to the input,
    // so the sampler skips the call when the amount is 0.
    sd::Tensor<float> apply_haar_dcw(const sd::Tensor<float>& latent,
                                     float sigma,
                                     float sigma_ref,
                                     float amount);

    struct CardCopy {
        bool found       = false;
        bool has_cards   = false;
        bool running     = false;
        int id           = 0;
        int parent       = 0;
        int width        = 0;
        int height       = 0;
        std::string kind;
        float sigma      = 0.f;
        float sigma_in   = 0.f;
        float eta        = 0.f;
        float sigma_ref  = 0.f;
        bool ancestral   = false;
        int64_t seed     = 0;
        sd::Tensor<float> x_before;
        sd::Tensor<float> x_after;
        sd::Tensor<float> clean;
        std::shared_ptr<RNG> rng_before;
        std::shared_ptr<RNG> rng_after;
        std::vector<float> tail;
    };

    struct RunStamp {
        int anchor = 0;
        std::string settings;
        float sigma_ref = 0.f;
        float eta       = 0.f;
        bool ancestral  = false;
        int64_t seed    = 0;
        const std::vector<float>* sigmas = nullptr;
    };

    bool session_query(StableDiffusionGGML* sd, int card_id, CardCopy* out);
    bool session_begin(StableDiffusionGGML* sd,
                       const RunStamp& stamp,
                       int width,
                       int height,
                       bool planting,
                       std::string* error);
    void session_record(StableDiffusionGGML* sd, LatentStepRecord rec);
    void session_end(StableDiffusionGGML* sd);
    std::string session_json(StableDiffusionGGML* sd);
    bool session_card_rgb(StableDiffusionGGML* sd,
                          int card_id,
                          int which,
                          std::vector<uint8_t>* rgb,
                          int* width,
                          int* height,
                          int* channels);
    bool session_close(StableDiffusionGGML* sd, std::string* error);
    void session_release(StableDiffusionGGML* sd);
    bool session_bake(StableDiffusionGGML* sd, int card_id, sd_image_t* out, std::string* error);
    bool session_paint_export(StableDiffusionGGML* sd, int card_id, std::vector<uint8_t>* tiff, std::string* error);
    bool session_paint_import(StableDiffusionGGML* sd,
                              int card_id,
                              const uint8_t* data,
                              size_t size,
                              std::string* error);
    // Forks a child. op is "dcw" or "threshold". The parent card is left as it was.
    bool session_operate(StableDiffusionGGML* sd,
                         int card_id,
                         const char* op,
                         float amount,
                         int* child_id,
                         std::string* error);

}  // namespace sd::tree

#endif  // __SD_RUNTIME_LATENT_TREE_HPP__
