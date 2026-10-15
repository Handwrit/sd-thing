#ifndef __SD_MODEL_DIFFUSION_SHAPE_MASK_H__
#define __SD_MODEL_DIFFUSION_SHAPE_MASK_H__

#include <vector>

// A painted word, still in prompt bytes. weights is row-major, one value per
// latent cell, 0 for no permission and 1 for a full share.
struct ShapeMaskInput {
    int start = 0;
    int end   = 0;
    int grid_w = 0;
    int grid_h = 0;
    std::vector<float> weights;
};

// The same paint after the prompt has been tokenized. Every index shares weights.
struct ShapeTokenMask {
    std::vector<int> token_indices;
    int grid_w = 0;
    int grid_h = 0;
    std::vector<float> weights;
};

#endif  // __SD_MODEL_DIFFUSION_SHAPE_MASK_H__
