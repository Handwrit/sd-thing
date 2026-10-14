#ifndef __MEDIA_IO_H__
#define __MEDIA_IO_H__

#include <cstdint>
#include <string>
#include <vector>

#include "stable-diffusion.h"

enum class EncodedImageFormat {
    JPEG,
    PNG,
    WEBP,
    UNKNOWN,
};

EncodedImageFormat encoded_image_format_from_path(const std::string& path);

std::vector<uint8_t> encode_image_to_vector(EncodedImageFormat format,
                                            const uint8_t* image,
                                            int width,
                                            int height,
                                            int channels,
                                            const std::string& parameters = "",
                                            int quality                   = 90);

bool write_image_to_file(const std::string& path,
                         const uint8_t* image,
                         int width,
                         int height,
                         int channels,
                         const std::string& parameters = "",
                         int quality                   = 90);

enum class PngSourceRole : uint8_t {
    Init = 1,
    Ref  = 2,
};

struct PngSourceImage {
    PngSourceRole role = PngSourceRole::Init;
    uint8_t index      = 0;
    const uint8_t* data = nullptr;
    int width           = 0;
    int height          = 0;
    int channels        = 0;
};

// Stores source pixels in an ancillary PNG chunk. Viewers ignore it.
// A failure leaves the original file unchanged.
bool embed_png_source_images(const std::string& path, const std::vector<PngSourceImage>& images);

uint8_t* load_image_from_file(const char* image_path,
                              int& width,
                              int& height,
                              int expected_width   = 0,
                              int expected_height  = 0,
                              int expected_channel = 3);

bool load_sd_image_from_file(sd_image_t* image,
                             const char* image_path,
                             int expected_width   = 0,
                             int expected_height  = 0,
                             int expected_channel = 3);

uint8_t* load_image_from_memory(const char* image_bytes,
                                int len,
                                int& width,
                                int& height,
                                int expected_width   = 0,
                                int expected_height  = 0,
                                int expected_channel = 3);

int create_mjpg_avi_from_sd_images(const char* filename,
                                   sd_image_t* images,
                                   int num_images,
                                   int fps,
                                   int quality                   = 90,
                                   const sd_audio_t* audio       = nullptr,
                                   const std::string& parameters = "");
std::vector<uint8_t> create_mjpg_avi_from_sd_images_to_vector(sd_image_t* images,
                                                              int num_images,
                                                              int fps,
                                                              int quality                   = 90,
                                                              const sd_audio_t* audio       = nullptr,
                                                              const std::string& parameters = "");

#ifdef SD_USE_WEBP
int create_animated_webp_from_sd_images(const char* filename,
                                        sd_image_t* images,
                                        int num_images,
                                        int fps,
                                        int quality = 90);
std::vector<uint8_t> create_animated_webp_from_sd_images_to_vector(sd_image_t* images,
                                                                   int num_images,
                                                                   int fps,
                                                                   int quality = 90);
#endif

#ifdef SD_USE_WEBM
int create_webm_from_sd_images(const char* filename,
                               sd_image_t* images,
                               int num_images,
                               int fps,
                               int quality                   = 90,
                               const sd_audio_t* audio       = nullptr,
                               const std::string& parameters = "");
std::vector<uint8_t> create_webm_from_sd_images_to_vector(sd_image_t* images,
                                                          int num_images,
                                                          int fps,
                                                          int quality                   = 90,
                                                          const sd_audio_t* audio       = nullptr,
                                                          const std::string& parameters = "");
#endif

int create_video_from_sd_images(const char* filename,
                                sd_image_t* images,
                                int num_images,
                                int fps,
                                int quality                   = 90,
                                const sd_audio_t* audio       = nullptr,
                                const std::string& parameters = "");
std::vector<uint8_t> create_video_from_sd_images_to_vector(const std::string& output_format,
                                                           sd_image_t* images,
                                                           int num_images,
                                                           int fps,
                                                           int quality                   = 90,
                                                           const sd_audio_t* audio       = nullptr,
                                                           const std::string& parameters = "");

bool write_wav_to_file(const std::string& path,
                       const float* interleaved_samples,
                       uint64_t sample_count,
                       uint32_t channels,
                       uint32_t sample_rate);

bool load_wav_from_file(const std::string& path,
                        std::vector<float>& interleaved_samples,
                        uint32_t& sample_rate,
                        uint32_t& channels);

#endif  // __MEDIA_IO_H__
