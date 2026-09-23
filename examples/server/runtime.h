#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <json.hpp>
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "stable-diffusion.h"

using json = nlohmann::json;

struct ArgOptions;
struct SDContextParams;
struct AsyncJobManager;

struct PreviewState {
    std::mutex mutex;
    std::vector<uint8_t> png;
    int step  = 0;
    int steps = 0;
};

struct GenerationStatus {
    std::mutex mutex;
    bool busy = false;
    bool img2img = false;
    std::string job_id;
    std::string kind;
    std::string prompt;
    std::string negative_prompt;
    int width  = 0;
    int height = 0;
    int steps  = 0;
    int64_t seed = -1;
    std::string sampler;
    std::string scheduler;
    std::string last_finished_id;
    std::string last_finished_status;
    std::string last_finished_prompt;
    std::vector<std::string> last_finished_files;
};

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    std::string serve_html_path;
    std::string output_dir;
    bool normal_exit = false;
    sd_log_level_t log_level = SD_LOG_INFO;
    bool color       = false;

    ArgOptions get_options();
    bool validate();
    bool resolve_and_validate();
    std::string to_string() const;
};

struct LoraEntry {
    std::string name;
    std::string path;
    std::string fullpath;
};

struct UpscalerEntry {
    std::string name;
    std::string path;
    std::string fullpath;
    std::string model_name;
    int scale = 4;
};

struct ServerRuntime {
    sd_ctx_t* sd_ctx;
    std::mutex* sd_ctx_mutex;
    const SDSvrParams* svr_params;
    const SDContextParams* ctx_params;
    const SDGenerationParams* default_gen_params;
    std::vector<LoraEntry>* lora_cache;
    std::mutex* lora_mutex;
    std::vector<UpscalerEntry>* upscaler_cache;
    std::mutex* upscaler_mutex;
    AsyncJobManager* async_job_manager;
    PreviewState* preview_state = nullptr;
    GenerationStatus* generation_status = nullptr;
    std::atomic<bool> interrupt_requested{false};
};

struct ImgGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "png";
    int output_compression    = 100;

    sd_img_gen_params_t to_sd_img_gen_params_t() {
        return gen_params.to_sd_img_gen_params_t();
    }
};

struct VidGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "webm";
    int output_compression    = 100;

    sd_vid_gen_params_t to_sd_vid_gen_params_t() {
        return gen_params.to_sd_vid_gen_params_t();
    }
};

std::string base64_encode(const std::vector<uint8_t>& bytes);
std::string normalize_output_format(std::string output_format);
std::vector<std::string> supported_img_output_formats(bool allow_webp = true);
std::vector<std::string> supported_vid_output_formats();
bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message);
bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message);
std::string video_mime_type(const std::string& output_format);
bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode);
std::string unsupported_generation_mode_error(SDMode mode);
void refresh_lora_cache(ServerRuntime& rt);
std::string get_lora_full_path(ServerRuntime& rt, const std::string& path);
void refresh_upscaler_cache(ServerRuntime& rt);
int64_t unix_timestamp_now();
void server_preview_callback(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);
void server_progress_callback(int step, int steps, float time, void* data);
void begin_generation(ServerRuntime& runtime,
                      const std::string& job_id,
                      const char* kind,
                      const SDGenerationParams& params,
                      bool img2img);
void finish_generation(ServerRuntime& runtime,
                       const std::string& job_id,
                       const std::string& status,
                       const std::string& prompt,
                       const std::vector<std::string>& output_files);
void request_generation_interrupt(ServerRuntime& runtime);
bool consume_generation_interrupt(ServerRuntime& runtime);
std::string save_generated_image(const ServerRuntime& runtime,
                                 const sd_image_t& image,
                                 const std::string& metadata);
std::vector<std::string> save_generated_frames(const ServerRuntime& runtime,
                                               const sd_image_t* images,
                                               int count,
                                               const std::string& metadata);
std::string save_generated_bytes(const ServerRuntime& runtime,
                                 const std::vector<uint8_t>& bytes,
                                 const std::string& extension);
