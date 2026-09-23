#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <regex>
#include <sstream>
#include <system_error>

#include "common/common.h"
#include "common/log.h"
#include "common/media_io.h"

namespace fs = std::filesystem;

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool is_supported_model_ext(const fs::path& p) {
    auto ext = lower_ascii(p.extension().string());
    return ext == ".gguf" || ext == ".pt" || ext == ".pth" || ext == ".safetensors";
}

static const std::string k_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val  = 0;
    int valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(k_base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        ret.push_back(k_base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (ret.size() % 4) {
        ret.push_back('=');
    }
    return ret;
}

std::string normalize_output_format(std::string output_format) {
    std::transform(output_format.begin(), output_format.end(), output_format.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return output_format;
}

std::vector<std::string> supported_img_output_formats(bool allow_webp) {
    std::vector<std::string> formats = {"png", "jpeg"};
#ifdef SD_USE_WEBP
    if (allow_webp) {
        formats.push_back("webp");
    }
#else
    (void)allow_webp;
#endif
    return formats;
}

std::vector<std::string> supported_vid_output_formats() {
    std::vector<std::string> formats;
#ifdef SD_USE_WEBM
    formats.push_back("webm");
#endif
#ifdef SD_USE_WEBP
    formats.push_back("webp");
#endif
    formats.push_back("avi");
    return formats;
}

static std::string valid_vid_output_formats_message() {
    const std::vector<std::string> formats = supported_vid_output_formats();

    std::string message = "invalid output_format, must be one of [";
    for (size_t i = 0; i < formats.size(); ++i) {
        if (i > 0) {
            message += ", ";
        }
        message += formats[i];
    }
    message += "]";
    return message;
}

bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    const std::vector<std::string> valid_formats = supported_img_output_formats(allow_webp);
    const bool valid_format                      = std::find(valid_formats.begin(),
                                                             valid_formats.end(),
                                                             request.output_format) != valid_formats.end();
    if (!valid_format) {
        error_message = "invalid output_format, must be one of [";
        for (size_t i = 0; i < valid_formats.size(); ++i) {
            if (i > 0) {
                error_message += ", ";
            }
            error_message += valid_formats[i];
        }
        error_message += "]";
        return false;
    }

    return true;
}

bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    if (request.output_format == "avi") {
        return true;
    }

    if (request.output_format == "webm") {
#ifdef SD_USE_WEBM
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    if (request.output_format == "webp") {
#ifdef SD_USE_WEBP
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    error_message = valid_vid_output_formats_message();
    return false;
}

std::string video_mime_type(const std::string& output_format) {
    if (output_format == "webm") {
        return "video/webm";
    }
    if (output_format == "webp") {
        return "image/webp";
    }
    return "video/x-msvideo";
}

bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode) {
    if (mode == VID_GEN) {
        return sd_ctx_supports_video_generation(runtime.sd_ctx);
    }
    if (mode == IMG_GEN) {
        return sd_ctx_supports_image_generation(runtime.sd_ctx);
    }
    return true;
}

std::string unsupported_generation_mode_error(SDMode mode) {
    if (mode == VID_GEN) {
        return "loaded model does not support vid_gen";
    }
    if (mode == IMG_GEN) {
        return "loaded model does not support img_gen";
    }
    return "loaded model does not support requested mode";
}

ArgOptions SDSvrParams::get_options() {
    ArgOptions options;

    options.string_options = {
        {"-l", "--listen-ip", "server listen ip (default: 127.0.0.1)", 0, &listen_ip},
        {"", "--serve-html-path", "path to HTML file to serve at root (optional)", 0, &serve_html_path},
        {"", "--output-dir", "directory to save generated PNGs as YYYYMMDDHHMMSS_<model>.png (optional)", 0, &output_dir},
    };

    options.int_options = {
        {"", "--listen-port", "server listen port (default: 1234)", &listen_port},
    };

    options.bool_options = {
        {"", "--color", "colors the logging tags according to level", true, &color},
    };

    auto on_help_arg = [&](int, const char**, int, bool& valid) {
        normal_exit = true;
        valid       = true;
        return -1;
    };

    options.manual_options = {
        {"-h", "--help", "show this help message and exit", on_help_arg},
    };
    add_log_options(options, log_level);
    return options;
}

bool SDSvrParams::validate() {
    if (listen_ip.empty()) {
        LOG_ERROR("error: the following arguments are required: listen_ip");
        return false;
    }

    if (listen_port < 0 || listen_port > 65535) {
        LOG_ERROR("error: listen_port should be in the range [0, 65535]");
        return false;
    }

    if (!serve_html_path.empty() && !fs::exists(serve_html_path)) {
        LOG_ERROR("error: serve_html_path file does not exist: %s", serve_html_path.c_str());
        return false;
    }
    return true;
}

bool SDSvrParams::resolve_and_validate() {
    if (!validate()) {
        return false;
    }
    return true;
}

std::string SDSvrParams::to_string() const {
    std::ostringstream oss;
    oss << "SDSvrParams {\n"
        << "  log_level: " << log_level_name(log_level) << ",\n"
        << "  listen_ip: " << listen_ip << ",\n"
        << "  listen_port: \"" << listen_port << "\",\n"
        << "  serve_html_path: \"" << serve_html_path << "\",\n"
        << "  output_dir: \"" << output_dir << "\",\n"
        << "}";
    return oss.str();
}

void refresh_lora_cache(ServerRuntime& rt) {
    std::vector<LoraEntry> new_cache;

    fs::path lora_dir = rt.ctx_params->lora_model_dir;
    if (fs::exists(lora_dir) && fs::is_directory(lora_dir)) {
        for (auto& entry : fs::recursive_directory_iterator(lora_dir, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path& p = entry.path();
            if (!is_supported_model_ext(p)) {
                continue;
            }

            LoraEntry lora_entry;
            lora_entry.name     = p.stem().u8string();
            lora_entry.fullpath = p.u8string();
            std::string rel     = p.lexically_relative(lora_dir).u8string();
            std::replace(rel.begin(), rel.end(), '\\', '/');
            lora_entry.path = rel;

            new_cache.push_back(std::move(lora_entry));
        }
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const LoraEntry& a, const LoraEntry& b) {
        return a.path < b.path;
    });

    {
        std::lock_guard<std::mutex> lock(*rt.lora_mutex);
        *rt.lora_cache = std::move(new_cache);
    }
}

std::string get_lora_full_path(ServerRuntime& rt, const std::string& path) {
    std::lock_guard<std::mutex> lock(*rt.lora_mutex);
    auto it = std::find_if(rt.lora_cache->begin(), rt.lora_cache->end(),
                           [&](const LoraEntry& entry) { return entry.path == path; });
    return it != rt.lora_cache->end() ? it->fullpath : "";
}

void refresh_upscaler_cache(ServerRuntime& rt) {
    std::vector<UpscalerEntry> new_cache;

    fs::path upscaler_dir = rt.ctx_params->hires_upscalers_dir;
    if (fs::exists(upscaler_dir) && fs::is_directory(upscaler_dir)) {
        for (auto& entry : fs::directory_iterator(upscaler_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path& p = entry.path();
            if (!is_supported_model_ext(p)) {
                continue;
            }

            UpscalerEntry upscaler_entry;
            upscaler_entry.name       = p.stem().u8string();
            upscaler_entry.fullpath   = fs::absolute(p).lexically_normal().u8string();
            upscaler_entry.model_name = "ESRGAN_4x";
            upscaler_entry.path       = p.filename().u8string();

            new_cache.push_back(std::move(upscaler_entry));
        }
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const UpscalerEntry& a, const UpscalerEntry& b) {
        return a.name < b.name;
    });

    {
        std::lock_guard<std::mutex> lock(*rt.upscaler_mutex);
        *rt.upscaler_cache = std::move(new_cache);
    }
}

int64_t unix_timestamp_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static std::string sanitize_output_stem(const std::string& stem) {
    std::string out;
    out.reserve(stem.size());
    bool last_us = false;
    for (unsigned char c : stem) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == '-') {
            out.push_back(static_cast<char>(c));
            last_us = false;
        } else if (!out.empty() && !last_us) {
            out.push_back('_');
            last_us = true;
        }
    }
    while (!out.empty() && (out.front() == '.' || out.front() == '_')) {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    if (out.empty()) {
        return "model";
    }
    return out;
}

static std::string output_model_stem(const ServerRuntime& runtime) {
    if (runtime.ctx_params == nullptr) {
        return "model";
    }
    const auto& ctx = *runtime.ctx_params;
    fs::path model_path;
    if (!ctx.model_path.empty()) {
        model_path = ctx.model_path;
    } else if (!ctx.diffusion_model_path.empty()) {
        model_path = ctx.diffusion_model_path;
    }
    return sanitize_output_stem(model_path.stem().u8string());
}

void request_generation_interrupt(ServerRuntime& runtime) {
    runtime.interrupt_requested.store(true, std::memory_order_release);
    if (runtime.sd_ctx != nullptr) {
        sd_cancel_generation(runtime.sd_ctx, SD_CANCEL_ALL);
    }
}

bool consume_generation_interrupt(ServerRuntime& runtime) {
    return runtime.interrupt_requested.exchange(false, std::memory_order_acq_rel);
}

static std::string local_datetime_stamp() {
    std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now);
#else
    localtime_r(&now, &local_tm);
#endif
    char buf[16];
    if (std::strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &local_tm) == 0) {
        return "00000000000000";
    }
    return buf;
}

void server_preview_callback(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
    (void)is_noisy;
    auto* preview = static_cast<PreviewState*>(data);
    if (preview == nullptr || frames == nullptr || frame_count < 1 || frames[0].data == nullptr) {
        return;
    }
    std::vector<uint8_t> png = encode_image_to_vector(EncodedImageFormat::PNG,
                                                      frames[0].data,
                                                      frames[0].width,
                                                      frames[0].height,
                                                      frames[0].channel);
    if (png.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(preview->mutex);
    preview->png = std::move(png);
    // Two-pass samplers preview the first evaluation with a negative step.
    // Sampling progress owns the step counter, and it only reports a finished step.
    if (step > 0) {
        preview->step = step;
    }
}

void server_progress_callback(int step, int steps, float time, void* data) {
    (void)time;
    auto* preview = static_cast<PreviewState*>(data);
    if (preview == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(preview->mutex);
    preview->step  = step;
    preview->steps = steps;
}

static const char* scheduler_label(const SDGenerationParams& params) {
    const char* name = sd_scheduler_name(params.sample_params.scheduler);
    return name != nullptr ? name : "default";
}

static const char* sampler_label(const SDGenerationParams& params) {
    const char* name = sd_sample_method_name(params.sample_params.sample_method);
    return name != nullptr ? name : "default";
}

void begin_generation(ServerRuntime& runtime,
                      const std::string& job_id,
                      const char* kind,
                      const SDGenerationParams& params,
                      bool img2img) {
    runtime.interrupt_requested.store(false, std::memory_order_release);
    if (runtime.preview_state != nullptr) {
        std::lock_guard<std::mutex> lock(runtime.preview_state->mutex);
        runtime.preview_state->png.clear();
        runtime.preview_state->step  = 0;
        runtime.preview_state->steps = params.sample_params.sample_steps;
    }
    if (runtime.generation_status == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(runtime.generation_status->mutex);
    runtime.generation_status->busy            = true;
    runtime.generation_status->img2img         = img2img;
    runtime.generation_status->job_id          = job_id;
    runtime.generation_status->kind            = kind != nullptr ? kind : (img2img ? "img2img" : "txt2img");
    runtime.generation_status->prompt          = params.prompt;
    runtime.generation_status->negative_prompt = params.negative_prompt;
    runtime.generation_status->width           = params.get_resolved_width();
    runtime.generation_status->height          = params.get_resolved_height();
    runtime.generation_status->steps           = params.sample_params.sample_steps;
    runtime.generation_status->seed            = params.seed;
    runtime.generation_status->sampler         = sampler_label(params);
    runtime.generation_status->scheduler       = scheduler_label(params);
}

void finish_generation(ServerRuntime& runtime,
                       const std::string& job_id,
                       const std::string& status,
                       const std::string& prompt,
                       const std::vector<std::string>& output_files) {
    if (runtime.generation_status == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(runtime.generation_status->mutex);
    runtime.generation_status->busy                  = false;
    runtime.generation_status->job_id.clear();
    runtime.generation_status->last_finished_id      = job_id;
    runtime.generation_status->last_finished_status  = status;
    runtime.generation_status->last_finished_prompt  = prompt;
    runtime.generation_status->last_finished_files   = output_files;
}

std::string save_generated_image(const ServerRuntime& runtime,
                                 const sd_image_t& image,
                                 const std::string& metadata) {
    if (runtime.svr_params == nullptr || runtime.svr_params->output_dir.empty() || image.data == nullptr) {
        return {};
    }

    const fs::path dir = runtime.svr_params->output_dir;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        LOG_ERROR("failed to create output dir '%s': %s", dir.u8string().c_str(), ec.message().c_str());
        return {};
    }

    const std::string stamp = local_datetime_stamp();
    const std::string stem  = output_model_stem(runtime);
    const std::string base  = stamp + "_" + stem;
    fs::path path           = dir / (base + ".png");
    if (fs::exists(path, ec)) {
        for (int n = 2; n < 10000; ++n) {
            path = dir / (base + "_" + std::to_string(n) + ".png");
            if (!fs::exists(path, ec)) {
                break;
            }
        }
    }

    if (!write_image_to_file(path.u8string(),
                             image.data,
                             image.width,
                             image.height,
                             image.channel,
                             metadata)) {
        LOG_ERROR("save result image to '%s' failed", path.u8string().c_str());
        return {};
    }
    LOG_INFO("save result image to '%s'", path.u8string().c_str());
    return path.u8string();
}
