// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

const char* async_job_kind_name(AsyncJobKind kind) {
    switch (kind) {
        case AsyncJobKind::ImgGen:
            return "img_gen";
        case AsyncJobKind::VidGen:
            return "vid_gen";
        default:
            return "img_gen";
    }
}

const char* async_job_status_name(AsyncJobStatus status) {
    switch (status) {
        case AsyncJobStatus::Queued:
            return "queued";
        case AsyncJobStatus::Generating:
            return "generating";
        case AsyncJobStatus::Completed:
            return "completed";
        case AsyncJobStatus::Failed:
            return "failed";
        case AsyncJobStatus::Cancelled:
            return "cancelled";
        default:
            return "failed";
    }
}

void purge_expired_jobs(AsyncJobManager& manager) {
    const int64_t now = unix_timestamp_now();

    for (auto it = manager.expired_jobs.begin(); it != manager.expired_jobs.end();) {
        if (it->second <= now) {
            it = manager.expired_jobs.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = manager.jobs.begin(); it != manager.jobs.end();) {
        const auto& job = it->second;
        if (job->completed_at == 0) {
            ++it;
            continue;
        }

        int64_t ttl_seconds = job->status == AsyncJobStatus::Completed
                                  ? manager.completed_ttl_seconds
                                  : manager.failed_ttl_seconds;
        if (now - job->completed_at >= ttl_seconds) {
            manager.expired_jobs[job->id] = now + std::max<int64_t>(ttl_seconds, 60);
            it                            = manager.jobs.erase(it);
        } else {
            ++it;
        }
    }
}

size_t count_pending_jobs(const AsyncJobManager& manager) {
    size_t pending = 0;
    for (const auto& entry : manager.jobs) {
        if (entry.second->status == AsyncJobStatus::Queued ||
            entry.second->status == AsyncJobStatus::Generating) {
            ++pending;
        }
    }
    return pending;
}

std::string make_async_job_id(AsyncJobManager& manager) {
    std::ostringstream oss;
    oss << "job_" << std::hex << unix_timestamp_now() << "_" << std::setw(8)
        << std::setfill('0') << manager.next_id++;
    return oss.str();
}

bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job) {
    auto new_end = std::remove(manager.queue.begin(), manager.queue.end(), job.id);
    if (new_end == manager.queue.end()) {
        return false;
    }

    manager.queue.erase(new_end, manager.queue.end());
    job.status       = AsyncJobStatus::Cancelled;
    job.completed_at = unix_timestamp_now();
    job.result_images_b64.clear();
    job.result_output_files.clear();
    job.result_media_b64.clear();
    job.result_media_mime_type.clear();
    job.result_frame_count = 0;
    job.result_fps         = 0;
    job.error_code         = "cancelled";
    job.error_message      = "job cancelled by client";
    return true;
}

json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job) {
    json result;
    result["id"]             = job.id;
    result["kind"]           = async_job_kind_name(job.kind);
    result["status"]         = async_job_status_name(job.status);
    result["created"]        = job.created_at;
    result["started"]        = job.started_at == 0 ? json(nullptr) : json(job.started_at);
    result["completed"]      = job.completed_at == 0 ? json(nullptr) : json(job.completed_at);
    result["queue_position"] = 0;

    if (job.status == AsyncJobStatus::Queued) {
        size_t position = 1;
        for (const auto& queued_id : manager.queue) {
            if (queued_id == job.id) {
                result["queue_position"] = position;
                break;
            }
            ++position;
        }
    }

    if (job.status == AsyncJobStatus::Completed) {
        if (job.kind == AsyncJobKind::VidGen) {
            result["result"] = {
                {"output_format", job.vid_gen.output_format},
                {"mime_type", job.result_media_mime_type},
                {"fps", job.result_fps},
                {"frame_count", job.result_frame_count},
                {"b64_json", job.result_media_b64},
            };
        } else {
            json images = json::array();
            for (size_t i = 0; i < job.result_images_b64.size(); ++i) {
                images.push_back({{"index", i}, {"b64_json", job.result_images_b64[i]}});
            }
            result["result"] = {
                {"output_format", job.img_gen.output_format},
                {"images", images},
                {"output_files", job.result_output_files},
            };
        }
        result["error"] = nullptr;
    } else if (job.status == AsyncJobStatus::Failed ||
               job.status == AsyncJobStatus::Cancelled) {
        result["result"] = nullptr;
        result["error"]  = {
             {"code",
             job.error_code.empty()
                  ? (job.status == AsyncJobStatus::Cancelled ? "cancelled" : "generation_failed")
                  : job.error_code},
             {"message", job.error_message},
        };
    } else {
        result["result"] = nullptr;
        result["error"]  = nullptr;
    }

    return result;
}

json make_job_view(const AsyncGenerationJob& job, int queue_position) {
    const SDGenerationParams& params = job.kind == AsyncJobKind::VidGen
                                           ? job.vid_gen.gen_params
                                           : job.img_gen.gen_params;
    const char* sampler = sd_sample_method_name(params.sample_params.sample_method);
    const char* sched   = sd_scheduler_name(params.sample_params.scheduler);
    const char* kind    = job.kind == AsyncJobKind::VidGen
                              ? "vid_gen"
                              : (job.img2img ? "img2img" : "txt2img");
    json view;
    view["id"]              = job.id;
    view["status"]          = async_job_status_name(job.status);
    view["kind"]            = kind;
    view["prompt"]          = params.prompt;
    view["negative_prompt"] = params.negative_prompt;
    view["nag_prompt"]      = params.nag_prompt;
    view["width"]           = params.get_resolved_width();
    view["height"]          = params.get_resolved_height();
    view["steps"]           = params.sample_params.sample_steps;
    view["seed"]            = params.seed;
    view["sampler"]         = sampler != nullptr ? sampler : "default";
    view["scheduler"]       = sched != nullptr ? sched : "default";
    view["i2i"]             = job.img2img;
    view["queue_position"]  = queue_position;
    if (job.status == AsyncJobStatus::Completed) {
        view["output_files"] = job.result_output_files;
    }
    return view;
}

std::shared_ptr<AsyncGenerationJob> enqueue_img_gen_job(ServerRuntime& runtime,
                                                        ImgGenJobRequest request,
                                                        bool img2img,
                                                        std::string& error_message) {
    if (runtime.async_job_manager == nullptr) {
        error_message = "job queue is unavailable";
        return nullptr;
    }
    AsyncJobManager& manager                = *runtime.async_job_manager;
    std::shared_ptr<AsyncGenerationJob> job = std::make_shared<AsyncGenerationJob>();
    job->kind                               = AsyncJobKind::ImgGen;
    job->status                             = AsyncJobStatus::Queued;
    job->created_at                         = unix_timestamp_now();
    job->img_gen                            = std::move(request);
    job->img2img                            = img2img;

    {
        std::lock_guard<std::mutex> lock(manager.mutex);
        purge_expired_jobs(manager);
        if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
            error_message = "job queue is full";
            return nullptr;
        }
        job->id               = make_async_job_id(manager);
        manager.jobs[job->id] = job;
        manager.queue.push_back(job->id);
    }
    manager.cv.notify_one();
    return job;
}

bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         std::string& error_message) {
    sd_img_gen_params_t params = job.img_gen.to_sd_img_gen_params_t();

    SDImageVec results;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = nullptr;
        int num_results         = 0;
        if (!generate_image(runtime.sd_ctx, &params, &raw_results, &num_results)) {
            raw_results = nullptr;
            num_results = 0;
        }
        results.adopt(raw_results, num_results);
    }

    const int num_results = results.count();
    if (num_results <= 0) {
        if (consume_generation_interrupt(runtime)) {
            error_message = "cancelled";
        } else {
            error_message = "generate_image returned no results";
        }
        return false;
    }

    job.result_output_files.clear();

    EncodedImageFormat encoded_format = EncodedImageFormat::PNG;
    if (job.img_gen.output_format == "jpeg") {
        encoded_format = EncodedImageFormat::JPEG;
    } else if (job.img_gen.output_format == "webp") {
        encoded_format = EncodedImageFormat::WEBP;
    }

    int batch_count      = job.img_gen.gen_params.batch_count;
    int images_per_batch = batch_count > 0 ? std::max(1, num_results / batch_count) : 1;
    for (int i = 0; i < num_results; ++i) {
        if (results[i].data == nullptr) {
            continue;
        }

        const std::string metadata = job.img_gen.gen_params.embed_image_metadata
                                         ? get_image_params(*runtime.ctx_params,
                                                            job.img_gen.gen_params,
                                                            job.img_gen.gen_params.seed + i / images_per_batch)
                                         : "";
        auto image_bytes           = encode_image_to_vector(encoded_format,
                                                            results[i].data,
                                                            results[i].width,
                                                            results[i].height,
                                                            results[i].channel,
                                                            metadata,
                                                            job.img_gen.output_compression);
        if (image_bytes.empty()) {
            continue;
        }
        output_images.push_back(base64_encode(image_bytes));
        std::string saved_path = save_generated_image(runtime, results[i], metadata);
        if (!saved_path.empty()) {
            embed_output_source_images(saved_path, job.img_gen.gen_params);
            job.result_output_files.push_back(saved_path);
        }
    }

    if (output_images.empty()) {
        error_message = "generate_image returned empty encoded outputs";
        return false;
    }

    return true;
}

bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         std::string& error_message) {
    sd_vid_gen_params_t params = job.vid_gen.to_sd_vid_gen_params_t();
    std::string str_params     = job.vid_gen.gen_params.embed_image_metadata
                                     ? get_image_params(*runtime.ctx_params, job.vid_gen.gen_params, job.vid_gen.gen_params.seed, VID_GEN)
                                     : "";

    SDImageVec results;
    int num_results             = 0;
    sd_audio_t* generated_audio = nullptr;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = nullptr;
        if (!generate_video(runtime.sd_ctx, &params, &raw_results, &num_results, &generated_audio, &output_fps)) {
            raw_results = nullptr;
        }
        results.adopt(raw_results, num_results);
    }

    num_results = results.count();
    if (num_results <= 0) {
        free_sd_audio(generated_audio);
        if (consume_generation_interrupt(runtime)) {
            error_message = "cancelled";
        } else {
            error_message = "generate_video returned no results";
        }
        return false;
    }

    const char* version_name = sd_get_model_version_name(runtime.sd_ctx);
    if (version_name != nullptr && std::strcmp(version_name, "MiniMax-H3") == 0) {
        free_sd_audio(generated_audio);
        job.result_output_files = save_generated_frames(runtime, results.data(), num_results, str_params);
        if (!job.result_output_files.empty()) {
            embed_output_source_images(job.result_output_files.front(), job.vid_gen.gen_params);
        }
        if (job.result_output_files.empty()) {
            error_message = "failed to save generated frames";
            return false;
        }
        output_frame_count = num_results;
        output_media_b64.clear();
        output_media_mime_type.clear();
        return true;
    }

    std::vector<uint8_t> video_bytes = create_video_from_sd_images_to_vector(job.vid_gen.output_format,
                                                                             results.data(),
                                                                             num_results,
                                                                             output_fps,
                                                                             job.vid_gen.output_compression,
                                                                             generated_audio,
                                                                             str_params);
    free_sd_audio(generated_audio);
    if (video_bytes.empty()) {
        error_message = "failed to encode generated video container";
        return false;
    }

    job.result_output_files.clear();
    const std::string saved_path = save_generated_bytes(runtime, video_bytes, job.vid_gen.output_format);
    if (!saved_path.empty()) {
        job.result_output_files.push_back(saved_path);
    }

    output_media_b64       = base64_encode(video_bytes);
    output_media_mime_type = video_mime_type(job.vid_gen.output_format);
    output_frame_count     = num_results;
    return true;
}

void async_job_worker(ServerRuntime& runtime) {
    AsyncJobManager& manager = *runtime.async_job_manager;

    while (true) {
        std::shared_ptr<AsyncGenerationJob> job;
        {
            std::unique_lock<std::mutex> lock(manager.mutex);
            manager.cv.wait(lock, [&]() { return manager.stop || !manager.queue.empty(); });

            if (manager.stop && manager.queue.empty()) {
                break;
            }

            purge_expired_jobs(manager);
            if (manager.queue.empty()) {
                continue;
            }

            const std::string job_id = manager.queue.front();
            manager.queue.pop_front();

            auto it = manager.jobs.find(job_id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job             = it->second;
            job->status     = AsyncJobStatus::Generating;
            job->started_at = unix_timestamp_now();
        }

        const SDGenerationParams& gen_params = job->kind == AsyncJobKind::VidGen
                                                   ? job->vid_gen.gen_params
                                                   : job->img_gen.gen_params;
        const char* kind = job->kind == AsyncJobKind::VidGen
                               ? "vid_gen"
                               : (job->img2img ? "img2img" : "txt2img");
        begin_generation(runtime, job->id, kind, gen_params, job->img2img);

        std::vector<std::string> output_images;
        std::string output_media_b64;
        std::string output_media_mime_type;
        int output_frame_count = 0;
        int output_fps         = 0;
        std::string error_message;
        bool ok = false;

        if (job->kind == AsyncJobKind::ImgGen) {
            ok = execute_img_gen_job(runtime, *job, output_images, error_message);
        } else if (job->kind == AsyncJobKind::VidGen) {
            ok = execute_vid_gen_job(runtime,
                                     *job,
                                     output_media_b64,
                                     output_media_mime_type,
                                     output_frame_count,
                                     output_fps,
                                     error_message);
        } else {
            error_message = "unsupported job kind";
        }

        {
            std::lock_guard<std::mutex> lock(manager.mutex);
            auto it = manager.jobs.find(job->id);
            if (it == manager.jobs.end()) {
                finish_generation(runtime, job->id, "failed", "", {});
                continue;
            }

            job->completed_at = unix_timestamp_now();
            if (ok) {
                consume_generation_interrupt(runtime);
                job->status                 = AsyncJobStatus::Completed;
                job->result_images_b64      = std::move(output_images);
                job->result_media_b64       = std::move(output_media_b64);
                job->result_media_mime_type = std::move(output_media_mime_type);
                job->result_frame_count     = output_frame_count;
                job->result_fps             = output_fps;
                job->error_code.clear();
                job->error_message.clear();
            } else if (error_message == "cancelled") {
                job->status        = AsyncJobStatus::Cancelled;
                job->error_code    = "cancelled";
                job->error_message = "job cancelled by client";
                job->result_images_b64.clear();
                job->result_output_files.clear();
                job->result_media_b64.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps         = 0;
            } else {
                consume_generation_interrupt(runtime);
                job->status        = AsyncJobStatus::Failed;
                job->error_code    = "generation_failed";
                job->error_message = error_message.empty() ? "unknown generation error" : error_message;
                job->result_images_b64.clear();
                job->result_output_files.clear();
                job->result_media_b64.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps         = 0;
            }

            purge_expired_jobs(manager);
        }

        const std::string finished_status = ok
                                                ? "completed"
                                                : (error_message == "cancelled" ? "cancelled" : "failed");
        finish_generation(runtime,
                          job->id,
                          finished_status,
                          job->kind == AsyncJobKind::VidGen ? job->vid_gen.gen_params.prompt
                                                            : job->img_gen.gen_params.prompt,
                          job->result_output_files);
    }
}
