#include "routes.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

static bool output_name_is_safe(const std::string& name) {
    if (name.empty() || name[0] == '.' || name.find("..") != std::string::npos) {
        return false;
    }
    for (unsigned char c : name) {
        if (!(std::isalnum(c) || c == '.' || c == '_' || c == '-')) {
            return false;
        }
    }
    if (name.size() < 5) {
        return false;
    }
    std::string ext = name.substr(name.size() - 4);
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".png";
}

static fs::path resolve_output_png(const std::string& output_dir, const std::string& name) {
    if (output_dir.empty() || !output_name_is_safe(name)) {
        return {};
    }
    std::error_code ec;
    const fs::path dir = fs::weakly_canonical(output_dir, ec);
    if (ec || dir.empty() || !fs::is_directory(dir, ec)) {
        return {};
    }
    const fs::path file = fs::weakly_canonical(dir / name, ec);
    if (ec || file.empty() || !fs::is_regular_file(file, ec)) {
        return {};
    }
    const std::string dir_s  = dir.u8string();
    const std::string file_s = file.u8string();
    if (file_s.size() <= dir_s.size()) {
        return {};
    }
    if (file_s.compare(0, dir_s.size(), dir_s) != 0) {
        return {};
    }
    const char sep = file_s[dir_s.size()];
    if (sep != '/' && sep != '\\') {
        return {};
    }
    return file;
}

void register_index_endpoints(httplib::Server& svr, ServerRuntime& rt, const std::string& index_html) {
    ServerRuntime* runtime            = &rt;
    const std::string serve_html_path = rt.svr_params->serve_html_path;
    svr.Get("/", [serve_html_path, index_html](const httplib::Request&, httplib::Response& res) {
        if (!serve_html_path.empty()) {
            std::ifstream file(serve_html_path);
            if (file) {
                std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                res.set_content(content, "text/html");
            } else {
                res.status = 500;
                res.set_content("Error: Unable to read HTML file", "text/plain");
            }
        } else {
            res.set_content(index_html, "text/html");
        }
    });

    svr.Get("/preview.png", [runtime](const httplib::Request&, httplib::Response& res) {
        std::vector<uint8_t> png;
        int step  = 0;
        int steps = 0;
        if (runtime->preview_state != nullptr) {
            std::lock_guard<std::mutex> lock(runtime->preview_state->mutex);
            png   = runtime->preview_state->png;
            step  = runtime->preview_state->step;
            steps = runtime->preview_state->steps;
        }
        res.set_header("Cache-Control", "no-store");
        res.set_header("X-SD-Step", std::to_string(step));
        res.set_header("X-SD-Steps", std::to_string(steps));
        res.set_header("Access-Control-Expose-Headers", "X-SD-Step, X-SD-Steps");
        if (png.empty()) {
            res.status = 204;
            return;
        }
        res.set_content(reinterpret_cast<const char*>(png.data()), png.size(), "image/png");
    });

    svr.Get("/outputs", [runtime](const httplib::Request&, httplib::Response& res) {
        json files = json::array();
        const std::string& output_dir = runtime->svr_params->output_dir;
        std::error_code ec;
        if (!output_dir.empty() && fs::is_directory(output_dir, ec)) {
            struct OutputEntry {
                std::string name;
                fs::file_time_type mtime;
            };
            std::vector<OutputEntry> entries;
            fs::directory_iterator it(output_dir, ec);
            const fs::directory_iterator end;
            for (; !ec && it != end; it.increment(ec)) {
                if (!it->is_regular_file(ec) || ec) {
                    continue;
                }
                const std::string name = it->path().filename().u8string();
                if (!output_name_is_safe(name)) {
                    continue;
                }
                const auto mtime = it->last_write_time(ec);
                if (ec) {
                    continue;
                }
                entries.push_back({name, mtime});
            }
            std::sort(entries.begin(), entries.end(), [](const OutputEntry& a, const OutputEntry& b) {
                if (a.mtime != b.mtime) {
                    return a.mtime > b.mtime;
                }
                return a.name > b.name;
            });
            for (const auto& entry : entries) {
                files.push_back({
                    {"name", entry.name},
                    {"url", "/outputs/" + entry.name},
                });
            }
        }
        res.set_header("Cache-Control", "no-store");
        res.set_content(json({{"files", files}}).dump(), "application/json");
    });

    svr.Get(R"(/outputs/([A-Za-z0-9._-]+))", [runtime](const httplib::Request& req, httplib::Response& res) {
        const fs::path file = resolve_output_png(runtime->svr_params->output_dir, req.matches[1]);
        if (file.empty()) {
            res.status = 404;
            res.set_content(R"({"error":"not found"})", "application/json");
            return;
        }
        res.set_file_content(file.u8string(), "image/png");
    });
}
