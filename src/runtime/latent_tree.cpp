#include "runtime/latent_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/util.h"
#include "pipeline/diffusion_engine.h"

namespace sd::tree {

    namespace {

        struct Card {
            int id     = 0;
            int parent = 0;
            std::string kind;
            std::string settings;
            float sigma     = 0.f;
            float sigma_in  = 0.f;
            float eta       = 0.f;
            float sigma_ref = 0.f;
            bool ancestral  = false;
            int64_t seed    = 0;
            sd::Tensor<float> x_before;
            sd::Tensor<float> x_after;
            sd::Tensor<float> clean;
            std::shared_ptr<RNG> rng_before;
            std::shared_ptr<RNG> rng_after;
            std::vector<float> tail;
            std::vector<uint8_t> noisy_rgb;
            std::vector<uint8_t> clean_rgb;
            int img_w = 0;
            int img_h = 0;
            int img_c = 0;
        };

        struct Session {
            std::mutex mu;
            bool running = false;
            bool closed  = false;
            int width    = 0;
            int height   = 0;
            int next_id  = 1;
            int last_id  = 0;
            float sigma_ref = 0.f;
            std::string error;
            RunStamp stamp;
            std::unordered_map<int, Card> cards;
        };

        std::mutex g_map_mu;
        std::unordered_map<StableDiffusionGGML*, std::shared_ptr<Session>> g_sessions;

        std::shared_ptr<Session> session_get(StableDiffusionGGML* sd, bool create) {
            std::lock_guard<std::mutex> lock(g_map_mu);
            auto it = g_sessions.find(sd);
            if (it != g_sessions.end()) {
                return it->second;
            }
            if (!create) {
                return nullptr;
            }
            auto session = std::make_shared<Session>();
            g_sessions.emplace(sd, session);
            return session;
        }

        void copy_card(const Card& card, CardCopy* out) {
            out->found      = true;
            out->id         = card.id;
            out->parent     = card.parent;
            out->kind       = card.kind;
            out->sigma      = card.sigma;
            out->sigma_in   = card.sigma_in;
            out->eta        = card.eta;
            out->sigma_ref  = card.sigma_ref;
            out->ancestral  = card.ancestral;
            out->seed       = card.seed;
            out->x_before   = card.x_before;
            out->x_after    = card.x_after;
            out->clean      = card.clean;
            out->rng_before = card.rng_before;
            out->rng_after  = card.rng_after;
            out->tail       = card.tail;
        }

        void project_rgb(StableDiffusionGGML* sd,
                         const sd::Tensor<float>& latent,
                         std::vector<uint8_t>* rgb,
                         int* width,
                         int* height,
                         int* channels) {
            rgb->clear();
            *width = *height = *channels = 0;
            if (latent.empty()) {
                return;
            }
            sd->preview_image(0,
                              latent,
                              sd->version,
                              PREVIEW_PROJ,
                              [&](int, int frames, sd_image_t* images, bool, void*) {
                                  if (frames < 1 || images == nullptr || images[0].data == nullptr || images[0].channel == 0) {
                                      return;
                                  }
                                  *width            = static_cast<int>(images[0].width);
                                  *height           = static_cast<int>(images[0].height);
                                  *channels         = static_cast<int>(images[0].channel);
                                  const size_t bytes = static_cast<size_t>(*width) * static_cast<size_t>(*height) * static_cast<size_t>(*channels);
                                  rgb->assign(images[0].data, images[0].data + bytes);
                              },
                              nullptr,
                              true);
        }

        std::string json_escape(const std::string& text) {
            std::string out;
            out.reserve(text.size() + 8);
            for (unsigned char ch : text) {
                switch (ch) {
                    case '\\':
                        out += "\\\\";
                        break;
                    case '"':
                        out += "\\\"";
                        break;
                    case '\n':
                        out += "\\n";
                        break;
                    case '\r':
                        out += "\\r";
                        break;
                    default:
                        if (ch < 0x20) {
                            char buf[8];
                            std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                            out += buf;
                        } else {
                            out.push_back(static_cast<char>(ch));
                        }
                        break;
                }
            }
            return out;
        }

        std::string settings_object(const std::string& settings) {
            if (settings.empty() || settings.front() != '{') {
                return "{}";
            }
            return settings;
        }

        void push_u16(std::vector<uint8_t>& out, uint16_t value) {
            out.push_back(static_cast<uint8_t>(value & 0xff));
            out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
        }

        void push_u32(std::vector<uint8_t>& out, uint32_t value) {
            out.push_back(static_cast<uint8_t>(value & 0xff));
            out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
            out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
            out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
        }

        void ifd_entry(std::vector<uint8_t>& out, uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
            push_u16(out, tag);
            push_u16(out, type);
            push_u32(out, count);
            push_u32(out, value);
        }

        bool read_u16(const uint8_t* data, size_t size, size_t at, uint16_t* value) {
            if (at + 2 > size) {
                return false;
            }
            *value = static_cast<uint16_t>(data[at] | (data[at + 1] << 8));
            return true;
        }

        bool read_u32(const uint8_t* data, size_t size, size_t at, uint32_t* value) {
            if (at + 4 > size) {
                return false;
            }
            *value = static_cast<uint32_t>(data[at] | (data[at + 1] << 8) | (data[at + 2] << 16) | (data[at + 3] << 24));
            return true;
        }

        // count == 1 stores a SHORT or LONG in the value field. A longer list is an offset to that array.
        bool read_tiff_values(const uint8_t* data, size_t size, uint16_t type, uint32_t count, uint32_t value, std::vector<uint32_t>* out) {
            out->clear();
            if (count == 0 || count > 1000000) {
                return false;
            }
            if (count == 1) {
                if (type == 3) {
                    out->push_back(value & 0xffffu);
                    return true;
                }
                if (type == 4) {
                    out->push_back(value);
                    return true;
                }
                return false;
            }
            size_t at = value;
            out->reserve(count);
            for (uint32_t i = 0; i < count; ++i) {
                if (type == 3) {
                    uint16_t item = 0;
                    if (!read_u16(data, size, at, &item)) {
                        return false;
                    }
                    out->push_back(item);
                    at += 2;
                } else if (type == 4) {
                    uint32_t item = 0;
                    if (!read_u32(data, size, at, &item)) {
                        return false;
                    }
                    out->push_back(item);
                    at += 4;
                } else {
                    return false;
                }
            }
            return true;
        }

        float sample_rgb(const sd::Tensor<float>& tensor, int x, int y, int channel) {
            float value = tensor.dim() >= 4 ? tensor.index(x, y, channel, 0) : tensor.index(x, y, channel);
            if (value < 0.f) {
                return 0.f;
            }
            if (value > 1.f) {
                return 1.f;
            }
            return value;
        }

        bool tensor_to_tiff(const sd::Tensor<float>& tensor, std::vector<uint8_t>* tiff, std::string* error) {
            if (tensor.dim() < 3) {
                if (error) {
                    *error = "Decoded picture has no color planes";
                }
                return false;
            }
            const int width    = static_cast<int>(tensor.shape()[0]);
            const int height   = static_cast<int>(tensor.shape()[1]);
            const int channels = static_cast<int>(tensor.shape()[2]);
            if (width < 1 || height < 1 || channels < 3) {
                if (error) {
                    *error = "Decoded picture is not RGB";
                }
                return false;
            }
            const uint32_t pixels = static_cast<uint32_t>(width) * static_cast<uint32_t>(height);
            const uint32_t nbytes = pixels * 6;
            constexpr uint32_t kBitsAt = 8 + 2 + 10 * 12 + 4;
            constexpr uint32_t kPixels = kBitsAt + 8;
            tiff->clear();
            tiff->reserve(static_cast<size_t>(kPixels) + nbytes);
            push_u16(*tiff, 0x4949);
            push_u16(*tiff, 42);
            push_u32(*tiff, 8);
            push_u16(*tiff, 10);
            ifd_entry(*tiff, 256, 4, 1, static_cast<uint32_t>(width));
            ifd_entry(*tiff, 257, 4, 1, static_cast<uint32_t>(height));
            ifd_entry(*tiff, 258, 3, 3, kBitsAt);
            ifd_entry(*tiff, 259, 3, 1, 1);
            ifd_entry(*tiff, 262, 3, 1, 2);
            ifd_entry(*tiff, 273, 4, 1, kPixels);
            ifd_entry(*tiff, 277, 3, 1, 3);
            ifd_entry(*tiff, 278, 4, 1, static_cast<uint32_t>(height));
            ifd_entry(*tiff, 279, 4, 1, nbytes);
            ifd_entry(*tiff, 284, 3, 1, 1);
            push_u32(*tiff, 0);
            if (tiff->size() != kBitsAt) {
                if (error) {
                    *error = "TIFF layout failed";
                }
                return false;
            }
            push_u16(*tiff, 16);
            push_u16(*tiff, 16);
            push_u16(*tiff, 16);
            push_u16(*tiff, 0);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const float sample = sample_rgb(tensor, x, y, c);
                        push_u16(*tiff, static_cast<uint16_t>(sample * 65535.f + 0.5f));
                    }
                }
            }
            return true;
        }

        bool tiff_to_tensor(const uint8_t* data, size_t size, int expect_w, int expect_h, sd::Tensor<float>* image, std::string* error) {
            auto fail = [&](const char* message) {
                if (error) {
                    *error = message;
                }
                return false;
            };
            uint16_t order = 0;
            uint16_t magic = 0;
            uint32_t ifd   = 0;
            if (!read_u16(data, size, 0, &order) || order != 0x4949 || !read_u16(data, size, 2, &magic) || magic != 42 ||
                !read_u32(data, size, 4, &ifd)) {
                return fail("TIFF must be little-endian uncompressed 16-bit RGB");
            }
            uint16_t count = 0;
            if (!read_u16(data, size, ifd, &count)) {
                return fail("TIFF header is truncated");
            }
            uint32_t compression = 0, photometric = 0, samples = 0, planar = 1;
            uint16_t width_type = 0, height_type = 0, rows_type = 0, strip_type = 0, bytes_type = 0;
            uint32_t width_n = 0, height_n = 0, rows_n = 0, strip_n = 0, bytes_n = 0;
            uint32_t width_val = 0, height_val = 0, rows_val = 0, strip_val = 0, bytes_val = 0;
            uint32_t bits_offset = 0;
            uint32_t bits_count  = 0;
            for (uint16_t i = 0; i < count; ++i) {
                const size_t at = static_cast<size_t>(ifd) + 2 + static_cast<size_t>(i) * 12;
                uint16_t tag = 0, type = 0;
                uint32_t n = 0, value = 0;
                if (!read_u16(data, size, at, &tag) || !read_u16(data, size, at + 2, &type) || !read_u32(data, size, at + 4, &n) ||
                    !read_u32(data, size, at + 8, &value)) {
                    return fail("TIFF header is truncated");
                }
                if (tag == 256) {
                    width_type = type;
                    width_n    = n;
                    width_val  = value;
                } else if (tag == 257) {
                    height_type = type;
                    height_n    = n;
                    height_val  = value;
                } else if (tag == 258) {
                    bits_count  = n;
                    bits_offset = value;
                } else if (tag == 259) {
                    compression = value;
                } else if (tag == 262) {
                    photometric = value;
                } else if (tag == 273) {
                    strip_type = type;
                    strip_n    = n;
                    strip_val  = value;
                } else if (tag == 277) {
                    samples = value;
                } else if (tag == 278) {
                    rows_type = type;
                    rows_n    = n;
                    rows_val  = value;
                } else if (tag == 279) {
                    bytes_type = type;
                    bytes_n    = n;
                    bytes_val  = value;
                } else if (tag == 284) {
                    planar = value;
                }
            }
            std::vector<uint32_t> width_field, height_field, rows_field, strips, counts;
            if (compression != 1 || photometric != 2 || samples != 3 || planar != 1 || bits_count != 3 ||
                !read_tiff_values(data, size, width_type, width_n, width_val, &width_field) || width_field.size() != 1 ||
                !read_tiff_values(data, size, height_type, height_n, height_val, &height_field) || height_field.size() != 1) {
                return fail("TIFF must be little-endian uncompressed 16-bit RGB");
            }
            const uint32_t width  = width_field[0];
            const uint32_t height = height_field[0];
            uint16_t b0 = 0, b1 = 0, b2 = 0;
            if (!read_u16(data, size, bits_offset, &b0) || !read_u16(data, size, bits_offset + 2, &b1) || !read_u16(data, size, bits_offset + 4, &b2) ||
                b0 != 16 || b1 != 16 || b2 != 16) {
                return fail("TIFF must be little-endian uncompressed 16-bit RGB");
            }
            if (width == 0 || height == 0 || static_cast<uint64_t>(width) > 100000 || static_cast<uint64_t>(height) > 100000) {
                return fail("TIFF picture size is unusable");
            }
            if (static_cast<int>(width) != expect_w || static_cast<int>(height) != expect_h) {
                return fail("Paint file size does not match this session");
            }
            uint32_t rows = height;
            if (rows_n != 0) {
                if (!read_tiff_values(data, size, rows_type, rows_n, rows_val, &rows_field) || rows_field.size() != 1 || rows_field[0] == 0) {
                    return fail("TIFF picture size is unusable");
                }
                rows = rows_field[0];
            }
            if (!read_tiff_values(data, size, strip_type, strip_n, strip_val, &strips) ||
                !read_tiff_values(data, size, bytes_type, bytes_n, bytes_val, &counts) || strips.empty() || strips.size() != counts.size()) {
                return fail("TIFF pixel data is truncated");
            }
            const uint64_t row_bytes = static_cast<uint64_t>(width) * 6;
            std::vector<uint8_t> packed;
            packed.reserve(static_cast<size_t>(row_bytes * height));
            uint32_t filled = 0;
            for (size_t strip_i = 0; strip_i < strips.size(); ++strip_i) {
                if (filled >= height) {
                    return fail("TIFF picture size is unusable");
                }
                const uint32_t take = rows < (height - filled) ? rows : (height - filled);
                const uint64_t nbytes = row_bytes * take;
                if (counts[strip_i] != nbytes || static_cast<uint64_t>(strips[strip_i]) + nbytes > size) {
                    return fail("TIFF pixel data is truncated");
                }
                const uint8_t* strip_px = data + strips[strip_i];
                packed.insert(packed.end(), strip_px, strip_px + nbytes);
                filled += take;
            }
            if (filled != height) {
                return fail("TIFF picture size is unusable");
            }
            *image = sd::zeros<float>({static_cast<int64_t>(width), static_cast<int64_t>(height), 3, 1});
            const uint8_t* px = packed.data();
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const size_t at = (static_cast<size_t>(y) * width + x) * 6 + static_cast<size_t>(c) * 2;
                        const uint16_t sample = static_cast<uint16_t>(px[at] | (px[at + 1] << 8));
                        image->index(static_cast<int64_t>(x), static_cast<int64_t>(y), c, 0) = static_cast<float>(sample) / 65535.f;
                    }
                }
            }
            return true;
        }

        bool same_shape(const sd::Tensor<float>& a, const sd::Tensor<float>& b) {
            return a.shape() == b.shape();
        }

        // Flux 2 latents are channel-normalized, not kept in [-1, 1]. The percentile
        // is the range that stays. Dividing by it would rescale the whole tensor.
        sd::Tensor<float> apply_dynamic_threshold(const sd::Tensor<float>& latent, float percentile) {
            if (latent.empty() || !(percentile > 0.f) || percentile >= 1.f) {
                return latent;
            }
            const int64_t n = latent.numel();
            if (n < 1) {
                return latent;
            }
            std::vector<float> magnitudes(static_cast<size_t>(n));
            const float* src = latent.data();
            for (int64_t i = 0; i < n; ++i) {
                magnitudes[static_cast<size_t>(i)] = std::fabs(src[i]);
            }
            size_t rank = static_cast<size_t>(static_cast<double>(percentile) * static_cast<double>(n - 1));
            if (rank >= static_cast<size_t>(n)) {
                rank = static_cast<size_t>(n - 1);
            }
            std::nth_element(magnitudes.begin(), magnitudes.begin() + static_cast<std::ptrdiff_t>(rank), magnitudes.end());
            const float limit = magnitudes[rank];
            sd::Tensor<float> out = latent;
            float* dst            = out.data();
            for (int64_t i = 0; i < n; ++i) {
                if (dst[i] > limit) {
                    dst[i] = limit;
                } else if (dst[i] < -limit) {
                    dst[i] = -limit;
                }
            }
            return out;
        }

        // Caller holds session->mu. The parent card is not written.
        int fork_variant(Session* session,
                         StableDiffusionGGML* sd,
                         int parent_id,
                         const char* kind,
                         sd::Tensor<float> clean,
                         std::string* error) {
            auto fail = [&](const char* message) {
                if (error) {
                    *error = message;
                }
                return 0;
            };
            if (session == nullptr || sd == nullptr || kind == nullptr || clean.empty()) {
                return fail("That card has no clean prediction");
            }
            if (session->running) {
                return fail("A generation is still running");
            }
            auto it = session->cards.find(parent_id);
            if (it == session->cards.end() || it->second.clean.empty()) {
                return fail("That card has no clean prediction");
            }
            const Card& kept = it->second;
            if (!same_shape(clean, kept.clean)) {
                return fail("Edited latent does not match this card");
            }
            sd::Tensor<float> x = clean;
            if (kept.sigma > 1e-6f) {
                sd::Tensor<float> noise = (kept.x_after - kept.clean * (1.f - kept.sigma)) / kept.sigma;
                x                       = clean * (1.f - kept.sigma) + noise * kept.sigma;
            }
            std::vector<uint8_t> noisy, clean_rgb;
            int nw = 0, nh = 0, nc = 0, cw = 0, ch = 0, cc = 0;
            project_rgb(sd, x, &noisy, &nw, &nh, &nc);
            project_rgb(sd, clean, &clean_rgb, &cw, &ch, &cc);
            Card card;
            card.id         = session->next_id++;
            card.parent     = parent_id;
            card.kind       = kind;
            card.settings   = kept.settings;
            card.sigma      = kept.sigma;
            card.sigma_in   = kept.sigma;
            card.eta        = kept.eta;
            card.sigma_ref  = kept.sigma_ref;
            card.ancestral  = false;
            card.seed       = kept.seed;
            card.x_before   = clean;
            card.x_after    = std::move(x);
            card.clean      = std::move(clean);
            card.rng_before = kept.rng_after;
            card.rng_after  = kept.rng_after;
            card.tail       = kept.tail;
            card.noisy_rgb  = std::move(noisy);
            card.clean_rgb  = std::move(clean_rgb);
            card.img_w      = nw > 0 ? nw : cw;
            card.img_h      = nh > 0 ? nh : ch;
            card.img_c      = nc > 0 ? nc : cc;
            const int id    = card.id;
            session->cards.emplace(id, std::move(card));
            return id;
        }

    }  // namespace

    sd::Tensor<float> apply_haar_dcw(const sd::Tensor<float>& latent, float sigma, float sigma_ref, float amount) {
        if (amount == 0.f || latent.dim() < 2 || latent.empty()) {
            return latent;
        }
        const int64_t width  = latent.shape()[0];
        const int64_t height = latent.shape()[1];
        if (width < 2 || height < 2 || (width % 2) != 0 || (height % 2) != 0) {
            return latent;
        }
        const int64_t plane  = width * height;
        const int64_t planes = latent.numel() / plane;
        if (plane <= 0 || planes <= 0 || plane * planes != latent.numel()) {
            return latent;
        }
        const float ref = sigma_ref > 1e-8f ? sigma_ref : 1.f;
        float s         = sigma / ref;
        s               = std::min(1.f, std::max(0.f, s));
        const float low  = 1.f + amount * s;
        const float high = 1.f + amount * (1.f - s);
        sd::Tensor<float> out = latent;
        const float* src      = latent.data();
        float* dst            = out.data();
        for (int64_t p = 0; p < planes; ++p) {
            const float* in = src + p * plane;
            float* o        = dst + p * plane;
            for (int64_t y = 0; y < height; y += 2) {
                for (int64_t x = 0; x < width; x += 2) {
                    const float a  = in[x + width * y];
                    const float b  = in[x + 1 + width * y];
                    const float c  = in[x + width * (y + 1)];
                    const float d  = in[x + 1 + width * (y + 1)];
                    float ll       = (a + b + c + d) * 0.25f * low;
                    float lh       = (a - b + c - d) * 0.25f * high;
                    float hl       = (a + b - c - d) * 0.25f * high;
                    float hh       = (a - b - c + d) * 0.25f * high;
                    o[x + width * y]               = ll + lh + hl + hh;
                    o[x + 1 + width * y]           = ll - lh + hl - hh;
                    o[x + width * (y + 1)]         = ll + lh - hl - hh;
                    o[x + 1 + width * (y + 1)]     = ll - lh - hl + hh;
                }
            }
        }
        return out;
    }

    bool session_query(StableDiffusionGGML* sd, int card_id, CardCopy* out) {
        if (out == nullptr) {
            return false;
        }
        *out = CardCopy();
        if (sd == nullptr) {
            return false;
        }
        std::shared_ptr<Session> session = session_get(sd, false);
        if (!session) {
            return true;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        out->has_cards = !session->cards.empty();
        out->running   = session->running;
        out->width     = session->width;
        out->height    = session->height;
        out->sigma_ref = session->sigma_ref;
        auto it        = session->cards.find(card_id);
        if (it != session->cards.end()) {
            copy_card(it->second, out);
            out->has_cards = true;
            out->running   = session->running;
            out->width     = session->width;
            out->height    = session->height;
            if (out->sigma_ref <= 0.f) {
                out->sigma_ref = session->sigma_ref;
            }
        }
        return true;
    }

    bool session_begin(StableDiffusionGGML* sd, const RunStamp& stamp, int width, int height, bool planting, std::string* error) {
        auto fail = [&](const char* message) {
            if (error) {
                *error = message;
            }
            return false;
        };
        if (sd == nullptr || width < 1 || height < 1) {
            return fail("Latent Tree is missing a picture size");
        }
        std::shared_ptr<Session> session = session_get(sd, true);
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->closed) {
            return fail("Latent Tree session is closed");
        }
        if (session->running) {
            return fail("A generation is still running");
        }
        if (planting && !session->cards.empty()) {
            return fail("The root latent already exists");
        }
        if (!session->cards.empty() && session->width > 0 && (session->width != width || session->height != height)) {
            return fail("Resolution is locked for this session");
        }
        if (!planting && stamp.anchor > 0 && session->cards.find(stamp.anchor) == session->cards.end()) {
            return fail("That card is gone");
        }
        session->running = true;
        session->error.clear();
        session->width  = width;
        session->height = height;
        if (session->sigma_ref <= 0.f && stamp.sigma_ref > 0.f) {
            session->sigma_ref = stamp.sigma_ref;
        }
        session->stamp   = stamp;
        session->last_id = stamp.anchor;
        if (session->sigma_ref > 0.f) {
            session->stamp.sigma_ref = session->sigma_ref;
        }
        return true;
    }

    void session_record(StableDiffusionGGML* sd, LatentStepRecord rec) {
        if (sd == nullptr) {
            return;
        }
        std::shared_ptr<Session> session = session_get(sd, false);
        if (!session) {
            return;
        }
        std::vector<uint8_t> noisy;
        std::vector<uint8_t> clean;
        int nw = 0, nh = 0, nc = 0, cw = 0, ch = 0, cc = 0;
        project_rgb(sd, rec.x_after, &noisy, &nw, &nh, &nc);
        project_rgb(sd, rec.clean, &clean, &cw, &ch, &cc);
        std::lock_guard<std::mutex> lock(session->mu);
        if (!session->running) {
            return;
        }
        Card card;
        card.id        = session->next_id++;
        card.parent    = session->last_id;
        session->last_id = card.id;
        card.kind      = "step";
        card.settings  = session->stamp.settings;
        card.sigma     = rec.sigma;
        card.sigma_in  = rec.sigma_in;
        card.eta       = session->stamp.eta;
        card.sigma_ref = session->stamp.sigma_ref > 0.f ? session->stamp.sigma_ref : session->sigma_ref;
        card.ancestral = session->stamp.ancestral;
        card.seed      = session->stamp.seed;
        card.x_before  = std::move(rec.x_before);
        card.x_after   = std::move(rec.x_after);
        card.clean     = std::move(rec.clean);
        card.rng_before = std::move(rec.rng_before);
        card.rng_after  = std::move(rec.rng_after);
        if (session->stamp.sigmas != nullptr && rec.step >= 0 && rec.step < static_cast<int>(session->stamp.sigmas->size())) {
            card.tail.assign(session->stamp.sigmas->begin() + rec.step, session->stamp.sigmas->end());
        } else if (rec.sigma > 0.f) {
            card.tail = {rec.sigma, 0.f};
        } else {
            card.tail = {0.f};
        }
        card.noisy_rgb = std::move(noisy);
        card.clean_rgb = std::move(clean);
        card.img_w     = nw > 0 ? nw : cw;
        card.img_h     = nh > 0 ? nh : ch;
        card.img_c     = nc > 0 ? nc : cc;
        session->cards.emplace(card.id, std::move(card));
    }

    void session_end(StableDiffusionGGML* sd) {
        if (sd == nullptr) {
            return;
        }
        std::shared_ptr<Session> session = session_get(sd, false);
        if (!session) {
            return;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        session->running      = false;
        session->stamp.sigmas = nullptr;
    }

    std::string session_json(StableDiffusionGGML* sd) {
        std::shared_ptr<Session> session = sd == nullptr ? nullptr : session_get(sd, false);
        if (!session) {
            return "{\"open\":false,\"running\":false,\"cards\":[]}";
        }
        std::lock_guard<std::mutex> lock(session->mu);
        std::vector<int> ids;
        ids.reserve(session->cards.size());
        for (const auto& entry : session->cards) {
            ids.push_back(entry.first);
        }
        std::sort(ids.begin(), ids.end());
        std::string out = "{\"open\":true,\"running\":";
        out += session->running ? "true" : "false";
        out += ",\"width\":" + std::to_string(session->width);
        out += ",\"height\":" + std::to_string(session->height);
        out += ",\"error\":\"" + json_escape(session->error) + "\",\"cards\":[";
        bool first = true;
        for (int id : ids) {
            const Card& card = session->cards.at(id);
            if (!first) {
                out += ',';
            }
            first = false;
            const int remaining = card.tail.size() > 0 ? static_cast<int>(card.tail.size()) - 1 : 0;
            char sigma[32];
            std::snprintf(sigma, sizeof(sigma), "%.8g", card.sigma);
            out += "{\"id\":" + std::to_string(card.id);
            out += ",\"parent\":" + std::to_string(card.parent);
            out += ",\"sigma\":" + std::string(sigma);
            out += ",\"kind\":\"" + card.kind + "\"";
            out += ",\"ancestral\":";
            out += card.ancestral ? "true" : "false";
            out += ",\"remaining\":" + std::to_string(remaining);
            out += ",\"seed\":" + std::to_string(card.seed);
            out += ",\"settings\":" + settings_object(card.settings);
            out += '}';
        }
        out += "]}";
        return out;
    }

    bool session_card_rgb(StableDiffusionGGML* sd, int card_id, int which, std::vector<uint8_t>* rgb, int* width, int* height, int* channels) {
        if (sd == nullptr || rgb == nullptr) {
            return false;
        }
        std::shared_ptr<Session> session = session_get(sd, false);
        if (!session) {
            return false;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        auto it = session->cards.find(card_id);
        if (it == session->cards.end()) {
            return false;
        }
        const std::vector<uint8_t>& src = which == 1 ? it->second.clean_rgb : it->second.noisy_rgb;
        if (src.empty() || it->second.img_w < 1 || it->second.img_h < 1 || it->second.img_c < 1) {
            return false;
        }
        *rgb = src;
        if (width) {
            *width = it->second.img_w;
        }
        if (height) {
            *height = it->second.img_h;
        }
        if (channels) {
            *channels = it->second.img_c;
        }
        return true;
    }

    bool session_close(StableDiffusionGGML* sd, std::string* error) {
        if (sd == nullptr) {
            return true;
        }
        std::shared_ptr<Session> session = session_get(sd, false);
        if (!session) {
            return true;
        }
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (session->running) {
                if (error) {
                    *error = "A generation is still running";
                }
                return false;
            }
            session->closed = true;
        }
        std::lock_guard<std::mutex> lock(g_map_mu);
        auto it = g_sessions.find(sd);
        if (it != g_sessions.end() && it->second == session) {
            g_sessions.erase(it);
        }
        return true;
    }

    void session_release(StableDiffusionGGML* sd) {
        if (sd == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(g_map_mu);
        g_sessions.erase(sd);
    }

    bool session_bake(StableDiffusionGGML* sd, int card_id, sd_image_t* out, std::string* error) {
        if (out != nullptr) {
            *out = sd_image_t{0, 0, 0, nullptr};
        }
        CardCopy card;
        if (!session_query(sd, card_id, &card) || !card.found || card.clean.empty()) {
            if (error) {
                *error = "That card has no clean prediction";
            }
            return false;
        }
        sd::Tensor<float> image = sd->decode_first_stage(card.clean);
        if (image.empty()) {
            if (error) {
                *error = "Bake decode failed";
            }
            return false;
        }
        if (out != nullptr) {
            *out = tensor_to_sd_image(image);
        }
        return true;
    }

    bool session_paint_export(StableDiffusionGGML* sd, int card_id, std::vector<uint8_t>* tiff, std::string* error) {
        CardCopy card;
        if (!session_query(sd, card_id, &card) || !card.found || card.clean.empty()) {
            if (error) {
                *error = "That card has no clean prediction";
            }
            return false;
        }
        sd::Tensor<float> image = sd->decode_first_stage(card.clean);
        if (image.empty()) {
            if (error) {
                *error = "Paint decode failed";
            }
            return false;
        }
        return tensor_to_tiff(image, tiff, error);
    }

    bool session_paint_import(StableDiffusionGGML* sd, int card_id, const uint8_t* data, size_t size, std::string* error) {
        CardCopy parent;
        if (!session_query(sd, card_id, &parent) || !parent.found || parent.clean.empty()) {
            if (error) {
                *error = "That card has no clean prediction";
            }
            return false;
        }
        if (parent.running) {
            if (error) {
                *error = "A generation is still running";
            }
            return false;
        }
        sd::Tensor<float> picture;
        if (!tiff_to_tensor(data, size, parent.width, parent.height, &picture, error)) {
            return false;
        }
        sd::Tensor<float> encoded = sd->encode_first_stage(picture);
        if (encoded.empty() || !same_shape(encoded, parent.clean)) {
            if (error) {
                *error = "Encoded picture does not match this latent";
            }
            return false;
        }
        std::shared_ptr<Session> session = session_get(sd, false);
        if (!session) {
            if (error) {
                *error = "Latent Tree session is closed";
            }
            return false;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        return fork_variant(session.get(), sd, card_id, "paint", std::move(encoded), error) > 0;
    }

    bool session_operate(StableDiffusionGGML* sd,
                         int card_id,
                         const char* op,
                         float amount,
                         int* child_id,
                         std::string* error) {
        if (child_id != nullptr) {
            *child_id = 0;
        }
        auto fail = [&](const char* message) {
            if (error) {
                *error = message;
            }
            return false;
        };
        if (sd == nullptr || op == nullptr) {
            return fail("Unknown operation");
        }
        const std::string which(op);
        if (which != "dcw" && which != "threshold") {
            return fail("Unknown operation");
        }
        if (!std::isfinite(amount)) {
            return fail("That amount is unusable");
        }
        if (which == "dcw") {
            if (amount == 0.f) {
                return fail("Amount 0 leaves the latent unchanged");
            }
            if (!(amount > 0.f && amount <= 1.f)) {
                return fail("That amount is unusable");
            }
        } else if (amount >= 1.f) {
            return fail("Percentile 1 leaves the latent unchanged");
        } else if (!(amount >= 0.5f)) {
            return fail("That percentile is unusable");
        }
        std::shared_ptr<Session> session = session_get(sd, false);
        if (!session) {
            return fail("Latent Tree session is closed");
        }
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->running) {
            return fail("A generation is still running");
        }
        auto it = session->cards.find(card_id);
        if (it == session->cards.end() || it->second.clean.empty()) {
            return fail("That card has no clean prediction");
        }
        const Card& kept = it->second;
        sd::Tensor<float> edited;
        if (which == "dcw") {
            const int64_t width  = kept.clean.dim() > 0 ? kept.clean.shape()[0] : 0;
            const int64_t height = kept.clean.dim() > 1 ? kept.clean.shape()[1] : 0;
            if (width < 2 || height < 2 || (width % 2) != 0 || (height % 2) != 0) {
                return fail("DCW needs an even latent");
            }
            const float ref = kept.sigma_ref > 1e-8f ? kept.sigma_ref : session->sigma_ref;
            edited          = apply_haar_dcw(kept.clean, kept.sigma, ref, amount);
        } else {
            edited = apply_dynamic_threshold(kept.clean, amount);
        }
        if (edited.empty()) {
            return fail("That card has no clean prediction");
        }
        const int child = fork_variant(session.get(), sd, card_id, which.c_str(), std::move(edited), error);
        if (child <= 0) {
            return false;
        }
        if (child_id != nullptr) {
            *child_id = child;
        }
        return true;
    }

}  // namespace sd::tree
