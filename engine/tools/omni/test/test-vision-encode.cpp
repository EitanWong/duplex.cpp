#include "omni.h"
#include "vision.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

// Hardware regression probe: dump real image embeddings for attention A/B,
// including the multi-slice/batched layout. Models and dumps stay external.
int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s vision.gguf image.jpg output.f32 [slices=1] [batch=0] [preprocessed.f32] [--preprocess-only]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    auto * ctx = vision_init(argv[1], {true, GGML_LOG_LEVEL_WARN, nullptr});
    if (!ctx) return 2;
    vision_set_max_slice_nums(ctx, argc > 4 ? std::atoi(argv[4]) : 1);
    const bool batch = argc > 5 && std::atoi(argv[5]) != 0;
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    vision_image_u8 image{};
    vision_image_f32_batch images;
    const auto pre_start = std::chrono::steady_clock::now();
    if (bytes.empty() || !vision_image_load_from_bytes(bytes.data(), bytes.size(), &image) ||
        !vision_image_preprocess(ctx, &image, &images)) return 3;
    std::printf("preprocess_ms=%.3f\n", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pre_start).count());
    if (argc > 6) {
        std::ofstream pre_dump(argv[6], std::ios::binary);
        for (auto & pre_image : images.entries) {
            pre_dump.write(reinterpret_cast<const char *>(pre_image->buf.data()), pre_image->buf.size() * sizeof(float));
        }
        if (!pre_dump) return 7;
    }
    if (argc > 7 && std::strcmp(argv[7], "--preprocess-only") == 0) {
        vision_free(ctx);
        return 0;
    }
    const size_t stride = (size_t)vision_n_output_tokens(ctx) * vision_n_mmproj_embd(ctx);
    std::vector<float> output(stride * images.entries.size() * (batch ? 2 : 1));
    for (int repeat = 0; repeat < 6; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        size_t offset = 0;
        // Production encodes the overview and each preprocessed slice this way.
        for (auto & image_f32 : images.entries) {
            if (batch) {
                vision_image_f32_batch pair;
                pair.entries.emplace_back(new vision_image_f32(*image_f32));
                pair.entries.emplace_back(new vision_image_f32(*image_f32));
                if (!vision_image_batch_encode(ctx, 4, &pair, output.data() + offset)) return 4;
                offset += stride * 2;
            } else {
                if (!vision_image_encode(ctx, 4, image_f32.get(), output.data() + offset)) return 4;
                offset += stride;
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        std::printf("repeat=%d slices=%zu values=%zu ms=%.3f\n", repeat, images.entries.size(), output.size(), ms);
    }
    for (float value : output) if (!std::isfinite(value)) return 5;
    std::ofstream dump(argv[3], std::ios::binary);
    dump.write(reinterpret_cast<const char *>(output.data()), output.size()*sizeof(float));
    if (!dump) return 6;
    vision_free(ctx);
    return 0;
}
