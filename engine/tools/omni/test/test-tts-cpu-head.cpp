#include "tts-cpu-head.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

// Optional row-major F32 weights allow the same check against the real GGUF
// acoustic head. Synthetic signed inputs also exercise a partial lane group.
int main(int argc, char ** argv) {
    constexpr int outputs = 6562, width = 768, repeats = 100;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    std::vector<float> rows(size_t(outputs) * width), hidden(width), scalar(outputs), simd(outputs), interleaved(outputs), packed;
    for (auto & x : rows) { x = dist(rng); }
    if (argc > 1) {
        std::ifstream source(argv[1], std::ios::binary);
        source.read(reinterpret_cast<char *>(rows.data()), rows.size() * sizeof(float));
        if (!source) { return 2; }
    }
    if (!tts_cpu_head_pack(rows.data(), outputs, width, packed)) {
        std::puts("SKIP: AVX2 CPU head specialization not compiled");
        return 0;
    }
    const auto baseline = [&] {
        for (int i = 0; i < outputs; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < width; ++j) { sum += hidden[j] * rows[size_t(i) * width + j]; }
            scalar[i] = sum;
        }
    };
    const auto optimized = [&] { tts_cpu_head_forward(packed.data(), hidden.data(), outputs, width, simd.data()); };
    const auto ilp = [&] { tts_cpu_head_forward_interleaved(packed.data(), hidden.data(), outputs, width, interleaved.data()); };
    double scalar_ms = 0, simd_ms = 0, ilp_ms = 0;
    for (int trial = 0; trial < repeats; ++trial) {
        for (auto & x : hidden) { x = dist(rng); }
        // Alternate measurement order to reduce systematic cache bias.
        auto measure = [](const auto & fn) {
            const auto start = std::chrono::steady_clock::now();
            fn();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        };
        if (trial % 3 == 0) { scalar_ms += measure(baseline); simd_ms += measure(optimized); ilp_ms += measure(ilp); }
        else if (trial % 3 == 1) { simd_ms += measure(optimized); ilp_ms += measure(ilp); scalar_ms += measure(baseline); }
        else { ilp_ms += measure(ilp); scalar_ms += measure(baseline); simd_ms += measure(optimized); }
        if (std::memcmp(scalar.data(), simd.data(), outputs * sizeof(float)) ||
            std::memcmp(scalar.data(), interleaved.data(), outputs * sizeof(float))) {
            std::fprintf(stderr, "FAIL: acoustic logits differ at trial %d\n", trial);
            return 1;
        }
    }
    std::printf("PASS: %d x %d logits byte-identical; scalar=%.4f ms AVX2=%.4f ms interleaved=%.4f ms ILP_speedup=%.3fx extra_bytes=%zu\n",
                repeats, outputs, scalar_ms / repeats, simd_ms / repeats, ilp_ms / repeats, simd_ms / ilp_ms, packed.size() * sizeof(float));
    // Exercise both sides of the 32-output loop and the partial SIMD tail with
    // an odd hidden width. Guard values detect stores beyond the output span.
    for (const int n : {1, 7, 8, 31, 32, 33, 63, 64, 65}) {
        constexpr int w = 13;
        if (!tts_cpu_head_pack(rows.data(), n, w, packed)) return 3;
        std::vector<float> a(n + 2, 123.0f), b = a;
        tts_cpu_head_forward(packed.data(), hidden.data(), n, w, a.data() + 1);
        tts_cpu_head_forward_interleaved(packed.data(), hidden.data(), n, w, b.data() + 1);
        if (a != b || b.front() != 123.0f || b.back() != 123.0f) return 4;
    }
    std::puts("PASS: interleaved output boundaries and guard values");
    return 0;
}
