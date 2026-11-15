#include "tts-cuda-head.h"
#include "tts-cpu-head.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    constexpr int outputs = 6562, width = 768, repeats = 100;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    std::vector<float> weights(size_t(outputs) * width), hidden(width), cpu(outputs), gpu(outputs), packed;
    for (auto & x : weights) x = dist(rng);
    if (argc > 1) {
        std::ifstream source(argv[1], std::ios::binary);
        source.read(reinterpret_cast<char *>(weights.data()), weights.size() * sizeof(float));
        if (!source) return 2;
    }
    if (!tts_cpu_head_pack(weights.data(), outputs, width, packed)) return 3;
    auto * head = tts_cuda_head_create(weights.data(), outputs, width, 0);
    if (!head) return 4;
    double cpu_ms = 0, gpu_ms = 0;
    bool valid = true;
    const auto measure = [](auto fn) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    for (int trial = -5; trial < repeats; ++trial) {
        for (auto & x : hidden) x = dist(rng);
        const auto cpu_fn = [&] { tts_cpu_head_forward(packed.data(), hidden.data(), outputs, width, cpu.data()); };
        const auto gpu_fn = [&] { valid &= tts_cuda_head_forward(head, hidden.data(), gpu.data()); };
        double a, b;
        if (trial % 2) { a = measure(cpu_fn); b = measure(gpu_fn); }
        else { b = measure(gpu_fn); a = measure(cpu_fn); }
        if (!valid || std::memcmp(cpu.data(), gpu.data(), outputs * sizeof(float))) {
            std::fprintf(stderr, "FAIL: CUDA head differs at trial %d\n", trial);
            tts_cuda_head_free(head);
            return 1;
        }
        if (trial >= 0) { cpu_ms += a; gpu_ms += b; }
    }
    tts_cuda_head_free(head);
    std::printf("PASS: %d x %d logits byte-identical; AVX2=%.4f ms CUDA=%.4f ms speedup=%.3fx\n",
                repeats, outputs, cpu_ms / repeats, gpu_ms / repeats, cpu_ms / gpu_ms);
    return 0;
}
