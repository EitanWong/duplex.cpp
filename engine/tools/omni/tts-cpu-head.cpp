#include "tts-cpu-head.h"

#include <algorithm>
#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

bool tts_cpu_head_pack(const float * rows, int outputs, int hidden, std::vector<float> & packed) {
#if defined(__AVX2__)
    if (!rows || outputs <= 0 || hidden <= 0) {
        return false;
    }
    const int blocks = (outputs + 7) / 8;
    packed.assign(size_t(blocks) * hidden * 8, 0.0f);
    for (int block = 0; block < blocks; ++block) {
        for (int lane = 0; lane < std::min(8, outputs - block * 8); ++lane) {
            for (int j = 0; j < hidden; ++j) {
                packed[(size_t(block) * hidden + j) * 8 + lane] = rows[size_t(block * 8 + lane) * hidden + j];
            }
        }
    }
    return true;
#else
    (void) rows; (void) outputs; (void) hidden; (void) packed;
    return false;
#endif
}

void tts_cpu_head_forward(const float * packed, const float * hidden, int outputs, int width, float * logits) {
#if defined(__AVX2__)
    for (int i = 0; i < outputs; i += 8) {
        const float * block = packed + size_t(i / 8) * width * 8;
        __m256 sums = _mm256_setzero_ps();
        for (int j = 0; j < width; ++j) {
            const __m256 values = _mm256_loadu_ps(block + size_t(j) * 8);
            sums = _mm256_add_ps(sums, _mm256_mul_ps(values, _mm256_set1_ps(hidden[j])));
        }
        if (outputs - i >= 8) {
            _mm256_storeu_ps(logits + i, sums);
        } else {
            float tail[8];
            _mm256_storeu_ps(tail, sums);
            std::copy_n(tail, outputs - i, logits + i);
        }
    }
#else
    (void) packed; (void) hidden; (void) outputs; (void) width; (void) logits;
#endif
}

void tts_cpu_head_forward_interleaved(const float * packed, const float * hidden, int outputs, int width, float * logits) {
#if defined(__AVX2__)
    int i = 0;
    const size_t stride = size_t(width) * 8;
    for (; i + 31 < outputs; i += 32) {
        const float * block = packed + size_t(i / 8) * stride;
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        for (int j = 0; j < width; ++j) {
            const __m256 h = _mm256_set1_ps(hidden[j]);
            const size_t offset = size_t(j) * 8;
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(block + offset), h));
            b = _mm256_add_ps(b, _mm256_mul_ps(_mm256_loadu_ps(block + stride + offset), h));
            c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_loadu_ps(block + 2 * stride + offset), h));
            d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(block + 3 * stride + offset), h));
        }
        _mm256_storeu_ps(logits + i, a);
        _mm256_storeu_ps(logits + i + 8, b);
        _mm256_storeu_ps(logits + i + 16, c);
        _mm256_storeu_ps(logits + i + 24, d);
    }
    tts_cpu_head_forward(packed + size_t(i / 8) * stride, hidden, outputs - i, width, logits + i);
#else
    (void) packed; (void) hidden; (void) outputs; (void) width; (void) logits;
#endif
}
