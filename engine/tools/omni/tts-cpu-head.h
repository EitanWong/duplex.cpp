#pragma once

#include <vector>

// Eight output rows per SIMD lane group. Each row retains the scalar dot
// product's multiply/add order; there is no horizontal reduction or FMA.
bool tts_cpu_head_pack(const float * rows, int outputs, int hidden, std::vector<float> & packed);
void tts_cpu_head_forward(const float * packed, const float * hidden, int outputs, int width, float * logits);
// Four independent output groups hide add latency while preserving each dot
// product's order. Uses the same packed weights and handles the tail above.
void tts_cpu_head_forward_interleaved(const float * packed, const float * hidden, int outputs, int width, float * logits);
