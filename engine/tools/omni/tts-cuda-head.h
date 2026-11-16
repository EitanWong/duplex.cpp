#pragma once

struct tts_cuda_head;

// Each output follows the scalar CPU multiply/add order. No FMA or horizontal
// reduction; the transposed weight layout coalesces adjacent CUDA lanes.
tts_cuda_head * tts_cuda_head_create(const float * rows, int outputs, int width, int device);
bool tts_cuda_head_forward(tts_cuda_head * head, const float * hidden, float * logits);
void tts_cuda_head_free(tts_cuda_head * head);
