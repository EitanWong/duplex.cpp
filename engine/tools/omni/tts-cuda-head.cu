#include "tts-cuda-head.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

struct tts_cuda_head {
    int outputs = 0, width = 0, device = 0;
    cudaStream_t stream = nullptr;
    float * weights = nullptr;
    float * input = nullptr;
    float * output = nullptr;
    float * host_input = nullptr;
    float * host_output = nullptr;
};

static bool head_cuda_ok(cudaError_t status) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "TTS CUDA head: %s; using CPU fallback\n", cudaGetErrorString(status));
    return false;
}

__global__ static void acoustic_head(const float * weights, const float * hidden,
                                     float * logits, int outputs, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outputs) return;
    float sum = 0.0f;
    // Unrolling exposes independent loads, while explicit rounding intrinsics
    // preserve the ascending, non-fused scalar accumulation for every output.
    #pragma unroll 8
    for (int j = 0; j < width; ++j) {
        sum = __fadd_rn(sum, __fmul_rn(hidden[j], weights[size_t(j) * outputs + i]));
    }
    logits[i] = sum;
}

void tts_cuda_head_free(tts_cuda_head * head) {
    if (!head) return;
    cudaSetDevice(head->device);
    if (head->stream) cudaStreamSynchronize(head->stream);
    if (head->weights) cudaFree(head->weights);
    if (head->input) cudaFree(head->input);
    if (head->output) cudaFree(head->output);
    if (head->host_input) cudaFreeHost(head->host_input);
    if (head->host_output) cudaFreeHost(head->host_output);
    if (head->stream) cudaStreamDestroy(head->stream);
    delete head;
}

tts_cuda_head * tts_cuda_head_create(const float * rows, int outputs, int width, int device) {
    if (!rows || outputs <= 0 || width <= 0) return nullptr;
    auto * head = new (std::nothrow) tts_cuda_head();
    if (!head) return nullptr;
    head->outputs = outputs;
    head->width = width;
    head->device = device;
    if (!head_cuda_ok(cudaSetDevice(device)) ||
        !head_cuda_ok(cudaStreamCreateWithFlags(&head->stream, cudaStreamNonBlocking)) ||
        !head_cuda_ok(cudaMalloc(&head->weights, size_t(outputs) * width * sizeof(float))) ||
        !head_cuda_ok(cudaMalloc(&head->input, size_t(width) * sizeof(float))) ||
        !head_cuda_ok(cudaMalloc(&head->output, size_t(outputs) * sizeof(float))) ||
        !head_cuda_ok(cudaMallocHost(&head->host_input, size_t(width) * sizeof(float))) ||
        !head_cuda_ok(cudaMallocHost(&head->host_output, size_t(outputs) * sizeof(float)))) {
        tts_cuda_head_free(head);
        return nullptr;
    }
    std::vector<float> transposed(size_t(outputs) * width);
    for (int j = 0; j < width; ++j) {
        for (int i = 0; i < outputs; ++i) {
            transposed[size_t(j) * outputs + i] = rows[size_t(i) * width + j];
        }
    }
    if (!head_cuda_ok(cudaMemcpy(head->weights, transposed.data(), transposed.size() * sizeof(float), cudaMemcpyHostToDevice))) {
        tts_cuda_head_free(head);
        return nullptr;
    }
    return head;
}

bool tts_cuda_head_forward(tts_cuda_head * head, const float * hidden, float * logits) {
    if (!head || !hidden || !logits || !head_cuda_ok(cudaSetDevice(head->device))) return false;
    std::memcpy(head->host_input, hidden, size_t(head->width) * sizeof(float));
    if (!head_cuda_ok(cudaMemcpyAsync(head->input, head->host_input, size_t(head->width) * sizeof(float),
                                     cudaMemcpyHostToDevice, head->stream))) return false;
    acoustic_head<<<(head->outputs + 127) / 128, 128, 0, head->stream>>>(
        head->weights, head->input, head->output, head->outputs, head->width);
    if (!head_cuda_ok(cudaGetLastError()) ||
        !head_cuda_ok(cudaMemcpyAsync(head->host_output, head->output, size_t(head->outputs) * sizeof(float),
                                     cudaMemcpyDeviceToHost, head->stream)) ||
        !head_cuda_ok(cudaStreamSynchronize(head->stream))) return false;
    std::memcpy(logits, head->host_output, size_t(head->outputs) * sizeof(float));
    return true;
}
