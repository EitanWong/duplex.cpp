// Public-API diagnostic: logical KV pruning versus packed state restoration.
// State restoration is a diagnostic reference, not a recommended serving path.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static int physical_span = -1;
static int live_cells = -1;

static void log_callback(ggml_log_level level, const char * text, void *) {
    int span, used;
    const char * cells = std::strstr(text, ": stream[0], n =");
    if (cells && std::sscanf(cells, ": stream[0], n = %d, used = %d", &span, &used) == 2) {
        physical_span = span;
        live_cells = used;
    }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void decode(llama_context * ctx, int position, int count) {
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = 100 + (position + i) % 100;
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == count - 1);
    }
    int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (result) throw std::runtime_error("decode failed");
    llama_synchronize(ctx);
}

static double measure(llama_context * ctx, int start, int count) {
    double begin = now_ms();
    for (int i = 0; i < count; ++i) decode(ctx, start + i, 1);
    return now_ms() - begin;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: kv-fragmentation <LLM.gguf>\n");
        return 1;
    }
    // Set this in the parent shell: separate Windows CRTs can cache different
    // environments, so _putenv_s in this executable cannot configure llama.dll.
    if (!std::getenv("LLAMA_KV_CACHE_DEBUG")) {
        std::fprintf(stderr, "Set LLAMA_KV_CACHE_DEBUG=1 in the parent shell before launching.\n");
        return 1;
    }
    llama_log_set(log_callback, nullptr);
    ggml_backend_load_all();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) return 2;
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 128;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 8;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * fragmented = llama_init_from_model(model, ctx_params);
    auto * packed = llama_init_from_model(model, ctx_params);
    if (!fragmented || !packed) {
        if (packed) llama_free(packed);
        if (fragmented) llama_free(fragmented);
        llama_model_free(model);
        return 3;
    }
    try {
        // Repeat against one loaded model; alternate timing order to limit drift.
        for (int trial = 0; trial < 6; ++trial) {
            llama_memory_clear(llama_get_memory(fragmented), false);
            llama_memory_clear(llama_get_memory(packed), false);
            for (int pos = 0; pos < 3072; pos += 512) decode(fragmented, pos, 512);
            auto memory = llama_get_memory(fragmented);
            if (!llama_memory_seq_rm(memory, 0, 32, 1056)) throw std::runtime_error("prune failed");
            llama_memory_seq_add(memory, 0, 1056, 3072, -1024);
            // Apply the deferred K-shift before serializing states.
            decode(fragmented, 2048, 1);
            const int fragment_span = physical_span;
            const int fragment_live = live_cells;
            if (fragment_span < 0) throw std::runtime_error("KV physical-span diagnostics unavailable");
            double begin = now_ms();
            std::vector<uint8_t> state(llama_state_seq_get_size(fragmented, 0));
            if (llama_state_seq_get_data(fragmented, state.data(), state.size(), 0) != state.size()) {
                throw std::runtime_error("state save failed");
            }
            if (llama_state_seq_set_data(packed, state.data(), state.size(), 0) != state.size()) {
                throw std::runtime_error("state restore failed");
            }
            const double restore_ms = now_ms() - begin;
            decode(fragmented, 2049, 1);
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            auto * logits = llama_get_logits_ith(fragmented, -1);
            std::vector<float> reference(logits, logits + n_vocab);
            decode(packed, 2049, 1);
            const int packed_span = physical_span;
            logits = llama_get_logits_ith(packed, -1);
            double max_difference = 0;
            for (int i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
                max_difference = std::max(max_difference, double(std::abs(logits[i] - reference[i])));
            }
            // Warm both shapes once; the following 128 tokens have matched positions.
            measure(fragmented, 2050, 8);
            measure(packed, 2050, 8);
            double fragment_ms, packed_ms;
            if (trial % 2 == 0) {
                fragment_ms = measure(fragmented, 2058, 128);
                packed_ms = measure(packed, 2058, 128);
            } else {
                packed_ms = measure(packed, 2058, 128);
                fragment_ms = measure(fragmented, 2058, 128);
            }
            std::printf("{\"trial\":%d,\"live_cells_after_prune\":%d,\"physical_span_after_prune\":%d,"
                        "\"packed_span\":%d,\"restore_ms\":%.3f,\"state_bytes\":%zu,"
                        "\"fragmented_ms\":%.3f,\"packed_ms\":%.3f,\"max_logit_difference\":%.9g}\n",
                        trial, fragment_live, fragment_span, packed_span, restore_ms, state.size(),
                        fragment_ms, packed_ms, max_difference);
            std::fflush(stdout);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        llama_free(packed);
        llama_free(fragmented);
        llama_model_free(model);
        return 4;
    }
    llama_free(packed);
    llama_free(fragmented);
    llama_model_free(model);
    return 0;
}
