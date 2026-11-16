// Hardware regression for rejected mutations. Run explicitly with the local
// model after other GPU tests, never as an unconditional model-free CTest.
#include "token2wav/token2wav.h"
#include "omni.h"
#include "common.h"
#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>

static void check(bool value) { if (!value) throw std::runtime_error("KV failure path changed live state"); }

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: test-kv-ledger MODEL.gguf\n"); return 1; }
    ggml_backend_load_all();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp); if (!model) return 2;
    auto cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512;
    cp.n_ubatch = 128; cp.n_threads = 4; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * llm = llama_init_from_model(model, cp); if (!llm) { llama_model_free(model); return 3; }
    int result = 0;
    try {
        auto batch = llama_batch_init(8, 0, 1); batch.n_tokens = 8;
        for (int i = 0; i < 8; ++i) {
            batch.token[i] = 100+i; batch.pos[i] = i; batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0; batch.logits[i] = i == 7;
        }
        check(llama_decode(llm, batch) == 0); llama_batch_free(batch); llama_synchronize(llm);
        omni_context ctx; common_params params;
        ctx.ctx_llama = llm; ctx.n_past = 8; ctx.n_keep = ctx.system_preserve_length = 2;
        ctx.params = &params; ctx.duplex_mode = true; ctx.sliding_window_config.mode = "turn";
        ctx.round_start_positions = {4, 8};
        UnitEntry first; first.length = 2; first.type = "audio";
        UnitEntry second; second.length = 4; second.type = "response";
        ctx.unit_history = {first, second};
        auto unchanged = [&] {
            llama_kv_cache_stats stats = {}; check(llama_memory_get_kv_stats(llama_get_memory(llm), &stats));
            check(stats.used_cells == 8 && llama_memory_seq_pos_max(llama_get_memory(llm), 0) == 7);
            check(ctx.round_start_positions == std::vector<int>({4, 8}));
            check(ctx.unit_history.size() == 2 && ctx.unit_history[0].length == 2 && ctx.unit_history[1].length == 4);
            check(ctx.position_offset == 0 && ctx.kv_evictions.load() == 0);
        };
        check(!sliding_window_drop_tokens_from_cache(&ctx, 7)); unchanged();
        ctx.n_past = 9; // Simulate a stale application count; do not touch KV.
        check(!sliding_window_drop_tokens_from_cache(&ctx, 2)); check(ctx.n_past == 9); unchanged();
        ctx.n_past = 8;
        check(!prefill_with_emb(&ctx, &params, nullptr, 600, 512, &ctx.n_past));
        check(ctx.n_past == 8 && ctx.kv_input_failed); unchanged();
        std::puts("PASS: oversized deletion, stale ledger and impossible complete input leave GPU KV/boundaries intact");
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); result = 4; }
    llama_free(llm); llama_model_free(model); return result;
}
