#include "omni.h"
#include "token2wav/token2wav-impl.h"
#include "tts-condition-graph.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Hardware probe: compare conditioning graphs with the actual TTS/projector
// weights, covering allocator growth, repeated shapes and vocabulary boundaries.
int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s tts.gguf projector.gguf\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    omni_context ctx;
    const auto cleanup = [&] {
        tts_condition_graph_free(&ctx);
        projector_free(ctx.projector);
        std::free(ctx.emb_code_weight);
        std::free(ctx.emb_text_weight);
        std::free(ctx.head_code_weight);
        std::free(ctx.projector_semantic_linear1_weight);
        std::free(ctx.projector_semantic_linear1_bias);
        std::free(ctx.projector_semantic_linear2_weight);
        std::free(ctx.projector_semantic_linear2_bias);
    };
    if (!load_tts_weights_from_gguf(&ctx, argv[1]) || !projector_init(ctx.projector, argv[2], true)) {
        cleanup();
        return 3;
    }
    if (std::strncmp(ggml_backend_name(ctx.projector.backend), "CUDA", 4) != 0) {
        std::fprintf(stderr, "CUDA projector required for this hardware probe\n");
        cleanup();
        return 4;
    }
    const int width = ctx.projector.layer.linear1_weight->ne[0];
    const int shapes[] = {1, 10, 40, 2, 96, 10};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<std::vector<float>> hidden, reference;
    std::vector<std::vector<llama_token>> tokens;
    for (int n : shapes) {
        hidden.emplace_back(size_t(n) * width);
        tokens.emplace_back(n);
        for (auto & x : hidden.back()) x = dist(rng);
        for (auto & id : tokens.back()) id = rng() % ctx.emb_text_vocab_size;
        tokens.back().front() = 0;
        tokens.back().back() = ctx.emb_text_vocab_size - 1;
    }
    size_t compared = 0;
    bool valid = true;
    for (int host = 0; host <= 1 && valid; ++host) {
#ifdef _WIN32
        _putenv_s("OMNI_TTS_HOST_EMBED", host ? "1" : "0");
#else
        setenv("OMNI_TTS_HOST_EMBED", host ? "1" : "0", 1);
#endif
        tts_condition_graph_free(&ctx);
        if (!tts_condition_graph_init(&ctx)) { valid = false; break; }
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::vector<float> result;
            int output_width = 0;
            if (!tts_condition_graph_forward(&ctx, tokens[i].data(), hidden[i].data(),
                    int(tokens[i].size()), width, result, output_width)) { valid = false; break; }
            for (float x : result) valid &= std::isfinite(x);
            if (!host) reference.push_back(std::move(result));
            else {
                valid &= output_width == ctx.emb_text_hidden_size && result.size() == reference[i].size();
                if (valid && std::memcmp(result.data(), reference[i].data(), result.size() * sizeof(float))) {
                    std::fprintf(stderr, "FAIL: conditioning differs for %zu tokens\n", tokens[i].size());
                    valid = false;
                }
                compared += result.size();
            }
            if (!valid) break;
        }
    }
    cleanup();
    if (!valid) return 1;
    std::printf("PASS: %zu conditioning floats byte-identical across six shapes\n", compared);
    return 0;
}
