#include "omni.h"
#include "audition.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

// Hardware regression probe. Pass consecutive chunks followed by a long input
// to exercise history rollover and allocator growth; compare the F32 dumps.
int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s audio.gguf output.f32 input.wav [input.wav ...]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    auto * ctx = audition_init(argv[1], {true, GGML_LOG_LEVEL_INFO});
    if (!ctx) return 2;
    std::ofstream dump(argv[2], std::ios::binary);
    if (!dump) { audition_free(ctx); return 3; }
    int status = 0;
    for (int i = 3; i < argc; ++i) {
        const auto start = std::chrono::steady_clock::now();
        auto * emb = omni_audio_embed_make_with_filename(ctx, 4, argv[i]);
        if (!emb) { status = 4; break; }
        const size_t count = size_t(emb->n_pos) * audition_n_mmproj_embd(ctx);
        for (size_t j = 0; j < count; ++j) {
            if (!std::isfinite(emb->embed[j])) { status = 5; break; }
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("input=%d tokens=%d values=%zu ms=%.3f\n", i - 3, emb->n_pos, count, ms);
        dump.write(reinterpret_cast<const char *>(emb->embed), count * sizeof(float));
        omni_embed_free(emb);
        if (!dump || status) { if (!status) status = 6; break; }
    }
    audition_free(ctx);
    return status;
}
