// Isolated text history/recall probe. This is not an audio transcription or a
// speech-quality test. Pair it with pipeline.py at the same cache precision.
#include "llama.h"
#include "ggml-backend.h"
#include "../../engine/tools/server/fact-memory.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> out(text.size() + 32);
    int n = llama_tokenize(vocab, text.data(), (int)text.size(), out.data(), (int)out.size(), false, true);
    if (n < 0) { out.resize(-n); n = llama_tokenize(vocab, text.data(), (int)text.size(), out.data(), (int)out.size(), false, true); }
    if (n < 0) throw std::runtime_error("tokenize failed");
    out.resize(n); return out;
}
static void decode(llama_context * ctx, const std::vector<llama_token> & tokens, int offset, int count, int position) {
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[offset+i]; batch.pos[i] = position+i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = i == count-1;
    }
    int rc = llama_decode(ctx, batch); llama_batch_free(batch);
    if (rc) throw std::runtime_error("decode failed");
}
int main(int argc, char ** argv) {
    if (argc != 6 && argc != 7) { std::fprintf(stderr, "usage: long-history MODEL CTX DEPTH f16|q8_0 full|memory|crowded-memory|recent [TEST_INDEX]\n"); return 1; }
    try {
        const int capacity = std::stoi(argv[2]), depth = std::stoi(argv[3]);
        const std::string type = argv[4], mode = argv[5];
        if (depth < 512 || depth + 96 > capacity || capacity > 40960 ||
            (type != "f16" && type != "q8_0") || (mode != "full" && mode != "memory" && mode != "crowded-memory" && mode != "recent")) throw std::runtime_error("invalid configuration");
        ggml_backend_load_all();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) return 2;
        auto cp = llama_context_default_params(); cp.n_ctx = capacity; cp.n_batch = 512; cp.n_ubatch = 128;
        cp.n_threads = 4; cp.n_threads_batch = 8; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        auto * ctx = llama_init_from_model(model, cp); if (!ctx) { llama_model_free(model); return 3; }
        const auto * vocab = llama_model_get_vocab(model);
        const std::vector<std::string> facts = {
            "项目名是青鹭。门禁码是7319。", "楼层不是17楼，已纠正为23楼。",
            "English site name is Amber Cabin; cabinet key is B42.",
            "Alice只喝茶，不喝咖啡。", "周五之前给Alice发两份中文报告；这件事尚未完成。"};
        const std::vector<std::string> questions = {
            "请仅输出项目名和门禁码。", "楼层经过纠正后是几楼？只输出楼层。",
            "What are the English site name and cabinet key?",
            "Alice喝什么，不喝什么？", "尚未完成的任务是什么，发给谁，何时截止，几份，用什么语言？",
            "请联合输出：纠正后的楼层、门禁码、为Alice准备的饮料；三项都要。", "原始视频中门的颜色、说话者音色各是什么？未提供就回答UNKNOWN。"};
        omni_fact_memory memory;
        json rows = json::array();
        for (size_t i = 0; i < facts.size(); ++i) rows.push_back({{"id", "fact-"+std::to_string(i)}, {"source", "user_text"}, {"speaker", "user"},
            {"time", "2026-11-15T22:00:00+08:00"}, {"turn", 0}, {"text", facts[i]}, {"negated", i == 1 || i == 3}, {"status", i == 4 ? "pending" : "fact"}});
        memory.update(rows, 1);
        if (mode == "crowded-memory") {
            // External typed test quotations, not transcripts of fixture speech.
            // Model the immutable correction chain and a completed task.
            auto corrected = rows[1]; corrected["id"] = "floor-v2";
            corrected["text"] = "楼层再次纠正：现在是26楼，不是23楼也不是17楼。";
            corrected["supersedes"] = "fact-1"; corrected["turn"] = 1;
            memory.update(json::array({corrected}), 2);
            auto done = rows[4]; done["id"] = "task-done";
            done["text"] = "周五之前给Alice发两份中文报告的任务已完成。";
            done["supersedes"] = "fact-4"; done["status"] = "done"; done["turn"] = 2;
            memory.update(json::array({done}), 3);
            for (int i = 0; i < 112; ++i) {
                auto noise = rows[0]; noise["id"] = "unrelated-" + std::to_string(i);
                noise["text"] = "Garden log " + std::to_string(i) + ": green leaves, wind calm, path dry.";
                noise["speaker"] = "observer"; noise["turn"] = 2;
                memory.update(json::array({noise}), 3);
            }
        }
        const std::vector<std::string> expected = {
            "青鹭;7319", mode == "crowded-memory" ? "26;exclude obsolete 17/23" : "23;exclude obsolete 17",
            "Amber Cabin;B42", "tea;not coffee", mode == "crowded-memory" ? "no unfinished report task;already done" : "Alice;Friday;two;Chinese;pending",
            mode == "crowded-memory" ? "26;7319;tea" : "23;7319;tea", "UNKNOWN;no visual or voice fact supplied"};
        auto filler = tokenize(vocab, "这里记录无关的天气观察：天空有云，风速平稳。The garden has green leaves and quiet paths.\n");
        for (size_t test = 0; test < questions.size(); ++test) {
            if (argc == 7 && (int)test != std::stoi(argv[6])) continue;
            llama_memory_clear(llama_get_memory(ctx), false);
            std::string first = "<|im_start|>system\nYou answer questions using stated user facts. Respect corrections and negation. If facts are unavailable say UNKNOWN. Answer briefly. Do not think aloud.<|im_end|>\n<|im_start|>user\n";
            if (mode == "full") for (const auto & fact : facts) first += fact + "\n";
            first += "\nUnrelated observations follow:\n";
            auto input = tokenize(vocab, first);
            std::string end = "\n<|im_end|>\n<|im_start|>user\n";
            std::string retrieved;
            double retrieval_ms = 0;
            if (mode == "memory" || mode == "crowded-memory") {
                const double retrieval_started = now_ms();
                retrieved = memory.retrieve(questions[test], 512, [&](const std::string & text) { return (int)tokenize(vocab, text).size(); });
                retrieval_ms = now_ms() - retrieval_started;
                end += retrieved + "\n";
            }
            end += questions[test] + " /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            const auto tail = tokenize(vocab, end);
            const int filler_end = depth - (int)tail.size();
            if ((int)input.size() > filler_end) throw std::runtime_error("depth too small");
            while ((int)input.size() < filler_end) {
                const int count = std::min((int)filler.size(), filler_end - (int)input.size());
                input.insert(input.end(), filler.begin(), filler.begin()+count);
            }
            input.insert(input.end(), tail.begin(), tail.end());
            const double begin = now_ms();
            for (int pos = 0; pos < depth; pos += 512) decode(ctx, input, pos, std::min(512, depth-pos), pos);
            llama_synchronize(ctx);
            const double prefill_ms = now_ms()-begin;
            llama_kv_cache_stats stats = {}; llama_memory_get_kv_stats(llama_get_memory(ctx), &stats);
            std::string answer; int generated = 0; double first_ms = 0;
            for (; generated < 80; ++generated) {
                const auto * logits = llama_get_logits_ith(ctx, -1);
                const llama_token token = (llama_token)(std::max_element(logits, logits+llama_vocab_n_tokens(vocab))-logits);
                if (!generated) first_ms = now_ms()-begin;
                if (llama_vocab_is_eog(vocab, token)) break;
                char piece[1024]; int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
                if (n > 0) answer.append(piece, n);
                decode(ctx, {token}, 0, 1, depth+generated);
            }
            llama_synchronize(ctx);
            json out = {{"test", test}, {"expected_semantics", expected[test]}, {"mode", mode}, {"k_v", type}, {"question", questions[test]}, {"answer", answer},
                {"depth", depth}, {"used", stats.used_cells}, {"physical", stats.physical_span}, {"scan", stats.attention_span},
                {"allocated_bytes", stats.allocated_bytes}, {"prefill_ms", prefill_ms}, {"ttft_ms", first_ms}, {"generated", generated},
                {"total_ms", now_ms()-begin}, {"speech_quality_accepted", false},
                {"memory_records", memory.size()}, {"retrieved_tokens", tokenize(vocab, retrieved).size()},
                {"retrieved", retrieved}, {"retrieval_ms", retrieval_ms}};
            std::puts(out.dump().c_str()); std::fflush(stdout);
        }
        llama_free(ctx); llama_model_free(model); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 4; }
}
