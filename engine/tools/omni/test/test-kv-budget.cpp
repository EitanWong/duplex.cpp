#include "kv-budget.h"
#include "../../server/fact-memory.h"

#include <cstdio>
#include <stdexcept>

static void check(bool ok) { if (!ok) throw std::runtime_error("KV budget/memory assertion failed"); }

int main() {
    try {
        // Real boundaries: 4K context, protected voice/system prefix, 77-token
        // AV unit, generation headroom. Keep the unfinished turn intact.
        auto p = omni_plan_kv(4096, 200, 4000, 77, 32, {800, 1600, 3000}, {800, 1600, 3000, 3900});
        check(p.fits && p.end == 800 && !p.unit_fallback);
        p = omni_plan_kv(4096, 200, 4000, 77, 32, {}, {277, 1500, 3900});
        check(p.fits && p.unit_fallback && p.end == 1500);
        p = omni_plan_kv(4096, 200, 4000, 3900, 32, {3000}, {3900});
        check(!p.fits); // Input + prefix + reserve alone cannot fit.
        p = omni_plan_kv(4096, 200, 4000, 77, 32, {}, {});
        check(!p.fits); // Never cut an unregistered active input.
        p = omni_plan_kv(2048, 200, 1900, 77, 32, {500}, {500});
        check(p.fits && p.end == 200); // No premature crop.
        // Sweep uneven multimodal unit lengths, small contexts and oversized
        // batches. Every successful plan fits and ends on a real boundary.
        for (int cap : {512, 2048, 4096, 8192, 16384}) {
            for (int incoming : {1, 77, 512, cap}) {
                std::vector<int> ends;
                for (int end = 200; end <= cap - 20; end += 77) ends.push_back(end);
                auto plan = omni_plan_kv(cap, 100, cap - 20, incoming, 32, {}, ends);
                if (plan.fits) {
                    check(cap - 20 - (plan.end - 100) + incoming + 32 <= cap);
                    check(plan.end == 100 || std::find(ends.begin(), ends.end(), plan.end) != ends.end());
                }
            }
        }
        using json = nlohmann::ordered_json;
        omni_fact_memory memory;
        auto fact = [](const char * id, const char * text, int turn) {
            return json{{"id", id}, {"text", text}, {"source", "user_text"},
                {"speaker", "user"}, {"time", "2026-11-15T22:00:00+08:00"},
                {"turn", turn}, {"negated", false}, {"status", "fact"}};
        };
        memory.update(json::array({fact("name", "Name is Lan", 0), fact("number", "Room 17", 0)}), 1);
        auto correction = fact("number-v2", "Room is 23, not 17", 1);
        correction["supersedes"] = "number";
        correction["negated"] = true;
        memory.update(json::array({correction}), 2);
        auto count = [](const std::string & value) { return (int)value.size(); };
        auto selected = memory.retrieve("number", 4096, count);
        check(selected.find("Room 17") == std::string::npos && selected.find("not 17") != std::string::npos);
        check(selected.find("speaker") != std::string::npos && selected.find("supersedes") != std::string::npos);
        check(memory.retrieve("number", 1, count).empty()); // No partial quotation.
        auto reject = [&](const json & batch, int completed) {
            const auto revision = memory.revision(), size = memory.size();
            bool rejected = false;
            try { memory.update(batch, completed); } catch (const std::exception &) { rejected = true; }
            check(rejected && memory.revision() == revision && memory.size() == size);
        };
        reject(json::array({fact("future", "unfinished", 2)}), 2);
        auto huge_turn = fact("wrapped", "must not wrap into completed turn 1", 0);
        huge_turn["turn"] = uint64_t(4294967297ULL);
        reject(json::array({huge_turn}), 2);
        huge_turn["turn"] = uint64_t(-1);
        reject(json::array({huge_turn}), 2);
        auto assistant = fact("bad-source", "invented transcript", 0); assistant["source"] = "assistant";
        reject(json::array({assistant}), 2);
        reject(json::array({fact("atomic", "valid first row", 0), fact("name", "duplicate", 0)}), 2);
        auto bad_link = fact("bad-link", "correction", 0); bad_link["supersedes"] = "unknown";
        reject(json::array({bad_link}), 2);
        auto pending = fact("task", "Send both names and room numbers tomorrow", 1); pending["status"] = "pending";
        memory.update(json::array({pending}), 2);
        check(memory.retrieve("", 4096, count).find("pending") != std::string::npos);
        auto final_room = fact("number-v3", "Room is 26, not 23 or 17", 1);
        final_room["supersedes"] = "number-v2";
        memory.update(json::array({final_room}), 2);
        selected = memory.retrieve("number", 4096, count);
        check(selected.find("Room is 23") == std::string::npos && selected.find("Room is 26") != std::string::npos);
        while (memory.size() < 128) {
            auto id = "noise-" + std::to_string(memory.size());
            memory.update(json::array({fact(id.c_str(), "unrelated observation", 0)}), 2);
        }
        reject(json::array({fact("overflow", "reject without forgetting provenance", 0)}), 2);
        selected = memory.retrieve("number-v3", 512, count);
        check(count(selected) <= 512 && selected.find("Room is 26") != std::string::npos);
        check(selected.find("Room is 23") == std::string::npos);
        std::puts("PASS: complete-unit budgets, oversized/active boundaries, transactional provenance/corrections/retrieval");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
