#pragma once

#include <algorithm>
#include <vector>

// Position boundaries are exclusive ends. Planning never mutates live KV or
// its ledger. Only complete turns, then (at the hard limit) complete units,
// may be evicted. The pending input is never an eviction candidate.
struct omni_kv_plan {
    int end = 0;
    bool fits = true;
    bool unit_fallback = false;
};

inline omni_kv_plan omni_plan_kv(int capacity, int prefix, int past, int incoming,
                                int reserve, const std::vector<int> & turns,
                                const std::vector<int> & units) {
    omni_kv_plan plan;
    plan.end = prefix;
    if (capacity <= 0 || prefix < 0 || past < prefix || incoming < 0 || reserve < 0 ||
        static_cast<long long>(prefix) + incoming + reserve > capacity) {
        plan.fits = false;
        return plan;
    }
    const int required = std::max(0, past - (capacity - incoming - reserve));
    if (!required) return plan;
    const int hysteresis = std::min(1024, capacity / 8);
    const int desired = std::max(required, past - (capacity - incoming - reserve - hysteresis));
    // Keep an unfinished turn whenever removing completed turns is sufficient.
    for (int end : turns) {
        if (end > prefix && end <= past) {
            plan.end = end;
            if (end - prefix >= desired) return plan;
        }
    }
    if (plan.end - prefix >= required) return plan;
    // A single turn can outgrow the entire context. Sacrifice its oldest
    // complete units, never an arbitrary token tail or the current input.
    plan.unit_fallback = true;
    for (int end : units) {
        if (end > plan.end && end <= past) {
            plan.end = end;
            if (end - prefix >= desired) return plan;
        }
    }
    plan.fits = plan.end - prefix >= required;
    return plan;
}
