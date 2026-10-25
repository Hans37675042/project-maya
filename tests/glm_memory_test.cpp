// CPU-only policy checks: no HIP/CUDA headers, device queries or allocations.
#include "../src/core/glm_memory.hpp"

#undef NDEBUG   // the checks are asserts: keep them in Release builds
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {
using strata::core::glmfast::TierPin;
constexpr int64_t kGi = int64_t{1} << 30, kMi = int64_t{1} << 20, kRoom = 512 * kMi;   // kRoom: kPinRoom

// A driver with a pinning cap (STRATA_GLM_PIN_CAP_MB's model of Windows): it refuses what takes the bytes held past cap
struct PinSim {
    int64_t cap = 0, held = 0;
    // pin_room: probes of 1 GiB, then of 256 MiB, up to limit - freed again
    int64_t room(int64_t limit) const {
        int64_t got = 0;
        for (const int64_t probe : {kGi, 256 * kMi})
            while (got + probe <= limit && held + got + probe <= cap) got += probe;
        return got;
    }
};
// One part's RAM tier: its classes' planned slots, slot bytes and floors (16), and its small pinned buffers
struct Part {
    std::vector<int64_t> plan, stride, least;
    int64_t small = 64 * kMi;
    TierPin tp;
    Part(std::vector<int64_t> gib, std::vector<int64_t> st) : stride(st) {
        for (size_t c = 0; c < gib.size(); ++c) plan.push_back(gib[c] / st[c]);
        least.assign(plan.size(), 16);
    }
    int64_t planned() const {
        int64_t b = 0;
        for (size_t c = 0; c < plan.size(); ++c) b += plan[c] * stride[c];
        return b;
    }
    int64_t pinned() const {
        int64_t b = 0;
        for (size_t c = 0; c < tp.n.size(); ++c) b += tp.n[c] * stride[c];
        return b;
    }
};
// fast_setup's calls: a class pins, then the room kept above the tier must pin too (freed again), or the class goes
bool pin_part(PinSim& s, Part& p, const std::vector<int64_t>& n, int64_t keep, int64_t later, int64_t reserve,
              bool measure_later) {
    return strata::core::glmfast::pin_tier_classes(
        p.tp, n, p.stride, p.least, later, reserve, measure_later,
        [&](size_t c, int64_t k) {
            const int64_t b = k * p.stride[c];
            if (s.held + b > s.cap || s.held + b + keep > s.cap) return false;
            s.held += b;
            return true;
        },
        [&](size_t c, int64_t k) { s.held -= k * p.stride[c]; }, [&](int64_t limit, bool) { return s.room(limit); });
}
// A parallel split under a cap as fast_setup runs it: every part plans, then each in turn pins its small buffers and
// its tier at the split's scale - the first one measuring the room for the later ones.  capped false: v1.1.20's split
// (no later parts, one GPU's room), the reference
// ok: false at the first part whose start fails (the later ones are not tried)
bool run_split(PinSim& s, std::vector<Part>& parts, bool capped) {
    using namespace strata::core::glmfast;
    const int np = (int) parts.size();
    const int64_t keep = pin_room_kept(kRoom, np, capped);
    std::vector<int64_t> plan, small;
    for (const Part& p : parts) {
        plan.push_back(p.planned());
        small.push_back(p.small);
    }
    double scale = 1.0;
    bool measured = false;
    for (int i = 0; i < np; ++i) {
        Part& p = parts[(size_t) i];
        s.held += p.small;
        const auto lr = capped ? split_later(plan, small, i, scale, keep) : std::make_pair(int64_t{0}, keep);
        std::vector<int64_t> n(p.plan.size());
        for (size_t c = 0; c < n.size(); ++c) n[c] = scaled_slots(p.plan[c], p.least[c], scale);
        if (!pin_part(s, p, n, keep, lr.first, lr.second, capped && !measured && i + 1 < np)) return false;
        if (lr.first > 0 && p.tp.later_measured) {
            scale *= p.tp.later_scale;
            measured = true;
        }
    }
    return true;
}
void pin_split(PinSim& s, std::vector<Part>& parts, bool capped) {
    const bool ok = run_split(s, parts, capped);
    assert(ok);
}
// v1.1.20's class loop, step for step: what one GPU's tier must still come out as
bool old_tier(PinSim& s, const std::vector<int64_t>& cls_n, const std::vector<int64_t>& stride, std::vector<int64_t>& out) {
    int64_t pin_left = -1;
    out.assign(cls_n.size(), 0);
    const auto share = [&](size_t c) {
        int64_t planned = 0;
        for (size_t k = c; k < cls_n.size(); ++k) planned += cls_n[k] * stride[k];
        const double f = (double) std::max<int64_t>(0, pin_left - kRoom) / (double) std::max<int64_t>(1, planned);
        return (int64_t) (f * (double) cls_n[c]);
    };
    for (size_t c = 0; c < cls_n.size(); ++c) {
        int64_t n = cls_n[c];
        const int64_t least = 16;
        if (pin_left >= 0) n = std::max(least, std::min(n, share(c)));
        bool p = false;
        for (int tries = 0; n >= least; ++tries) {
            if (s.held + n * stride[c] <= s.cap && s.held + n * stride[c] + kRoom <= s.cap) {
                s.held += n * stride[c];
                p = true;
                break;
            }
            if (pin_left < 0) {
                int64_t planned = 0;
                for (size_t k = c; k < cls_n.size(); ++k) planned += cls_n[k] * stride[k];
                pin_left = s.room(planned + kRoom);
                const int64_t fit = share(c);
                if (fit < n) {
                    n = std::max(least, fit);
                    continue;
                }
            }
            const int64_t less = tries < 4 ? n * 31 / 32 : n * 7 / 8;
            n = (n > least && less < least) ? least : less;
        }
        if (p && pin_left >= 0) pin_left = std::max<int64_t>(0, pin_left - n * stride[c]);
        if (!p) return false;
        out[c] = n;
    }
    return true;
}
// Checks added for the dual-GPU pin tier.  A check named in kKnownFail is a finding the current code does not meet
// and this round does not fix: it must still be red (an expected failure), and the run fails if it turns green.
// STRATA_TEST_NO_XFAIL=1 reports those as plain failures.
const std::set<std::string> kKnownFail = {};
int g_checks = 0, g_xfail = 0, g_bad = 0;
void check(const char* id, bool ok, const std::string& detail = "") {
    ++g_checks;
    const bool no_xfail = std::getenv("STRATA_TEST_NO_XFAIL") != nullptr;
    const bool known = !no_xfail && kKnownFail.count(id) != 0;
    if (known) {
        if (ok) {
            ++g_bad;
            std::printf("UNEXPECTED PASS (strict xfail): %s\n", id);
        } else {
            ++g_xfail;
            std::printf("xfail: %s  %s\n", id, detail.c_str());
        }
    } else if (!ok) {
        ++g_bad;
        std::printf("FAIL: %s  %s\n", id, detail.c_str());
    }
}
// a sweep keeps its first failing input
struct Sweep {
    bool ok = true;
    int fails = 0;
    std::string first, last;
    void expect(bool c, const std::string& what) {
        if (c) return;
        if (ok) first = what;
        ok = false;
        ++fails;
        last = what;
    }
    std::string why() const { return first + (fails > 1 ? "  [" + std::to_string(fails) + " failing inputs, last: " + last + "]" : ""); }
};
std::string gib(int64_t b) { return std::to_string((double) b / (double) kGi) + "GiB"; }
// the 64 GiB PC's two-card split: a 30.8 GiB plan and a 13.5 GiB one, in two size classes each
std::vector<Part> two_parts() {
    const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024};
    return {Part({9 * kGi, (int64_t) (21.8 * (double) kGi)}, st), Part({4 * kGi, (int64_t) (9.5 * (double) kGi)}, st)};
}
std::vector<Part> three_parts() {
    const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024};
    return {Part({8 * kGi, 12 * kGi}, st), Part({5 * kGi, 10 * kGi}, st), Part({4 * kGi, 6 * kGi}, st)};
}
// the smallest cap that can hold every floor (16 slots a class), the small buffers and the room kept above them
int64_t floors_cap(const std::vector<Part>& ps, int64_t keep) {
    int64_t b = keep;
    for (const Part& p : ps) {
        b += p.small;
        for (size_t c = 0; c < p.plan.size(); ++c) b += p.least[c] * p.stride[c];
    }
    return b;
}
}  // namespace

int main() {
    using strata::core::glmfast::expert_pool_budget;
    using strata::core::glmfast::minimal_ram_tier;
    using strata::core::glmfast::full_unified_pool;
    using strata::core::glmfast::warm_pool_slots;
    using strata::core::glmfast::complete_pool;
    using strata::core::glmfast::complete_pool_slots;
    constexpr size_t G = size_t{1} << 30;
    constexpr size_t experts = 80 * G + 174 * (size_t{1} << 20);

    // A 128 GiB APU with 110 GiB available after setup: the 112 GiB runtime
    // aperture (and even a tiny reported free value) cannot change the OS budget.
    assert(expert_pool_budget(true, 112 * G, 110 * G, G, 16 * G, experts) == experts);
    assert(expert_pool_budget(true, G, 110 * G, G, 16 * G, experts) == experts);
    // Memory pressure leaves 53 GiB; missing MemAvailable or less than headroom
    // leaves zero rather than wrapping or trusting HIP's much larger free figure.
    assert(expert_pool_budget(true, 112 * G, 70 * G, G, 16 * G, experts) == 53 * G);
    assert(expert_pool_budget(true, 112 * G, 0, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 16 * G, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 17 * G, G, 16 * G, experts) == 0);
    // User headroom/reserve and the discrete formula retain their meaning.
    assert(expert_pool_budget(true, 112 * G, 70 * G, 2 * G, 8 * G, experts) == 60 * G);
    assert(expert_pool_budget(false, 12 * G, 110 * G, 3 * G, 16 * G, experts) == 9 * G);
    assert(expert_pool_budget(false, G, 110 * G, 3 * G, 16 * G, experts) == 0);
    assert(minimal_ram_tier(true, true, false));
    assert(!minimal_ram_tier(true, true, true));  // STRATA_GLM_RAM_GB / supplied budget wins
    assert(!minimal_ram_tier(true, false, false));
    assert(!minimal_ram_tier(false, true, false));
    // Warm every expert when it fits, and never evict residents to make spares.
    // Smaller APU pools and all discrete pools retain the existing spare quota.
    assert(warm_pool_slots(true, 288, 288, 3) == 288);
    assert(full_unified_pool(true, 288, 288));
    assert(warm_pool_slots(true, 192, 288, 3) == 189);
    assert(!full_unified_pool(true, 192, 288));
    assert(warm_pool_slots(false, 288, 288, 3) == 285);
    assert(!full_unified_pool(false, 288, 288));
    // A complete pool (every expert + the spares in the main slots, the prompt's tail beyond them - the Windows APU's
    // carve-out): the warm-up fills each expert once, never past the 288-entry order
    assert(warm_pool_slots(false, 288 + 3 + 21, 288, 3) == 288);
    assert(warm_pool_slots(false, 290, 288, 3) == 287);
    assert(complete_pool(291, 288, 3));
    assert(!complete_pool(290, 288, 3));
    assert(!complete_pool(288 - 21, 288, 3));
    // 158 GiB of budget, 308 MiB a slot over every MoE layer: 288 + 3 + 18 slots fit; 40 GiB do not; nothing on zeros
    constexpr size_t M = size_t{1} << 20;
    assert(complete_pool_slots(158 * G, 308 * M, 288, 3, 18) == 309);
    assert(complete_pool_slots(309 * 308 * M, 308 * M, 288, 3, 18) == 309);
    assert(complete_pool_slots(309 * 308 * M - 1, 308 * M, 288, 3, 18) == 0);
    assert(complete_pool_slots(40 * G, 308 * M, 288, 3, 18) == 0);
    assert(complete_pool_slots(158 * G, 0, 288, 3, 18) == 0);
    assert(complete_pool_slots(158 * G, 308 * M, 288, 3, -5) == 291);
    // The pool's reserve: 700 MiB, except where it leaves the pool short of its floor (here 2 GiB) - an automatic budget
    // then lowers it to leave exactly the floor, never below 300 MiB; one set by hand or a fixed budget keeps it
    using strata::core::glmfast::pool_reserve;
    const size_t R = 700 * M, S = 300 * M;
    assert(pool_reserve(10 * G, R, false, false, 2 * G, S) == std::make_pair(R, false));            // room for both
    assert(pool_reserve(2 * G + 500 * M, R, false, false, 2 * G, S) == std::make_pair(500 * M, true));
    assert(pool_reserve(2 * G + 300 * M, R, false, false, 2 * G, S) == std::make_pair(S, true));
    assert(pool_reserve(2 * G + 299 * M, R, false, false, 2 * G, S) == std::make_pair(R, false));    // short anyway
    assert(pool_reserve(2 * G + 500 * M, R, true, false, 2 * G, S) == std::make_pair(R, false));     // given
    assert(pool_reserve(2 * G + 500 * M, R, false, true, 2 * G, S) == std::make_pair(R, false));     // --expert-cache N
    // The written budget's check (20 GiB asked, 700 MiB reserve): free as planned gives nothing back; 200 MiB free
    // gives back the 500 short + 64; nearly nothing free 1 GiB twice, then a quarter; a refused piece what it refused
    using strata::core::glmfast::pool_give_back;
    int zr = 0;
    assert(pool_give_back(20 * G, 20 * G, 700 * M, R, zr) == 0);
    assert(pool_give_back(20 * G, 20 * G, 640 * M, R, zr) == 0);
    assert(pool_give_back(20 * G, 20 * G, 200 * M, R, zr) == 564 * M);
    assert(pool_give_back(20 * G, 20 * G, 0, R, zr) == G && zr == 1);
    assert(pool_give_back(20 * G, 20 * G, 0, R, zr) == G && zr == 2);
    assert(pool_give_back(20 * G, 20 * G, 0, R, zr) == 5 * G && zr == 3);
    zr = 0;
    assert(pool_give_back(20 * G, 18 * G, 2 * G, R, zr) == 2 * G);
    assert(pool_give_back(20 * G, 18 * G, 100 * M, R, zr) == 2 * G + 664 * M);
    assert(pool_give_back(G, 0, 0, R, zr) == G);   // never more than was asked
    // Windows' commit on an APU (128 GiB RAM: 12.8 GiB kept for the other programs).  A 140 GiB budget, a pool that can
    // take at most 120 GiB, 4 GiB its floor: 100 GiB of commit left caps it at 87.2 GiB and keeps 32.8 off the GPU;
    // with 150 GiB left it is capped to 137.2, still above all the pool takes, so nothing stays off; with 130 it is
    // capped to 117.2 and 2.8 GiB stay off; with no room at all the floor stays; a 16 GiB PC keeps 8 GiB for the others
    using strata::core::glmfast::commit_margin;
    using strata::core::glmfast::commit_pool_cap;
    using strata::core::glmfast::pagefile_hint_gib;
    assert(commit_margin(128 * G) == 128 * G / 10 && commit_margin(16 * G) == 8 * G);
    const size_t mg = 128 * G / 10;
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 100 * G, 128 * G) == std::make_pair(100 * G - mg, 20 * G + mg));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 150 * G, 128 * G) == std::make_pair(150 * G - mg, size_t{0}));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 200 * G, 128 * G) == std::make_pair(140 * G, size_t{0}));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 130 * G, 128 * G) == std::make_pair(130 * G - mg, mg - 10 * G));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 5 * G, 128 * G) == std::make_pair(4 * G, 116 * G));
    assert(commit_pool_cap(140 * G, 4 * G, 0, 100 * G, 128 * G) == std::make_pair(100 * G - mg, 40 * G + mg));
    assert(commit_pool_cap(2 * G, 4 * G, 120 * G, 0, 128 * G) == std::make_pair(2 * G, size_t{0}));   // (<= budget)
    assert(commit_pool_cap(10 * G, G, 0, 12 * G, 16 * G) == std::make_pair(4 * G, 6 * G));
    // the page file that keeps 33 GiB more on the GPU, with 128 GiB of RAM and a 32 GiB page file now: 65 GiB
    assert(pagefile_hint_gib(160 * G, 128 * G, 33 * G) == 65);
    assert(pagefile_hint_gib(160 * G, 128 * G, 33 * G - 1) == 65 && pagefile_hint_gib(128 * G, 128 * G, 1) == 1);

    // The RAM tier under a pinning cap (pin_tier_classes).  The room kept above the tiers: one GPU's 512 MiB, or under
    // a cap one for each GPU of a split and one for their warm-ups; what a split's part pins beside; the scaled slots
    using strata::core::glmfast::pin_room_kept;
    using strata::core::glmfast::split_later;
    using strata::core::glmfast::scaled_slots;
    assert(pin_room_kept(kRoom, 1, true) == kRoom && pin_room_kept(kRoom, 2, false) == kRoom);
    assert(pin_room_kept(kRoom, 2, true) == 3 * kRoom && pin_room_kept(kRoom, 3, true) == 4 * kRoom);
    assert(split_later({30 * kGi, 13 * kGi}, {64 * kMi, 64 * kMi}, 0, 1.0, 3 * kRoom) ==
           std::make_pair(13 * kGi, 3 * kRoom + 64 * kMi));
    assert(split_later({30 * kGi, 13 * kGi}, {64 * kMi, 64 * kMi}, 1, 1.0, 3 * kRoom) ==
           std::make_pair(int64_t{0}, 3 * kRoom));
    assert(split_later({30 * kGi, 13 * kGi, 8 * kGi}, {kMi, kMi, kMi}, 0, 0.5, kRoom) ==
           std::make_pair(21 * kGi / 2, kRoom + 2 * kMi));
    assert(scaled_slots(100, 16, 1.0) == 100 && scaled_slots(100, 16, 0.5) == 50 && scaled_slots(20, 16, 0.5) == 16);
    // One GPU (no later parts, 512 MiB kept): the tier comes out as v1.1.20's loop made it, step for step - caps that
    // hold it all, cut it at the first class or a later one, and leave a class only its floor or not even that
    {
        const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024, 4096 * 1024};
        for (int64_t cap = 0; cap <= 64 * kGi; cap += 97 * kMi + 12345) {
            for (const auto& gib : {std::vector<int64_t>{9 * kGi, 41 * kGi, kGi}, std::vector<int64_t>{30 * kGi},
                                    std::vector<int64_t>{2 * kGi, 200 * kMi, 20 * kGi}}) {
                const std::vector<int64_t> s3(st.begin(), st.begin() + (std::ptrdiff_t) gib.size());
                Part p(gib, s3);
                PinSim a{cap, 0}, b{cap, 0};
                std::vector<int64_t> ref;
                const bool ok_ref = old_tier(a, p.plan, p.stride, ref);
                const bool ok = pin_part(b, p, p.plan, kRoom, 0, kRoom, false);
                assert(ok == ok_ref && b.held == a.held && !p.tp.later_measured && !p.tp.repinned);
                if (ok) assert(p.tp.n == ref);
            }
        }
        // this PC's one-GPU start: a 9 GiB class pins, the 41 GiB one meets the cap, 35.5 GiB more pin - the tier
        // ~44 GiB, 512 MiB+ still free above it
        Part p({9 * kGi, 41 * kGi}, {6656 * 1024, 7168 * 1024});
        PinSim s{(int64_t) (44.6 * (double) kGi), 0};
        assert(pin_part(s, p, p.plan, kRoom, 0, kRoom, false));
        assert(p.tp.n[0] == p.plan[0] && p.pinned() > 43 * kGi + 768 * kMi && s.cap - s.held >= kRoom);
    }
    // Two GPUs under the cap of this PC's 64 GiB (~44.6 GiB of tier pinned once nothing else was): the first part
    // plans 30.8 GiB, the second 13.5.  v1.1.20 pinned the first one whole and left the second what was left - here
    // 44.0 GiB in all and 0.6 free, as STRATA_GLM_RAM_GB=44's start whose warmed tables did not upload; now the first
    // one measures the room for the second after its tier, finds it short, and both pin ~97% of their plans (43.0 GiB,
    // as =43's start), 1.5 GiB+ (a GPU's room each and one for the warm-ups) still free
    {
        const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024};
        std::vector<Part> was = {Part({9 * kGi, (int64_t) (21.8 * (double) kGi)}, st),
                                 Part({4 * kGi, (int64_t) (9.5 * (double) kGi)}, st)};
        PinSim w{(int64_t) (44.75 * (double) kGi), 0};
        pin_split(w, was, false);
        assert(was[0].pinned() == was[0].planned() && w.cap - w.held < 2 * kRoom);
        std::vector<Part> parts = {Part({9 * kGi, (int64_t) (21.8 * (double) kGi)}, st),
                                   Part({4 * kGi, (int64_t) (9.5 * (double) kGi)}, st)};
        PinSim s{(int64_t) (44.75 * (double) kGi), 0};
        pin_split(s, parts, true);
        const double f0 = (double) parts[0].pinned() / (double) parts[0].planned();
        const double f1 = (double) parts[1].pinned() / (double) parts[1].planned();
        assert(parts[0].tp.repinned && parts[0].tp.later_room >= 0 && !parts[1].tp.later_measured);
        assert(f0 > 0.95 && f0 < 1.0 && std::fabs(f0 - f1) < 0.01);
        assert(parts[1].pinned() > 12 * kGi + 768 * kMi);
        assert(s.cap - s.held >= 3 * kRoom);
        // ... with room for both: the first part's probe finds it, nothing pins again, both pin all of their plans
        std::vector<Part> big = {Part({9 * kGi, (int64_t) (21.8 * (double) kGi)}, st),
                                 Part({4 * kGi, (int64_t) (9.5 * (double) kGi)}, st)};
        PinSim r{100 * kGi, 0};
        pin_split(r, big, true);
        assert(big[0].tp.later_measured && big[0].tp.later_scale == 1.0 && !big[0].tp.repinned);
        assert(big[0].pinned() == big[0].planned() && big[1].pinned() == big[1].planned());
        assert(r.held == big[0].planned() + big[1].planned() + 2 * 64 * kMi);
        // ... a cap the first part's own plan meets (32 GiB): v1.1.20 kept the first part whole and gave the second
        // under a tenth of its plan; now the first part's refusal measures the room for both - its classes from the
        // refused one on and the second part each pin ~60% of their plans (its first class pinned whole before)
        std::vector<Part> old = {Part({9 * kGi, (int64_t) (21.8 * (double) kGi)}, st),
                                 Part({4 * kGi, (int64_t) (9.5 * (double) kGi)}, st)};
        PinSim o{32 * kGi, 0};
        pin_split(o, old, false);
        assert(old[0].pinned() == old[0].planned() && old[1].pinned() < old[1].planned() / 10);
        std::vector<Part> tight = {Part({9 * kGi, (int64_t) (21.8 * (double) kGi)}, st),
                                   Part({4 * kGi, (int64_t) (9.5 * (double) kGi)}, st)};
        PinSim t{32 * kGi, 0};
        pin_split(t, tight, true);
        const double g0 = (double) tight[0].pinned() / (double) tight[0].planned();
        const double g1 = (double) tight[1].pinned() / (double) tight[1].planned();
        const double c1 = (double) tight[0].tp.n[1] / (double) tight[0].plan[1];
        assert(!tight[0].tp.repinned && tight[0].tp.later_room < 0 && tight[0].tp.later_measured);
        assert(tight[0].tp.n[0] == tight[0].plan[0] && g0 > g1 && g1 > 0.55 && std::fabs(c1 - g1) < 0.01);
        assert(t.cap - t.held >= 3 * kRoom);
    }
    // Three GPUs, 40 GiB under the cap, plans of 20, 15 and 10 GiB: each ~84%, 2 GiB still free; the later two pin
    // at the first one's measurement without probing again
    {
        const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024};
        std::vector<Part> parts = {Part({8 * kGi, 12 * kGi}, st), Part({5 * kGi, 10 * kGi}, st),
                                   Part({4 * kGi, 6 * kGi}, st)};
        PinSim s{40 * kGi, 0};
        pin_split(s, parts, true);
        double lo = 1.0, hi = 0.0;
        for (const Part& p : parts) {
            const double f = (double) p.pinned() / (double) p.planned();
            lo = std::min(lo, f);
            hi = std::max(hi, f);
        }
        assert(lo > 0.8 && hi < 0.87 && hi - lo < 0.01);
        assert(parts[0].tp.later_measured && !parts[1].tp.later_measured && !parts[2].tp.later_measured);
        assert(s.cap - s.held >= 4 * kRoom);
    }
    // ---- Dual-GPU pin tier: boundaries, error paths and silent wrong values (check/xfail harness above) ----
    using strata::core::glmfast::pin_tier_classes;
    {   // pin_room_kept (spec: a capped split keeps one room per GPU plus the warm-ups'; one GPU or uncapped keeps one)
        check("room_kept.zero_room", pin_room_kept(0, 1, true) == 0 && pin_room_kept(0, 2, true) == 0 &&
                                         pin_room_kept(0, 3, false) == 0);
        check("room_kept.n_parts_edges", pin_room_kept(kRoom, 0, true) == kRoom && pin_room_kept(kRoom, 1, false) == kRoom &&
                                             pin_room_kept(kRoom, 3, false) == kRoom && pin_room_kept(kRoom, 8, true) == 9 * kRoom);
    }
    {   // split_later (spec: a part's later tiers at the split's scale, reserve = keep + the later parts' small buffers)
        const std::vector<int64_t> plan = {30 * kGi, 13 * kGi}, small = {64 * kMi, 32 * kMi};
        check("split_later.past_the_end", split_later(plan, small, 2, 1.0, kRoom) == std::make_pair(int64_t{0}, kRoom) &&
                                              split_later(plan, small, 99, 1.0, kRoom) == std::make_pair(int64_t{0}, kRoom));
        check("split_later.negative_part_counts_all",
              split_later(plan, small, -1, 1.0, kRoom) == std::make_pair(43 * kGi, kRoom + 96 * kMi));
        check("split_later.scale_zero_keeps_reserve",
              split_later(plan, small, 0, 0.0, kRoom) == std::make_pair(int64_t{0}, kRoom + 32 * kMi));
        check("split_later.empty_and_single", split_later({}, {}, 0, 1.0, kRoom) == std::make_pair(int64_t{0}, kRoom) &&
                                                  split_later({7 * kGi}, {kMi}, 0, 1.0, kRoom) == std::make_pair(int64_t{0}, kRoom));
        check("split_later.length_mismatch_ignores_unmatched",
              split_later({30 * kGi, 13 * kGi, 8 * kGi}, {kMi, kMi}, 0, 1.0, kRoom) == std::make_pair(13 * kGi, kRoom + kMi) &&
                  split_later({30 * kGi, 13 * kGi}, {kMi, kMi, kMi}, 0, 1.0, kRoom) == std::make_pair(13 * kGi, kRoom + kMi));
    }
    {   // scaled_slots (spec: never negative, never under the floor, never over the plan, scale 1.0 changes nothing)
        check("scaled.scale_one_is_plan", scaled_slots(100, 16, 1.0) == 100 && scaled_slots(0, 16, 1.0) == 0 &&
                                              scaled_slots(100, 16, 7.0) == 100);
        check("scaled.floor_cases", scaled_slots(100, 16, 0.0) == 16 && scaled_slots(100, 16, -1.0) == 16 &&
                                        scaled_slots(0, 16, 0.5) == 16 && scaled_slots(32, 16, 0.5) == 16 &&
                                        scaled_slots(33, 16, 0.5) == 16 && scaled_slots(34, 16, 0.5) == 17);
        Sweep sw;
        for (int64_t n = 16; n <= 400; n += 7)
            for (double sc = 0.0; sc < 1.0; sc += 0.0625) {
                const int64_t v = scaled_slots(n, 16, sc);
                sw.expect(v >= 16 && v <= n, "n=" + std::to_string(n) + " scale=" + std::to_string(sc) + " -> " + std::to_string(v));
                sw.expect(scaled_slots(n, 16, sc + 0.0625) >= v || sc + 0.0625 >= 1.0, "monotone n=" + std::to_string(n));
            }
        check("scaled.in_floor_and_plan_range", sw.ok, sw.why());
        // a plan already under the floor: the floor wins over the plan (16, not 10; scaled.floor_cases pins the plan 0
        // case; fast_setup never plans under the floor)
        check("scaled.floor_wins_over_plan_under_floor",
              scaled_slots(10, 16, 0.5) == 16 && scaled_slots(10, 16, 0.0) == 16 && scaled_slots(10, 16, 1.0) == 10,
              "scaled_slots(10, 16, 0.5) = " + std::to_string(scaled_slots(10, 16, 0.5)));
    }
    {   // pin_tier_classes, one part: boundaries and error paths
        const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024};
        {   // no classes at all
            TierPin tp;
            const bool ok = pin_tier_classes(tp, {}, {}, {}, 0, kRoom, false, [](size_t, int64_t) { return false; },
                                             [](size_t, int64_t) {}, [](int64_t, bool) { return int64_t{0}; });
            check("tier.no_classes", ok && tp.n.empty() && tp.pin_left == -1);
        }
        {   // a cap exactly the plan plus the room kept: all of it, no refusal; one byte less: not all, never over the cap
            Part p({30 * kGi}, {st[0]});
            PinSim s{p.planned() + kRoom, 0};
            const bool ok = pin_part(s, p, p.plan, kRoom, 0, kRoom, false);
            check("tier.cap_equals_plan_plus_keep", ok && p.tp.n[0] == p.plan[0] && p.tp.pin_left == -1 && s.held + kRoom <= s.cap);
            Part q({30 * kGi}, {st[0]});
            PinSim t{q.planned() + kRoom - 1, 0};
            const bool ok2 = pin_part(t, q, q.plan, kRoom, 0, kRoom, false);
            check("tier.cap_one_byte_short", ok2 && q.tp.n[0] < q.plan[0] && q.tp.n[0] >= 16 && t.held + kRoom <= t.cap,
                  "ok=" + std::to_string(ok2) + " n=" + std::to_string(q.tp.n[0]) + " plan=" + std::to_string(q.plan[0]));
        }
        {   // a class planned at 0 slots: the start fails exactly as v1.1.20's loop did (n 0 is under the floor)
            Part p({0, 5 * kGi}, st);
            PinSim a{64 * kGi, 0}, b{64 * kGi, 0};
            std::vector<int64_t> ref;
            const bool r = old_tier(a, p.plan, p.stride, ref);
            const bool ok = pin_part(b, p, p.plan, kRoom, 0, kRoom, false);
            check("tier.class_planned_zero_as_v1120", ok == r && b.held == a.held, "new=" + std::to_string(ok) + " old=" + std::to_string(r));
        }
        {   // every pin refused and no room: the start fails, terminates, holds nothing
            Part p({9 * kGi, 5 * kGi}, st);
            int calls = 0, unpins = 0;
            const bool ok = pin_tier_classes(p.tp, p.plan, p.stride, p.least, 0, kRoom, false,
                                             [&](size_t, int64_t) { ++calls; return false; },
                                             [&](size_t, int64_t) { ++unpins; }, [](int64_t, bool) { return int64_t{0}; });
            check("tier.pin_always_refused", !ok && calls > 0 && calls < 200 && unpins == 0 && p.tp.pin_left == 0);
            Part q({9 * kGi, 5 * kGi}, st);
            int qc = 0;
            const bool ok2 = pin_tier_classes(q.tp, q.plan, q.stride, q.least, 4 * kGi, kRoom, true,
                                              [&](size_t, int64_t) { ++qc; return false; }, [](size_t, int64_t) {},
                                              [](int64_t, bool) { return int64_t{0}; });
            check("tier.pin_always_refused_with_later", !ok2 && qc < 200);
        }
        {   // room measured 0 but the floor pins: every class gets its floor, not 0 or a negative count
            Part p({9 * kGi, 5 * kGi}, st);
            const bool ok = pin_tier_classes(p.tp, p.plan, p.stride, p.least, 0, kRoom, false,
                                             [&](size_t c, int64_t k) { return k <= p.least[c]; }, [](size_t, int64_t) {},
                                             [](int64_t, bool) { return int64_t{0}; });
            check("tier.room_zero_gives_floor", ok && p.tp.n[0] == 16 && p.tp.n[1] == 16 && p.tp.pin_left == 0);
        }
        {   // a cap under the room kept: nothing pins, no count under 0, nothing left held
            Part p({9 * kGi, 5 * kGi}, st);
            PinSim s{kRoom / 2, 0};
            const bool ok = pin_part(s, p, p.plan, kRoom, 0, kRoom, false);
            bool neg = false;
            for (const int64_t v : p.tp.n) neg = neg || v < 0;
            check("tier.cap_below_keep", !ok && !neg && s.held == 0);
        }
        {   // a negative `later` is none (clamped): same tier and same cap use as later 0
            Part p({9 * kGi, 41 * kGi}, st), q({9 * kGi, 41 * kGi}, st);
            PinSim a{44 * kGi, 0}, b{44 * kGi, 0};
            const bool oa = pin_part(a, p, p.plan, kRoom, 0, kRoom, true);
            const bool ob = pin_part(b, q, q.plan, kRoom, -5 * kGi, kRoom, true);
            check("tier.negative_later_is_zero", oa == ob && p.tp.n == q.tp.n && a.held == b.held && !q.tp.later_measured);
        }
        {   // the later parts' probe finds too little (the tier took the room): the tier pins again at its share, no leak
            Part p({10 * kGi}, {st[0]});
            PinSim s{p.planned() + kRoom, 0};
            const bool ok = pin_part(s, p, p.plan, kRoom, 5 * kGi, kRoom, true);
            Sweep sw;
            sw.expect(ok, "start failed");
            sw.expect(p.tp.repinned && p.tp.later_measured && p.tp.later_room >= 0, "not repinned/measured");
            sw.expect(p.tp.n.size() == 1 && p.tp.n[0] >= 16 && p.tp.n[0] < p.plan[0], "n=" + std::to_string(p.tp.n[0]));
            sw.expect(s.held == p.pinned(), "held " + gib(s.held) + " != pinned " + gib(p.pinned()) + " (unpin leak)");
            sw.expect(s.held + kRoom <= s.cap, "over cap");
            sw.expect(p.tp.later_scale >= 0.0 && p.tp.later_scale <= 1.0, "later_scale " + std::to_string(p.tp.later_scale));
            check("tier.later_probe_short_repins", sw.ok, sw.why());
        }
        {   // the later parts' probe finds all they need (rounded to the probe step): no repin
            Part p({10 * kGi}, {st[0]});
            PinSim s{p.planned() + kRoom + 5 * kGi, 0};
            const bool ok = pin_part(s, p, p.plan, kRoom, 5 * kGi, kRoom, true);
            check("tier.later_probe_exact_no_repin", ok && !p.tp.repinned && p.tp.later_measured && p.tp.later_scale == 1.0 &&
                                                         p.tp.n[0] == p.plan[0]);
        }
    }
    {   // A split's part keeps the floors (guarded: later parts, or a reserve over one GPU's room); one GPU does not
        const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024};
        const int64_t keep3 = 3 * kRoom;
        {   // the last part: its first class would pin whole and leave the next one less than its floor
            Part p({9 * kGi, 21 * kGi}, st);
            PinSim s{p.plan[0] * p.stride[0] + keep3 + 64 * kMi, 0};
            const bool ok = pin_part(s, p, p.plan, keep3, 0, keep3, false);
            check("guard.last_part_leaves_next_class_floor",
                  ok && p.tp.n[0] >= 16 && p.tp.n[0] < p.plan[0] && p.tp.n[1] >= 16 && s.held + keep3 <= s.cap,
                  "ok=" + std::to_string(ok) + " n0=" + std::to_string(p.tp.n.empty() ? -1 : p.tp.n[0]) + " n1=" +
                      std::to_string(p.tp.n.size() < 2 ? -1 : p.tp.n[1]));
            // ... the same cap on one GPU (the room kept 512 MiB, no later parts) is v1.1.20's: the first class pins whole
            Part q({9 * kGi, 21 * kGi}, st);
            PinSim a{s.cap, 0}, b{s.cap, 0};
            std::vector<int64_t> ref;
            const bool ok_ref = old_tier(a, q.plan, q.stride, ref);
            const bool ok_new = pin_part(b, q, q.plan, kRoom, 0, kRoom, false);
            check("guard.one_gpu_same_cap_as_v1120", ok_new == ok_ref && a.held == b.held && (!ok_new || q.tp.n == ref),
                  "new=" + std::to_string(ok_new) + " old=" + std::to_string(ok_ref));
        }
        {   // a part with a later part: its first class would pin whole and leave the later part under its floors
            Part p({9 * kGi, 21 * kGi}, st);
            const int64_t later = 13 * kGi, later_floors = 16 * (st[0] + st[1]), reserve = keep3 + 64 * kMi;
            PinSim s{p.plan[0] * p.stride[0] + reserve + later_floors / 2, 0};
            const bool ok = pin_part(s, p, p.plan, keep3, later, reserve, false);
            check("guard.part_leaves_later_floors",
                  ok && p.tp.n[0] >= 16 && p.tp.n[1] >= 16 && s.held + reserve + later_floors <= s.cap,
                  "ok=" + std::to_string(ok) + " held=" + gib(s.held) + " cap=" + gib(s.cap));
        }
        {   // a cap that cannot hold even the floors still fails the start, and holds nothing it could not keep
            Part p({9 * kGi, 21 * kGi}, st);
            PinSim s{keep3 + 16 * st[0], 0};
            const bool ok = pin_part(s, p, p.plan, keep3, 0, keep3, false);
            check("guard.cap_under_floors_fails", !ok && s.held + keep3 <= s.cap && s.held == p.pinned(),
                  "ok=" + std::to_string(ok) + " held=" + gib(s.held) + " pinned=" + gib(p.pinned()));
        }
    }
    {   // One, two and three GPUs under a pinning cap.  Spec: every part gets > 0 of every class, the total plus the room
        // kept stays under the cap, a roomy cap gives each part its whole plan, a count is never negative or over its plan
        for (int np = 1; np <= 3; ++np) {
            const std::string tag = "dual" + std::to_string(np);
            const auto make = [&]() { return np == 1 ? std::vector<Part>{two_parts()[0]} : (np == 2 ? two_parts() : three_parts()); };
            int64_t total_plan = 0, smalls = 0;
            for (const Part& p : make()) {
                total_plan += p.planned();
                smalls += p.small;
            }
            const int64_t keep = pin_room_kept(kRoom, np, true);
            Sweep inv, floors, full, tier0;
            for (int64_t cap = 0; cap <= total_plan + smalls + keep + 3 * kGi; cap += 173 * kMi + 12345) {
                std::vector<Part> parts = make();
                PinSim s{cap, 0};
                const bool ok = run_split(s, parts, true);
                const std::string at = "cap=" + gib(cap) + " ";
                if (ok) {
                    int64_t sum = 0;
                    for (size_t i = 0; i < parts.size(); ++i) {
                        const Part& p = parts[i];
                        sum += p.small + p.pinned();
                        for (size_t c = 0; c < p.tp.n.size(); ++c) {
                            const std::string pc = "part " + std::to_string(i) + " class " + std::to_string(c);
                            inv.expect(p.tp.n[c] > 0, at + pc + " got 0");
                            inv.expect(p.tp.n[c] >= 16, at + pc + " under its floor");
                            inv.expect(p.tp.n[c] <= p.plan[c], at + pc + " over its plan: " + std::to_string(p.tp.n[c]) + ">" + std::to_string(p.plan[c]));
                        }
                        tier0.expect(p.pinned() > 0, at + "part " + std::to_string(i) + " tier 0");
                    }
                    inv.expect(s.held == sum, at + "held " + gib(s.held) + " != parts' sum " + gib(sum));
                    inv.expect(s.held + keep <= cap, at + "total " + gib(s.held) + " + kept " + gib(keep) + " over the cap");
                }
                // one GPU keeps v1.1.20 (a class that pins whole before the first refusal may leave the next one nothing): not a dual spec
                if (np > 1 && cap >= floors_cap(parts, keep) + kGi) floors.expect(ok, at + "a part's start failed though all floors fit");
                if (cap >= total_plan + smalls + keep + kGi) {
                    bool whole = ok;
                    for (const Part& p : parts) whole = whole && p.pinned() == p.planned();
                    full.expect(whole, at + "roomy cap but a part is under its plan");
                }
            }
            check((tag + ".total_plus_keep_within_cap_and_counts_in_range").c_str(), inv.ok, inv.why());
            check((tag + ".every_part_gets_tier").c_str(), tier0.ok, tier0.why());
            check((tag + ".floors_fit_all_parts_start").c_str(), floors.ok, floors.why());
            check((tag + ".roomy_cap_whole_plans").c_str(), full.ok, full.why());
        }
        {   // one class per part
            const std::vector<int64_t> st1 = {6656 * 1024};
            Sweep sw;
            for (int64_t cap = 0; cap <= 50 * kGi; cap += 211 * kMi + 777) {
                std::vector<Part> parts = {Part({30 * kGi}, st1), Part({13 * kGi}, st1)};
                PinSim s{cap, 0};
                const bool ok = run_split(s, parts, true);
                const std::string at = "cap=" + gib(cap) + " ";
                if (ok) {
                    sw.expect(parts[0].tp.n[0] > 0 && parts[1].tp.n[0] > 0, at + "a part got 0");
                    sw.expect(s.held + 3 * kRoom <= cap, at + "total over the cap");
                    sw.expect(parts[0].tp.n[0] <= parts[0].plan[0] && parts[1].tp.n[0] <= parts[1].plan[0], at + "over plan");
                }
                if (cap >= 43 * kGi + 128 * kMi + 3 * kRoom + kGi)
                    sw.expect(ok && parts[0].pinned() == parts[0].planned() && parts[1].pinned() == parts[1].planned(), at + "roomy but not whole");
                if (cap >= floors_cap(parts, 3 * kRoom) + kGi) sw.expect(ok, at + "floors fit but start failed");
            }
            check("dual2.single_class_per_part", sw.ok, sw.why());
        }
    }
    {   // Spec: one GPU, and capped false, give v1.1.20's slots and pin bytes one for one (a 1/2/3-part uncapped split
        // is v1.1.20's parts pinned one after another; one GPU with the cap flag set keeps the single room)
        for (int np = 1; np <= 3; ++np) {
            Sweep un, flag;
            for (int64_t cap = 0; cap <= 90 * kGi; cap += 191 * kMi + 4321) {
                for (int mode = 0; mode < 2; ++mode) {
                    if (mode == 1 && np != 1) continue;   // the cap flag on one GPU
                    std::vector<Part> parts = np == 1 ? std::vector<Part>{two_parts()[0]} : (np == 2 ? two_parts() : three_parts());
                    std::vector<Part> mine = parts;
                    PinSim a{cap, 0}, b{cap, 0};
                    std::vector<std::vector<int64_t>> refs;
                    bool ok_ref = true;
                    for (Part& p : parts) {
                        a.held += p.small;
                        std::vector<int64_t> r;
                        if (!old_tier(a, p.plan, p.stride, r)) {
                            ok_ref = false;
                            break;
                        }
                        refs.push_back(r);
                    }
                    const bool ok = run_split(b, mine, mode == 1);
                    Sweep& sw = mode == 1 ? flag : un;
                    const std::string at = "cap=" + gib(cap) + " ";
                    sw.expect(ok == ok_ref, at + "start ok " + std::to_string(ok) + " vs v1.1.20 " + std::to_string(ok_ref));
                    sw.expect(a.held == b.held, at + "held " + gib(b.held) + " vs v1.1.20 " + gib(a.held));
                    for (size_t i = 0; i < refs.size() && ok == ok_ref; ++i) sw.expect(mine[i].tp.n == refs[i], at + "slots differ in part " + std::to_string(i));
                }
            }
            check(("parity" + std::to_string(np) + ".uncapped_as_v1120").c_str(), un.ok, un.why());
            if (np == 1) check("parity1.cap_flag_one_gpu_as_v1120", flag.ok, flag.why());
        }
    }
    {   // Warm-up out of memory: an auto-sized RAM tier shrinks and the warm-up runs again (warm_retry_slots)
        using strata::core::glmfast::kWarmRetryMax;
        using strata::core::glmfast::warm_oom_class;
        using strata::core::glmfast::warm_retry_slots;
        constexpr int kOom = 2;   // cudaErrorMemoryAllocation, hipErrorOutOfMemory
        const auto bytes = [](const std::vector<int64_t>& n, const std::vector<int64_t>& st) {
            int64_t b = 0;
            for (size_t c = 0; c < n.size(); ++c) b += n[c] * st[c];
            return b;
        };
        check("warmretry.oom_class", warm_oom_class(kOom) && !warm_oom_class(0) && !warm_oom_class(1) &&
                                         !warm_oom_class(700) && !warm_oom_class(719) && !warm_oom_class(-1));
        check("warmretry.max_is_four", kWarmRetryMax == 4);
        const std::vector<int64_t> st1 = {6656 * 1024}, least1 = {16};
        const std::vector<int64_t> n30 = {(int64_t) (29.97 * (double) kGi) / st1[0]};   // the observed CUDA0 tier
        const std::vector<int64_t> n13 = {(int64_t) (12.99 * (double) kGi) / st1[0]};   // ... and CUDA1's
        {   // no retry unless auto-sized AND an out-of-memory error AND tries left
            check("warmretry.manual_size_never_retries", warm_retry_slots(false, kOom, 0, n30, st1, least1).empty());
            check("warmretry.other_error_never_retries", warm_retry_slots(true, 1, 0, n30, st1, least1).empty() &&
                                                           warm_retry_slots(true, 0, 0, n30, st1, least1).empty() &&
                                                           warm_retry_slots(true, 700, 0, n30, st1, least1).empty());
            Sweep sw;
            for (int t = 0; t < kWarmRetryMax; ++t) sw.expect(!warm_retry_slots(true, kOom, t, n30, st1, least1).empty(), "try " + std::to_string(t));
            check("warmretry.tries_left_retry", sw.ok, sw.why());
            check("warmretry.out_of_tries_gives_up", warm_retry_slots(true, kOom, kWarmRetryMax, n30, st1, least1).empty() &&
                                                         warm_retry_slots(true, kOom, kWarmRetryMax + 7, n30, st1, least1).empty());
            check("warmretry.negative_tries_is_zero", !warm_retry_slots(true, kOom, -3, n30, st1, least1).empty());
        }
        {   // the step: 1 GiB or 5% of the tier, whichever is more
            const int64_t b30 = bytes(n30, st1), b13 = bytes(n13, st1);
            const auto r30 = warm_retry_slots(true, kOom, 0, n30, st1, least1);
            const auto r13 = warm_retry_slots(true, kOom, 0, n13, st1, least1);
            const int64_t d30 = b30 - bytes(r30, st1), d13 = b13 - bytes(r13, st1);
            check("warmretry.big_tier_loses_5pct", d30 >= b30 / 20 && d30 <= b30 / 20 + 2 * st1[0], "lost " + gib(d30) + " of " + gib(b30));
            check("warmretry.small_tier_loses_1gib", d13 >= kGi && d13 <= kGi + 2 * st1[0], "lost " + gib(d13) + " of " + gib(b13));
        }
        {   // four retries from the observed tiers: each smaller, never under a floor, ends well above the floors
            Sweep sw;
            for (const auto& start : {n30, n13}) {
                std::vector<int64_t> n = start;
                for (int t = 0; t < kWarmRetryMax; ++t) {
                    const auto r = warm_retry_slots(true, kOom, t, n, st1, least1);
                    sw.expect(!r.empty() && r.size() == n.size() && r[0] < n[0] && r[0] >= least1[0], "step " + std::to_string(t));
                    if (r.empty()) break;
                    n = r;
                }
                sw.expect(bytes(n, st1) >= bytes(start, st1) * 60 / 100, "four steps lost more than 40%");
                sw.expect(bytes(n, st1) <= bytes(start, st1) - 4 * kGi + 4 * st1[0], "four steps lost under 4 GiB");
            }
            check("warmretry.four_steps_shrink_and_stay_above_floor", sw.ok, sw.why());
        }
        {   // the floors: a tier at them cannot shrink; one just over goes to them exactly and no further
            check("warmretry.at_floor_gives_up", warm_retry_slots(true, kOom, 0, {16}, st1, least1).empty() &&
                                                     warm_retry_slots(true, kOom, 0, {3}, st1, least1).empty() &&
                                                     warm_retry_slots(true, kOom, 0, {}, {}, {}).empty());
            const auto r = warm_retry_slots(true, kOom, 0, {40}, st1, least1);
            check("warmretry.near_floor_lands_on_floor", r.size() == 1 && r[0] == 16, r.empty() ? "empty" : std::to_string(r[0]));
            const auto r2 = warm_retry_slots(true, kOom, 1, r, st1, least1);
            check("warmretry.on_floor_then_gives_up", r2.empty());
        }
        {   // several size classes: every class shrinks in proportion, none under its own floor, none grows
            const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024, 2 * kMi};
            const std::vector<int64_t> lst = {16, 16, 16};
            Sweep sw;
            for (int64_t a = 16; a <= 6000; a += 487)
                for (int64_t b = 16; b <= 4000; b += 311)
                    for (int64_t c = 16; c <= 40; c += 12) {
                        const std::vector<int64_t> n = {a, b, c};
                        for (int t = 0; t < kWarmRetryMax; ++t) {
                            const auto r = warm_retry_slots(true, kOom, t, n, st, lst);
                            const std::string at = "n=" + std::to_string(a) + "/" + std::to_string(b) + "/" + std::to_string(c) + " t=" + std::to_string(t);
                            const bool floors = a == 16 && b == 16 && c == 16;
                            if (floors) {
                                sw.expect(r.empty(), at + " shrank at the floors");
                                continue;
                            }
                            if (r.empty()) {
                                sw.expect(false, at + " gave up above the floors");
                                continue;
                            }
                            sw.expect(r.size() == 3, at + " size");
                            if (r.size() != 3) continue;
                            bool less = false;
                            for (size_t k = 0; k < 3; ++k) {
                                sw.expect(r[k] <= n[k] && r[k] >= lst[k], at + " class " + std::to_string(k) + " " + std::to_string(r[k]));
                                less = less || r[k] < n[k];
                            }
                            sw.expect(less, at + " no class shrank");
                            sw.expect(bytes(r, st) < bytes(n, st), at + " bytes did not fall");
                        }
                    }
            check("warmretry.multi_class_sweep", sw.ok, sw.why());
        }
        {   // a class already under its floor (a tiny class of one layer, cap 1) is left alone, the others still shrink
            const auto r = warm_retry_slots(true, kOom, 0, {3, 4000}, {2 * kMi, 6656 * 1024}, {16, 16});
            check("warmretry.class_under_floor_untouched", r.size() == 2 && r[0] == 3 && r[1] < 4000 && r[1] >= 16);
        }
        {   // degenerate inputs: nothing to shrink, or inputs that do not line up, give no retry
            check("warmretry.zero_slots_or_zero_stride_gives_up",
                  warm_retry_slots(true, kOom, 0, {0}, st1, least1).empty() && warm_retry_slots(true, kOom, 0, {0, 0}, {st1[0], st1[0]}, {16, 16}).empty() &&
                      warm_retry_slots(true, kOom, 0, {500}, {0}, least1).empty());
            check("warmretry.negative_slots_give_up", warm_retry_slots(true, kOom, 0, {-5}, st1, least1).empty());
            check("warmretry.size_mismatch_gives_up", warm_retry_slots(true, kOom, 0, {500, 500}, st1, {16, 16}).empty() &&
                                                          warm_retry_slots(true, kOom, 0, {500, 500}, {st1[0], st1[0]}, {16}).empty() &&
                                                          warm_retry_slots(true, kOom, 0, {500}, st1, {16, 16}).empty());
            const auto r0 = warm_retry_slots(true, kOom, 0, {500}, st1, {0});   // a zero floor is a floor: the tier may shrink, to 0 at most
            check("warmretry.zero_floor_shrinks_not_below_zero", r0.size() == 1 && r0[0] >= 0 && r0[0] < 500);
        }
        {   // every other error code gives up, whatever the tries and the size
            Sweep sw;
            for (int code = -2000; code <= 2000; ++code)
                if (code != kOom) sw.expect(warm_retry_slots(true, code, 0, n30, st1, least1).empty(), "code " + std::to_string(code));
            for (int code : {INT_MIN, INT_MAX, 1 << 16, kOom + 256, kOom << 8})
                sw.expect(warm_retry_slots(true, code, 0, n30, st1, least1).empty(), "code " + std::to_string(code));
            check("warmretry.every_other_code_gives_up", sw.ok, sw.why());
            check("warmretry.manual_size_with_any_tries_gives_up", warm_retry_slots(false, kOom, -1, n30, st1, least1).empty() &&
                                                                       warm_retry_slots(false, kOom, kWarmRetryMax - 1, n30, st1, least1).empty());
        }
        {   // the tries boundary
            check("warmretry.last_try_still_retries", !warm_retry_slots(true, kOom, kWarmRetryMax - 1, n30, st1, least1).empty());
            check("warmretry.tries_extremes", !warm_retry_slots(true, kOom, INT_MIN, n30, st1, least1).empty() &&
                                                  warm_retry_slots(true, kOom, INT_MAX, n30, st1, least1).empty());
            int ran = 0;
            std::vector<int64_t> n = n30;
            for (int t = 0; t < 100; ++t) {
                const auto r = warm_retry_slots(true, kOom, t, n, st1, least1);
                if (r.empty()) break;
                n = r;
                ++ran;
            }
            check("warmretry.big_tier_runs_exactly_max_retries", ran == kWarmRetryMax, std::to_string(ran) + " retries");
            ran = 0;
            n = {40};
            for (int t = 0; t < 100; ++t) {
                const auto r = warm_retry_slots(true, kOom, t, n, st1, least1);
                if (r.empty()) break;
                n = r;
                ++ran;
            }
            check("warmretry.near_floor_tier_stops_at_floor_before_max", ran == 1 && n[0] == 16, std::to_string(ran) + " retries, " + std::to_string(n[0]) + " slots");
        }
        {   // tiny tiers: one slot over the floor goes to the floor; a tier at or under its floor gives up
            const auto a = warm_retry_slots(true, kOom, 0, {17}, st1, least1);
            check("warmretry.one_slot_over_floor_lands_on_floor", a.size() == 1 && a[0] == 16);
            const auto b = warm_retry_slots(true, kOom, 0, {1}, st1, {1});
            const auto c = warm_retry_slots(true, kOom, 0, {1}, st1, {16});   // under its floor: left alone, nothing to give
            check("warmretry.below_floor_tier_gives_up", b.empty() && c.empty());
        }
        {   // the step equals the margin above the floors exactly
            const int64_t g = kGi;   // 1 GiB slots, so the step (1 GiB) is one slot
            const auto a = warm_retry_slots(true, kOom, 0, {5}, {g}, {4});   // margin 1 slot == step: lands on the floor
            const auto b = warm_retry_slots(true, kOom, 0, {6}, {g}, {4});   // margin 2 slots, step 1: 5
            const auto c = warm_retry_slots(true, kOom, 0, {4}, {g}, {4});   // no margin
            check("warmretry.step_equal_to_margin_lands_on_floor", a.size() == 1 && a[0] == 4);
            check("warmretry.step_half_margin_loses_one_slot", b.size() == 1 && b[0] == 5, b.empty() ? "empty" : std::to_string(b[0]));
            check("warmretry.no_margin_gives_up", c.empty());
        }
        {   // the 5% side of the step, in whole slots
            const auto r = warm_retry_slots(true, kOom, 0, {100}, {kGi}, {1});   // 100 GiB: 5 GiB
            const int64_t lost = r.empty() ? -1 : 100 - r[0];
            check("warmretry.five_percent_step_in_whole_slots", lost >= 5 && lost <= 6, "lost " + std::to_string(lost) + " slots");
            const auto r2 = warm_retry_slots(true, kOom, 0, {40}, {kGi}, {1});   // 5% = 2 GiB
            const int64_t lost2 = r2.empty() ? -1 : 40 - r2[0];
            check("warmretry.step_not_overshot_by_float_rounding", lost2 == 2, "lost " + std::to_string(lost2) + " slots (a step of 2)");
        }
        {   // rounding, single and several classes: the sum never goes over the target, every retry is strictly smaller,
            // the loss is the step (at most one slot per class beyond it), the floors hold
            Sweep sw;
            uint64_t seed = 12345;
            const auto rnd = [&](int64_t lo, int64_t hi) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                return lo + (int64_t) ((seed >> 33) % (uint64_t) (hi - lo + 1));
            };
            for (int it = 0; it < 4000; ++it) {
                const size_t C = (size_t) rnd(1, 4);
                std::vector<int64_t> n(C), st(C), ls(C);
                for (size_t c = 0; c < C; ++c) {
                    st[c] = rnd(1, 24) * kMi * 3 / 2;
                    ls[c] = rnd(0, 20);
                    n[c] = ls[c] + rnd(0, c % 2 ? 3000 : 40);
                }
                int64_t have = 0, fl = 0;
                for (size_t c = 0; c < C; ++c) {
                    have += n[c] * st[c];
                    fl += std::min(n[c], ls[c]) * st[c];
                }
                const std::string at = "it=" + std::to_string(it) + " C=" + std::to_string(C);
                const auto r = warm_retry_slots(true, kOom, 0, n, st, ls);
                if (have <= fl) {
                    sw.expect(r.empty(), at + " retried at the floors");
                    continue;
                }
                if (r.size() != C) {
                    sw.expect(false, at + " gave up above the floors");
                    continue;
                }
                const int64_t step = std::max<int64_t>(kGi, have / 20), target = std::max(fl, have - step), got = bytes(r, st);
                int64_t slack = 0;
                for (size_t c = 0; c < C; ++c) {
                    sw.expect(r[c] <= n[c] && r[c] >= std::min(n[c], ls[c]), at + " class " + std::to_string(c) + " out of range");
                    slack += st[c];
                }
                sw.expect(got < have, at + " not smaller");
                sw.expect(got <= target, at + " over the target by " + std::to_string(got - target));
                sw.expect(got >= target - slack, at + " lost far more than the step");
            }
            check("warmretry.rounding_sweep_sum_under_target_strictly_smaller", sw.ok, sw.why());
        }
        {   // one common factor: the share of its above-floor slots a class keeps agrees across classes to a slot
            Sweep sw;
            const std::vector<int64_t> st = {6656 * 1024, 7168 * 1024, 2 * kMi}, ls = {16, 16, 16};
            for (int64_t a = 100; a <= 6000; a += 997)
                for (int64_t b = 100; b <= 4000; b += 733) {
                    const std::vector<int64_t> n = {a, b, 40};
                    const auto r = warm_retry_slots(true, kOom, 0, n, st, ls);
                    if (r.size() != 3) {
                        sw.expect(false, "gave up");
                        continue;
                    }
                    const double fa = (double) (r[0] - 16) / (double) (n[0] - 16), fb = (double) (r[1] - 16) / (double) (n[1] - 16);
                    sw.expect(std::fabs(fa - fb) <= 1.0 / (double) (n[0] - 16) + 1.0 / (double) (n[1] - 16),
                              "a=" + std::to_string(a) + " b=" + std::to_string(b) + " shares " + std::to_string(fa) + " vs " + std::to_string(fb));
                }
            check("warmretry.classes_shrink_by_one_common_share", sw.ok, sw.why());
        }
        {   // a class exactly on its floor stays, the others carry the whole step
            const auto r = warm_retry_slots(true, kOom, 0, {16, 4000}, {2 * kMi, 6656 * 1024}, {16, 16});
            check("warmretry.class_on_floor_untouched_others_carry_step", r.size() == 2 && r[0] == 16 && r[1] < 4000 && r[1] >= 16);
        }
        {   // a slot bigger than the step (2 GiB slots): one slot is still given up, never past the floor
            const auto r = warm_retry_slots(true, kOom, 0, {10}, {2 * kGi}, {4});
            check("warmretry.slot_bigger_than_step_still_shrinks", r.size() == 1 && r[0] < 10 && r[0] >= 4, r.empty() ? "empty" : std::to_string(r[0]));
            const auto q = warm_retry_slots(true, kOom, 0, {5}, {2 * kGi}, {4});
            check("warmretry.slot_bigger_than_step_one_over_floor", q.size() == 1 && q[0] == 4);
        }
        {   // a very big tier (32 TiB) and one whose byte count passes int64
            const int64_t nb = int64_t{5} << 20, sb = int64_t{6656} * 1024;
            const auto big = warm_retry_slots(true, kOom, 0, {nb}, {sb}, {16});
            check("warmretry.huge_tier_loses_5pct", big.size() == 1 && big[0] < nb && (nb - big[0]) * sb >= nb * sb / 20 - sb);
            const int64_t nn = int64_t{1} << 50;   // 2^50 slots of 1 MiB: 2^70 bytes
            const auto ov = warm_retry_slots(true, kOom, 0, {nn}, {kMi}, {16});
            check("warmretry.byte_count_overflow_is_no_silent_wrong_answer", ov.empty() || (ov.size() == 1 && ov[0] >= 16 && ov[0] < nn),
                  ov.empty() ? "empty" : std::to_string(ov[0]));
        }
        {   // with scaled_slots (a split's scale, never under a floor): retries from its result keep the floor too
            Sweep sw;
            for (double scale : {0.01, 0.1, 0.5, 0.97, 1.0, 1.5})
                for (int64_t plan : {10, 17, 100, 4000}) {
                    std::vector<int64_t> n = {strata::core::glmfast::scaled_slots(plan, 16, scale)};
                    for (int t = 0; t < kWarmRetryMax; ++t) {
                        const auto r = warm_retry_slots(true, kOom, t, n, st1, least1);
                        if (r.empty()) break;
                        sw.expect(r[0] < n[0] && r[0] >= std::min<int64_t>(n[0], 16), "scale " + std::to_string(scale) + " plan " + std::to_string(plan) + " t " + std::to_string(t));
                        n = r;
                    }
                }
            check("warmretry.after_scaled_slots_keeps_floor", sw.ok, sw.why());
        }
    }
    std::printf("dual-pin checks: %d run, %d expected failures, %d bad\n", g_checks, g_xfail, g_bad);
    if (g_bad) return 1;
    std::puts("GLM memory policy checks passed (CPU only)");
}
