// CPU-only policy checks: no HIP/CUDA headers, device queries or allocations.
#include "../src/core/glm_memory.hpp"

#undef NDEBUG   // the checks are asserts: keep them in Release builds
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
void pin_split(PinSim& s, std::vector<Part>& parts, bool capped) {
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
        assert(pin_part(s, p, n, keep, lr.first, lr.second, capped && !measured && i + 1 < np));
        if (lr.first > 0 && p.tp.later_measured) {
            scale *= p.tp.later_scale;
            measured = true;
        }
    }
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
    std::puts("GLM memory policy checks passed (CPU only)");
}
