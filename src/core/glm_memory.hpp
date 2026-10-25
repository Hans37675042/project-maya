#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::glmfast {

// The complete pool and the all-resident launch skip (STRATA_GLM_COMPLETE_POOL, STRATA_GLM_RESIDENT_SKIP): on by
// default on AMD, where the Strix / Gorgon Halo APUs hold every expert (measured there); an NVIDIA build keeps the
// capped pool and every route's launches unless the setting is 1.  The setting, when given, wins either way.
inline bool on_by_default_on_amd(const char* env) {
    if (const char* v = std::getenv(env)) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
    return true;
#else
    return false;
#endif
}
// ... and the other way round: on by default on NVIDIA, off on AMD unless the setting is 1
inline bool off_by_default_on_amd(const char* env) {
    if (const char* v = std::getenv(env)) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
    return false;
#else
    return true;
#endif
}
// STRATA_GLM_GROUP_EXPERTS=1: the batched rows' routed experts each run once for all the rows of a group (glm_rows.cu)
inline bool group_experts_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GLM_GROUP_EXPERTS");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
// STRATA_GLM_SKIP_EMPTY=1: the batched rows leave out the launches their routes have no work for (glm_rows.cu)
inline bool skip_empty_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GLM_SKIP_EMPTY");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

// All inputs are bytes. On an APU the GPU pool and pinned host allocations spend
// the same physical memory; the runtime's GTT/carve-out size is not free RAM.
inline size_t expert_pool_budget(bool unified, size_t device_free, size_t ram_available,
                                size_t reserve, size_t ram_headroom, size_t expert_bytes) {
    if (!unified) return device_free > reserve ? device_free - reserve : 0;
    const size_t room = ram_available > ram_headroom ? ram_available - ram_headroom : 0;
    return std::min(expert_bytes, room > reserve ? room - reserve : 0);
}

// The VRAM kept free beside a discrete card's expert pool: the reserve, except on a small card where the default
// one would leave the pool short of its floor (a route's experts and the spares in every layer) - there an automatic
// budget lowers it to what leaves exactly the floor, down to `small`.  A reserve set by hand, a fixed budget
// (--expert-cache N) and a card short even at `small` keep it.  {reserve, lowered}
inline std::pair<size_t, bool> pool_reserve(size_t device_free, size_t reserve, bool given, bool fixed, size_t floor,
                                            size_t small) {
    if (given || fixed || device_free >= reserve + floor || device_free < floor + small) return {reserve, false};
    return {device_free - floor, true};
}

// The check of a written budget: the bytes it gives back once `have` of the `asked` bytes were allocated and touched
// and `free_after` was read - what the driver refused, plus what the reserve is short by and 64 MiB.  A read of
// nearly nothing free (under 16 MiB) only says "at least": the first two give back 1 GiB each, later ones a quarter.
inline size_t pool_give_back(size_t asked, size_t have, size_t free_after, size_t reserve, int& zero_reads) {
    constexpr size_t kSlack = size_t{64} << 20, kGiB = size_t{1} << 30;
    size_t give = asked > have ? asked - have : 0;
    if (free_after + kSlack < reserve) {
        size_t g = reserve - free_after + kSlack;
        if (free_after < (size_t{16} << 20)) g = std::max(g, ++zero_reads <= 2 ? kGiB : asked / 4);
        give += g;
    }
    return std::min(give, asked);
}

inline bool minimal_ram_tier(bool unified, bool all_experts_fit, bool explicit_budget) {
    return unified && all_experts_fit && !explicit_budget;
}

inline bool full_unified_pool(bool unified, int slots, int experts) {
    return unified && slots >= experts;
}

// A COMPLETE pool: a layer's main slots (the lendable tail aside) hold every expert AND the spares - the Windows APU's
// carve-out, a big card, a split over many GPUs.  Nothing is ever evicted there (a spare always finds a free slot), the
// prompt path borrows only the tail beyond them, and the decode never misses.
inline bool complete_pool(int main_slots, int experts, int spares) {
    return main_slots >= experts + spares;
}

// The slots a layer gets where the budget (avail bytes; stride_sum: one slot in every MoE layer) holds a complete pool
// plus the lendable tail - every expert, the spares and `tail` slots - else 0 (the pool is sized from the budget)
inline int complete_pool_slots(size_t avail, size_t stride_sum, int experts, int spares, int tail) {
    const size_t per = (size_t) std::max(0, experts + spares + std::max(0, tail));
    return stride_sum > 0 && experts > 0 && avail / stride_sum >= per ? (int) per : 0;
}

// Windows' commit on an APU: Windows charges every allocation of an integrated GPU to its commit (RAM + page file),
// though it does not count the carve-out as RAM (DeepSeek's engine measured it on a Radeon 8065S: src/ds4/ds4_model.cu,
// commit_room).  The other programs keep max(8 GiB, 10% of the RAM) of it, as there.
inline size_t commit_margin(size_t ram_total) {
    return std::max(size_t{8} << 30, ram_total / 10);
}
// The expert pool's budget within what Windows can still commit (commit_avail) beside that margin - never below its
// floor (a route's experts and the spares in every layer: the least a decode runs with), nor above `budget`.  `all`:
// the most the pool can ever take (0: no such bound).  {the budget, the bytes of experts that keeps off the GPU}
inline std::pair<size_t, size_t> commit_pool_cap(size_t budget, size_t floor, size_t all, size_t commit_avail,
                                                 size_t ram_total) {
    const size_t margin = commit_margin(ram_total);
    const size_t room = commit_avail > margin ? commit_avail - margin : 0;
    const size_t cap = std::min(budget, std::max(room, floor));
    const size_t top = all > 0 ? all : budget;
    return {cap, std::min(budget, top) - std::min(cap, top)};
}
// The page file (GiB, rounded up) that holds `kept` bytes more: today's - the commit limit less the RAM - and those
inline size_t pagefile_hint_gib(size_t commit_limit, size_t ram_total, size_t kept) {
    const size_t pf = commit_limit > ram_total ? commit_limit - ram_total : 0;
    return (pf + kept + (size_t{1} << 30) - 1) >> 30;
}

// The slots the warm-up fills: every expert where a unified pool holds them all, else all but the spares - and never
// more than there are experts (a complete pool has more slots than experts: the warm-up's order lists each once)
inline int warm_pool_slots(bool unified, int slots, int experts, int spares) {
    return full_unified_pool(unified, slots, experts) ? experts : std::min(experts, std::max(0, slots - spares));
}

// A RAM tier's size classes pinned one after another under a pinning cap (Windows' driver pins less than the free RAM;
// STRATA_GLM_PIN_CAP_MB simulates one).  The caller's three calls:
//   pin(c, n)            pins n slots of class c and keeps them only while the room kept above the tier still pins
//   unpin(c, n)          frees what pin(c, n) kept
//   room(limit, later)   how much more pins now, up to `limit` bytes (probes, freed again); later: the probe of a
//                        split's later parts, not one after a refusal
// planned/stride/least: a class's planned slots, its slot bytes and the floor it never goes under.  later: the tier
// bytes a split's later parts plan to pin after this one (0 on one GPU); reserve: the bytes that must stay pinnable
// beside all the tiers (the room kept above them, and the later parts' small pinned buffers).
// At the first refusal the room is measured and each class takes its planned share of it - its share beside the later
// parts' tiers too: the part that meets the cap first then leaves the later ones their share, not what happens to be
// left (a split's first card on 64 GB of Windows pinned its whole 30.8 GB plan and its second card found 0.00 GB).
// measure_later (once a split, on its first part): when no refusal measured the room, it is measured after the tier
// for the later parts, and a tier that leaves them short is pinned again at its share.  Each class of one GPU (later
// 0, the reserve the room kept above it) keeps exactly the steps it always had.
// A split's part (later > 0, or a reserve over one GPU's room: "guarded") also keeps the floors: a class that pins
// before any refusal must leave the floors of the classes after it and of the later parts (and the reserve) still
// pinnable, or it is unpinned and takes its share of the measured room; and a share is the floor plus its part of
// what is left beyond all the floors, so a cap that holds every floor gives every class of every part its floor.
// (The later parts' floors are estimated from this part's: its floors for each multiple of its plan they plan.)
// guard: guarded whatever the later parts and the reserve - a warm-up retry's re-pin (fast_setup's retier), which pins
// near a limit just met and must leave each class after the one pinning its floor, on one GPU too.
struct TierPin {
    std::vector<int64_t> n;       // the slots each class got (n[c] 0: class c did not pin, the start fails)
    int64_t pin_left = -1;        // the room measured under the cap less the classes pinned since (-1: none measured)
    int64_t later_room = -1;      // what the later parts' probe found (-1: no probe)
    bool later_measured = false;  // later_scale comes from a measurement
    bool repinned = false;        // the tier was pinned again at its share
    double later_scale = 1.0;     // the share of their plans the later parts have room for
};
constexpr int64_t kPinProbeStep = int64_t{256} << 20;   // pin_room's smallest probe
constexpr int64_t kPinRoomOneGpu = int64_t{512} << 20;   // kPinRoom (glm_fast_path.cu): the room one GPU keeps above its tier
template <class Pin, class Unpin, class Room>
inline bool pin_tier_classes(TierPin& out, const std::vector<int64_t>& planned, const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& least, int64_t later, int64_t reserve, bool measure_later,
                             Pin pin, Unpin unpin, Room room, bool guard = false) {
    const size_t C = planned.size();
    later = std::max<int64_t>(0, later);
    const auto rest_of = [&](size_t c) {   // the planned bytes from class c on, the later parts' with them
        int64_t rest = later;
        for (size_t k = c; k < C; ++k) rest += planned[k] * stride[k];
        return rest;
    };
    // a split's part (see above); one GPU, and a split without a cap, are not: their steps are v1.1.20's
    const bool guarded = guard || later > 0 || reserve > kPinRoomOneGpu;
    int64_t own_least = 0, own_plan = 0;   // this part's floors and plan, bytes
    for (size_t k = 0; k < C; ++k) {
        own_least += least[k] * stride[k];
        own_plan += planned[k] * stride[k];
    }
    const int64_t later_least =
        later > 0 ? std::min(later, own_least * ((later + std::max<int64_t>(1, own_plan) - 1) / std::max<int64_t>(1, own_plan))) : 0;
    const auto least_from = [&](size_t c) {   // the floors, in bytes, of the classes from class c on
        int64_t b = 0;
        for (size_t k = c; k < C; ++k) b += least[k] * stride[k];
        return b;
    };
    const auto share = [&](size_t c) {
        if (guarded) {   // the floor, and its part of the room beyond every floor (the later parts' with them)
            int64_t extra = later - later_least;
            for (size_t k = c; k < C; ++k) extra += std::max<int64_t>(0, planned[k] - least[k]) * stride[k];
            const int64_t spare = out.pin_left - reserve - least_from(c) - later_least;
            const double f = spare > 0 ? std::min(1.0, (double) spare / (double) std::max<int64_t>(1, extra)) : 0.0;
            return least[c] + (int64_t) (f * (double) std::max<int64_t>(0, planned[c] - least[c]));
        }
        const double f = (double) std::max<int64_t>(0, out.pin_left - reserve) / (double) std::max<int64_t>(1, rest_of(c));
        return (int64_t) (f * (double) planned[c]);
    };
    const auto run = [&]() {
        out.n.assign(C, 0);
        for (size_t c = 0; c < C; ++c) {
            int64_t n = planned[c];
            const int64_t lo = least[c];
            if (out.pin_left >= 0) n = std::max(lo, std::min(n, share(c)));
            bool ok = false;
            // a refusal near a pinning cap: the size measured, and should that be refused too, 1/32 less a try for
            // the first four (a tier just over a cap loses ~3%, not the 12% of a 7/8 step), then 1/8 less
            for (int tries = 0; n >= lo; ++tries) {
                bool pinned = pin(c, n), held_back = false;
                if (pinned && guarded && out.pin_left < 0) {
                    // before any refusal: what is left must still pin the floors after this class, the reserve, and
                    // (but the first part, whose probe after the tier measures the later parts) the later parts' floors
                    const int64_t need = least_from(c + 1) + (measure_later ? 0 : later_least) + reserve;
                    if (room((need + kPinProbeStep - 1) / kPinProbeStep * kPinProbeStep, true) < need) {
                        unpin(c, n);
                        pinned = false;
                        held_back = true;
                    }
                }
                if (pinned) {
                    ok = true;
                    break;
                }
                if (out.pin_left < 0) {
                    out.pin_left = room(rest_of(c) + reserve, false);
                    const int64_t fit = share(c);
                    if (fit < n) {
                        n = std::max(lo, fit);
                        continue;
                    }
                    if (held_back) continue;   // (the room is measured now: the same size pins in its share)
                }
                const int64_t less = tries < 4 ? n * 31 / 32 : n * 7 / 8;
                n = (n > lo && less < lo) ? lo : less;
            }
            if (!ok) return false;
            out.n[c] = n;
            if (out.pin_left >= 0) out.pin_left = std::max<int64_t>(0, out.pin_left - n * stride[c]);
        }
        return true;
    };
    if (!run()) return false;
    if (later <= 0) return true;
    if (out.pin_left < 0 && measure_later) {
        // (rounded up to the probes' step: room that pins all of it then reads as all of it)
        const int64_t need = (later + reserve + kPinProbeStep - 1) / kPinProbeStep * kPinProbeStep;
        const int64_t r = room(need, true);
        out.later_room = r;
        out.later_measured = true;
        if (r >= need) return true;
        // this tier leaves the later parts short: it goes, and pins again at its share of the room both have
        int64_t own = 0;
        for (size_t c = 0; c < C; ++c) own += out.n[c] * stride[c];
        for (size_t c = 0; c < C; ++c) unpin(c, out.n[c]);
        out.pin_left = own + r;
        out.repinned = true;
        if (!run()) return false;
    }
    if (guarded && later > 0 && !out.repinned && out.pin_left >= 0 && out.pin_left - reserve < later_least) {
        // a refusal measured the room, but this tier left the later parts under their floors: pin again at its share
        int64_t own = 0;
        for (size_t c = 0; c < C; ++c) own += out.n[c] * stride[c];
        for (size_t c = 0; c < C; ++c) unpin(c, out.n[c]);
        out.pin_left += own;
        out.repinned = true;
        if (!run()) return false;
    }
    if (out.pin_left >= 0) {
        out.later_scale = std::min(1.0, std::max(0.0, (double) (out.pin_left - reserve) / (double) later));
        out.later_measured = true;
    }
    return true;
}

// The room kept pinnable above the RAM tiers (`room` a GPU): one GPU's, or under a pinning cap one for each GPU of a
// split and one more for their warm-ups, which run at once
inline int64_t pin_room_kept(int64_t room, int n_parts, bool capped) {
    return room * (capped && n_parts > 1 ? n_parts + 1 : 1);
}
// What a split's part `part` pins beside: its later parts' tiers - their plans (bytes) at the split's scale - and the
// bytes that must stay pinnable beside every tier: the room kept, and the later parts' small pinned buffers.
// {later, reserve}
inline std::pair<int64_t, int64_t> split_later(const std::vector<int64_t>& plan, const std::vector<int64_t>& small,
                                               int part, double scale, int64_t keep) {
    int64_t later = 0, reserve = keep;
    for (size_t j = (size_t) std::max(0, part + 1); j < plan.size() && j < small.size(); ++j) {
        later += plan[j];
        reserve += small[j];
    }
    return {(int64_t) ((double) later * scale), reserve};
}
// a class's planned slots at the split's scale, never under its floor
inline int64_t scaled_slots(int64_t n, int64_t least, double scale) {
    return scale < 1.0 ? std::max(least, (int64_t) ((double) n * scale)) : n;
}

// A warm-up that runs out of memory (a copy or the tables' upload answers "out of memory" when the RAM tier is pinned
// too near what the driver and Windows can still lock for their own buffers): a RAM tier SIZED BY THE ENGINE gives
// some of its slots back and the warm-up runs again - at most kWarmRetryMax times, each a step smaller.  A size the user
// set (STRATA_GLM_RAM_GB) is theirs: that start fails with the error, as it always did.
constexpr int kWarmRetryMax = 4;
constexpr int kCudaOutOfMemory = 2;   // cudaErrorMemoryAllocation, and hipErrorOutOfMemory
inline bool warm_oom_class(int code) {
    return code == kCudaOutOfMemory;
}
// The slots each class of the RAM tier keeps for the next try, after `tries` retries already ran (n, stride and least:
// the classes' pinned slots, slot bytes and floors) - empty: no retry (a manual size, another error, out of tries, or
// the tier is at its floors).  The step is 1 GiB or 5% of the tier, whichever is more; what a class holds above its
// floor shrinks by one common factor, so the floors are reached together and never passed.  Every retry is smaller.
inline std::vector<int64_t> warm_retry_slots(bool auto_sized, int code, int tries, const std::vector<int64_t>& n,
                                             const std::vector<int64_t>& stride, const std::vector<int64_t>& least) {
    const size_t C = n.size();
    if (!auto_sized || !warm_oom_class(code) || tries >= kWarmRetryMax || stride.size() != C || least.size() != C)
        return {};
    int64_t have = 0, floor_b = 0;
    std::vector<int64_t> keep(C);   // a class's floor (or all of it, where it holds less)
    for (size_t c = 0; c < C; ++c) {
        keep[c] = std::max<int64_t>(0, std::min(n[c], least[c]));   // (a negative floor is 0: no class goes under 0)
        have += n[c] * stride[c];
        floor_b += keep[c] * stride[c];
    }
    if (have <= floor_b) return {};
    const int64_t step = std::max<int64_t>(int64_t{1} << 30, have / 20);
    const int64_t target = std::max(floor_b, have - step);
    const double f = (double) (target - floor_b) / (double) (have - floor_b);
    std::vector<int64_t> out(C);
    bool less = false;
    for (size_t c = 0; c < C; ++c) {
        out[c] = keep[c] + (int64_t) ((double) (n[c] - keep[c]) * f);
        less = less || out[c] < n[c];
    }
    if (!less) {   // (a step under one slot of any class: the biggest class above its floor gives one up)
        size_t b = C;
        for (size_t c = 0; c < C; ++c)
            if (out[c] > keep[c] && (b == C || (out[c] - keep[c]) * stride[c] > (out[b] - keep[b]) * stride[b])) b = c;
        if (b == C) return {};
        --out[b];
    }
    return out;
}

// A split's warm-ups and their retries' re-pins (fast_setup).  The parts warm up at once, but a part whose warm-up ran
// out of memory pins its smaller tier again only while no other part warms up or re-pins: the room under a pinning
// cap read while another card streamed its experts in came to 0.00 GB (why every part waits at wait_pin before its
// first warm-up), and a re-pin near the cap can take the pinned buffers another card's copy needs.  A warm-up is a
// shared turn, a re-pin an exclusive one:
//   - a part asks for a re-pin only once its own warm-up has ended (end_warm), and while it waits it holds no turn:
//     two parts retrying at once take turns, neither waits on the other's warm-up or re-pin
//   - a re-pin asked for goes before every warm-up not yet begun (the ones running end; none starts past it), so the
//     re-pin is not held off by the other parts' next warm-ups
//   - one GPU (no split) never takes the gate: its path has no wait at all
//   - an aborted split (a part failed, SplitSync::fail): every waiter wakes without a turn, and no turn is given after -
//     a begin_ returns false then.  A turn already held runs on and is given back as before
// The start's line when it failed for want of pinned memory - a RAM tier's pin refused, or a warm-up out of memory
// (warm_oom_class) - with the RAM tiers this start set out to pin (all GPUs, GB of 2^30 bytes) and whether their size
// was set by hand (STRATA_GLM_RAM_GB, the API's budget) or is not the auto size (staging only, RAM-resident).  Within one
// process the tier cannot pin again what it gave back (Windows: ~32 GB unpinned, then not even 256 MB pinned), so
// serve/server.py starts the engine again with a smaller STRATA_GLM_RAM_GB from it (ram_retry_gb) - manual=0 only
inline std::string ram_tier_oom_line(int64_t total_bytes, bool manual) {
    char b[96];
    std::snprintf(b, sizeof b, "glm fast: RAM-TIER-OOM total_gb=%.2f manual=%d",
                  (double) std::max<int64_t>(0, total_bytes) / 1073741824.0, manual ? 1 : 0);
    return b;
}

class WarmGate {
public:
    bool begin_warm() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return aborted_ || (!repinning_ && repin_asked_ == 0); });
        if (aborted_) return false;
        ++warming_;
        return true;
    }
    void end_warm() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            --warming_;
        }
        cv_.notify_all();
    }
    bool begin_repin() {
        std::unique_lock<std::mutex> lk(mu_);
        ++repin_asked_;
        cv_.wait(lk, [&] { return aborted_ || (!repinning_ && warming_ == 0); });
        --repin_asked_;
        if (aborted_) return false;
        repinning_ = true;
        return true;
    }
    void end_repin() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            repinning_ = false;
        }
        cv_.notify_all();
    }
    void abort() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            aborted_ = true;
        }
        cv_.notify_all();
    }
    bool aborted() {
        std::lock_guard<std::mutex> lk(mu_);
        return aborted_;
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    int warming_ = 0;       // the parts warming up now
    int repin_asked_ = 0;   // the parts waiting for their re-pin's turn
    bool repinning_ = false;
    bool aborted_ = false;
};
// a turn of a WarmGate, given back however its scope ends (gate nullptr: none - one GPU).  ok(): the turn was taken
// (false: the split was aborted - nothing to give back)
class WarmTurn {
public:
    WarmTurn(WarmGate* gate, bool repin) : gate_(gate), repin_(repin) {
        if (gate_ != nullptr) held_ = repin_ ? gate_->begin_repin() : gate_->begin_warm();
    }
    ~WarmTurn() {
        if (gate_ != nullptr && held_) repin_ ? gate_->end_repin() : gate_->end_warm();
    }
    bool ok() const { return gate_ == nullptr || held_; }
    WarmTurn(const WarmTurn&) = delete;
    WarmTurn& operator=(const WarmTurn&) = delete;

private:
    WarmGate* gate_;
    bool repin_;
    bool held_ = false;
};

// A parallel split load's turns (glm_model.cu's load_pack_split, fast_setup): the parts take the RAM budget in part
// order (wait / pass), then pin their RAM tiers in part order too (wait_pin / pass_pin), and warm up under `gate`.
// A part that fails (fail) aborts the split: every wait here and at the gate wakes and returns false, and the parts
// still loading stop instead of warming up or retrying - a 2-GPU start whose second card's pin was refused had the
// first card warm up, run out of memory and report that instead, and CUDA1's "did not allocate" went unseen.  A part
// that stops because of the abort says so (stopped): failures() keeps the parts that failed of their own, the earliest
// first.  Every part's thread ends with finish(), which passes its turns however it ended (a part that never reached
// them lets the others on)
class SplitSync {
public:
    explicit SplitSync(int n) : passed_((size_t) std::max(0, n), false), pinned_((size_t) std::max(0, n), false),
                                stopped_((size_t) std::max(0, n), false) {}
    int parts() const { return (int) passed_.size(); }
    // wait for the turn of `part` (parts: all have passed); false: the split was aborted
    bool wait(int part) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return aborted_ || turn_ >= part; });
        return !aborted_;
    }
    void pass(int part) { mark(passed_, turn_, part); }
    bool wait_pin(int part) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return aborted_ || pin_turn_ >= part; });
        return !aborted_;
    }
    void pass_pin(int part) { mark(pinned_, pin_turn_, part); }
    // `part` failed: the split is aborted (the gate first: a part that reads aborted() true finds no turn there)
    void fail(int part) {
        gate.abort();
        {
            std::lock_guard<std::mutex> lk(mu_);
            const bool own = part >= 0 && part < parts() && !stopped_[(size_t) part];
            if (own && std::find(failed_.begin(), failed_.end(), part) == failed_.end()) failed_.push_back(part);
            aborted_ = true;
            aborted_flag_.store(true);
        }
        cv_.notify_all();
    }
    // `part` stops because the split was aborted: its end (finish(part, false)) is no failure of its own
    void stopped(int part) {
        std::lock_guard<std::mutex> lk(mu_);
        if (part >= 0 && part < parts()) stopped_[(size_t) part] = true;
    }
    // a part's thread ends: a failure aborts, and its turns pass whatever it reached
    void finish(int part, bool ok) {
        if (!ok) fail(part);
        pass(part);
        pass_pin(part);
    }
    bool aborted() const { return aborted_flag_.load(); }
    std::vector<int> failures() {
        std::lock_guard<std::mutex> lk(mu_);
        return failed_;
    }
    WarmGate gate;   // the warm-ups and their retries' re-pins

private:
    void mark(std::vector<bool>& done, int& turn, int part) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (part >= 0 && part < parts()) done[(size_t) part] = true;
            while (turn < parts() && done[(size_t) turn]) ++turn;
        }
        cv_.notify_all();
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<bool> passed_, pinned_, stopped_;
    std::vector<int> failed_;   // the parts that failed of their own, in order
    int turn_ = 0, pin_turn_ = 0;
    bool aborted_ = false;
    std::atomic<bool> aborted_flag_{false};   // (aborted_, read without the lock: a warm-up's poll)
};

}  // namespace strata::core::glmfast
