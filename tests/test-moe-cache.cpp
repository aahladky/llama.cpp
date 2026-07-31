// Deterministic unit tests for the SYCL MoE expert cache.
//
// These run the real moe_expert_cache in host-only mode: the slot pool is
// ordinary memory and promotions are plain memcpy, so admission, eviction,
// phase and geometry logic are exercised with no GPU and no model. That is
// deliberate -- these tests cover policy bookkeeping, and a test that
// needs a 16 GiB MoE and two Arc cards to run is a test that does not
// run.
//
// Cases requiring real devices (two contexts on one GPU, multi-GPU,
// dynamic backend loading) are not covered here; they need the end-to-end
// fixtures under scripts/moe-cache-correctness/.

#include "ggml-sycl/moe-cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// The policy lives in moe-cache.cpp and links here; the device operations
// live in moe-cache-device.cpp, which needs SYCL and is deliberately not
// linked. Host-only mode never reaches them, so these stubs exist only to
// satisfy the linker -- and abort rather than no-op, so a test that strays
// onto the device path fails loudly instead of quietly passing.
static void device_path_unreachable(const char * what) {
    fprintf(stderr, "moe_cache: host-only test reached the device path (%s)\n", what);
    abort();
}

// GCC notices these never return and suggests marking them noreturn, which
// would contradict the return types the header declares. The suggestion is
// correct and inapplicable, so it is silenced here rather than repo-wide.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif
void * moe_cache_device_alloc(size_t, void *) { device_path_unreachable("alloc"); return nullptr; }
void   moe_cache_device_free(void *, void *) { device_path_unreachable("free"); }
bool   moe_cache_device_copy(void *, const void *, size_t, void *) { device_path_unreachable("copy"); return false; }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

static int g_failures = 0;
static std::string g_case;

#define CASE(name) do { g_case = (name); } while (0)

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL [%s] %s:%d: %s\n",                            \
                g_case.c_str(), __FILE__, __LINE__, #cond);                 \
        g_failures++;                                                       \
    }                                                                       \
} while (0)

#define CHECK_EQ(a, b) do {                                                 \
    const auto _a = (a); const auto _b = (b);                               \
    if (!(_a == _b)) {                                                      \
        fprintf(stderr, "FAIL [%s] %s:%d: %s == %s (%lld vs %lld)\n",       \
                g_case.c_str(), __FILE__, __LINE__, #a, #b,                 \
                (long long)_a, (long long)_b);                              \
        g_failures++;                                                       \
    }                                                                       \
} while (0)

// Projection ids, matching moe_cache_parse_projection.
enum { GATE = 0, UP = 1, DOWN = 2 };

static const size_t PROJ_BYTES = 256;

static moe_cache_config make_config(int n_slots_wanted, int admission = 1,
                                    const char * policy = "lru",
                                    bool prefill_admit = false) {
    moe_cache_config cfg;
    cfg.device_id     = 0;
    cfg.n_layers      = 4;
    cfg.n_experts     = 8;
    cfg.gate_bytes    = PROJ_BYTES;
    cfg.up_bytes      = PROJ_BYTES;
    cfg.down_bytes    = PROJ_BYTES;
    cfg.expert_bytes  = PROJ_BYTES * 3;
    cfg.budget_bytes  = PROJ_BYTES * 3 * (size_t)n_slots_wanted;
    cfg.policy        = policy;
    cfg.admission_misses = admission;
    cfg.prefill_admit = prefill_admit;
    cfg.host_only_for_testing = true;
    return cfg;
}

static std::vector<uint8_t> weights(uint8_t fill) {
    return std::vector<uint8_t>(PROJ_BYTES, fill);
}

// One full use of an expert: every projection is looked up, and each miss
// is recorded and offered for promotion, exactly as the scheduler hook does.
static void use_expert(moe_expert_cache & cache, int layer, int expert,
                       uint8_t fill = 0xAB) {
    const auto w = weights(fill);
    for (int proj = GATE; proj <= DOWN; proj++) {
        if (cache.lookup(layer, expert, proj, PROJ_BYTES)) {
            continue;
        }
        cache.record_miss(layer, expert, proj);
        cache.promote_projection(layer, expert, proj, w.data(), PROJ_BYTES);
    }
}

// ---------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------

static void test_admission_threshold_one() {
    CASE("admission threshold 1 admits on first use");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));

    use_expert(cache, 0, 0);
    // Every projection is resident after one use.
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(0, 0, UP,   PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(0, 0, DOWN, PROJ_BYTES) != nullptr);
    CHECK_EQ(cache.slots_used(), 1);
}

static void test_admission_threshold_two_needs_a_second_use() {
    // The bug this exists for: miss counts were shared across gate/up/down,
    // so one use produced three misses and a threshold of 2 was reached
    // within that single use -- "admit on the second use" behaved like
    // "admit on the first".
    CASE("admission threshold 2 requires a second use, not a second projection");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/2), nullptr));

    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) == nullptr);
    CHECK(cache.lookup(0, 0, UP,   PROJ_BYTES) == nullptr);
    CHECK(cache.lookup(0, 0, DOWN, PROJ_BYTES) == nullptr);
    CHECK_EQ(cache.slots_used(), 0);

    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(0, 0, UP,   PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(0, 0, DOWN, PROJ_BYTES) != nullptr);
    CHECK_EQ(cache.slots_used(), 1);
}

static void test_admission_threshold_three() {
    CASE("admission threshold 3 requires a third use");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/3), nullptr));

    use_expert(cache, 0, 0);
    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) == nullptr);
    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
}

static void test_cached_projection_does_not_reset_others() {
    // A hit used to clear the whole expert's shared miss count, so an
    // already-cached gate wiped the admission progress the still-uncached
    // up and down projections had accumulated.
    CASE("a cached projection does not reset another projection's admission");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/2), nullptr));

    const auto w = weights(0x11);

    // Drive gate to residency on its own.
    cache.record_miss(0, 0, GATE);
    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, w.data(), PROJ_BYTES) != nullptr);

    // Up accumulates one miss, then gate is hit repeatedly.
    cache.record_miss(0, 0, UP);
    for (int i = 0; i < 5; i++) {
        CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
    }
    // Up's single remaining miss must still count: one more reaches 2.
    cache.record_miss(0, 0, UP);
    CHECK(cache.promote_projection(0, 0, UP, w.data(), PROJ_BYTES) != nullptr);
}

static void test_all_projections_eventually_populate() {
    CASE("all cacheable projections eventually populate");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/2), nullptr));
    for (int i = 0; i < 4; i++) {
        use_expert(cache, 1, 3);
    }
    CHECK(cache.lookup(1, 3, GATE, PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(1, 3, UP,   PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(1, 3, DOWN, PROJ_BYTES) != nullptr);
}

// ---------------------------------------------------------------------
// Contents, partial fill, geometry
// ---------------------------------------------------------------------

static void test_cached_bytes_are_the_promoted_bytes() {
    CASE("a hit returns the bytes that were promoted");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));

    const auto w = weights(0x5A);
    cache.record_miss(0, 2, GATE);
    void * slot = cache.promote_projection(0, 2, GATE, w.data(), PROJ_BYTES);
    CHECK(slot != nullptr);

    void * hit = cache.lookup(0, 2, GATE, PROJ_BYTES);
    CHECK_EQ(hit, slot);
    CHECK_EQ(memcmp(hit, w.data(), PROJ_BYTES), 0);
}

static void test_projections_occupy_distinct_regions() {
    CASE("gate/up/down occupy distinct regions of one slot");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));

    const auto g = weights(0x01);
    const auto u = weights(0x02);
    const auto d = weights(0x03);
    cache.record_miss(0, 0, GATE);
    void * pg = cache.promote_projection(0, 0, GATE, g.data(), PROJ_BYTES);
    cache.record_miss(0, 0, UP);
    void * pu = cache.promote_projection(0, 0, UP, u.data(), PROJ_BYTES);
    cache.record_miss(0, 0, DOWN);
    void * pd = cache.promote_projection(0, 0, DOWN, d.data(), PROJ_BYTES);

    CHECK(pg && pu && pd);
    CHECK(pg != pu && pu != pd && pg != pd);
    CHECK_EQ(*(uint8_t *)pg, 0x01);
    CHECK_EQ(*(uint8_t *)pu, 0x02);
    CHECK_EQ(*(uint8_t *)pd, 0x03);
    // Still one expert, so one slot.
    CHECK_EQ(cache.slots_used(), 1);
}

static void test_partial_fill_misses_on_the_absent_projection() {
    CASE("partial projection fill: only the filled projection hits");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));

    const auto w = weights(0x77);
    cache.record_miss(0, 1, GATE);
    CHECK(cache.promote_projection(0, 1, GATE, w.data(), PROJ_BYTES) != nullptr);

    CHECK(cache.lookup(0, 1, GATE, PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(0, 1, UP,   PROJ_BYTES) == nullptr);
    CHECK(cache.lookup(0, 1, DOWN, PROJ_BYTES) == nullptr);
}

static void test_wrong_geometry_is_rejected() {
    CASE("a projection of unexpected size is never cached");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));

    std::vector<uint8_t> wrong(PROJ_BYTES / 2, 0x99);
    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, wrong.data(),
                                   PROJ_BYTES / 2) == nullptr);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES / 2) == nullptr);
    CHECK_EQ(cache.slots_used(), 0);
}

static void test_unknown_geometry_refuses_to_initialize() {
    // Guessing a projection size would mean over-reading the host weights.
    CASE("unknown projection geometry fails init closed");
    moe_cache_config cfg = make_config(4);
    cfg.gate_bytes = cfg.up_bytes = cfg.down_bytes = 0;
    moe_expert_cache cache;
    CHECK(!cache.init(cfg, nullptr));
    CHECK(!cache.is_initialized());
}

static void test_unequal_projection_sizes_are_honoured() {
    CASE("unequal gate/up/down sizes each get their own region");
    moe_cache_config cfg = make_config(4);
    cfg.gate_bytes = 128;
    cfg.up_bytes   = 256;
    cfg.down_bytes = 512;
    cfg.expert_bytes = 128 + 256 + 512;
    cfg.budget_bytes = cfg.expert_bytes * 4;
    moe_expert_cache cache;
    CHECK(cache.init(cfg, nullptr));

    std::vector<uint8_t> g(128, 0xA1), u(256, 0xB2), d(512, 0xC3);
    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, g.data(), 128) != nullptr);
    cache.record_miss(0, 0, UP);
    CHECK(cache.promote_projection(0, 0, UP, u.data(), 256) != nullptr);
    cache.record_miss(0, 0, DOWN);
    CHECK(cache.promote_projection(0, 0, DOWN, d.data(), 512) != nullptr);

    CHECK_EQ(memcmp(cache.lookup(0, 0, GATE, 128), g.data(), 128), 0);
    CHECK_EQ(memcmp(cache.lookup(0, 0, UP,   256), u.data(), 256), 0);
    CHECK_EQ(memcmp(cache.lookup(0, 0, DOWN, 512), d.data(), 512), 0);
}

// ---------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------

static void test_lru_evicts_the_least_recently_used() {
    CASE("LRU evicts the least recently used expert");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(2, /*admission=*/1, "lru"), nullptr));

    use_expert(cache, 0, 0);
    use_expert(cache, 0, 1);
    CHECK_EQ(cache.slots_used(), 2);

    // Touch expert 0 so expert 1 becomes least recent.
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);

    use_expert(cache, 0, 2);
    CHECK_EQ(cache.slots_used(), 2);
    CHECK(cache.lookup(0, 2, GATE, PROJ_BYTES) != nullptr);
    CHECK(cache.lookup(0, 1, GATE, PROJ_BYTES) == nullptr);
    CHECK(cache.stats().evictions.load() > 0);
}

static void test_slru_keeps_a_reused_expert_over_a_one_shot() {
    CASE("SLRU protects a reused expert from a one-shot");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(2, /*admission=*/1, "slru"), nullptr));

    use_expert(cache, 0, 0);
    // Repeated hits promote expert 0 into the protected segment.
    for (int i = 0; i < 4; i++) {
        CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
    }
    use_expert(cache, 0, 1);
    use_expert(cache, 0, 2);

    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
}

// ---------------------------------------------------------------------
// Phase and reset
// ---------------------------------------------------------------------

static void test_prefill_does_not_warm_the_cache_by_default() {
    CASE("prefill admission off: prompt processing does not populate");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1, "lru",
                                 /*prefill_admit=*/false), nullptr));
    cache.set_phase(true);
    for (int i = 0; i < 5; i++) {
        use_expert(cache, 0, 0);
    }
    CHECK_EQ(cache.slots_used(), 0);
    CHECK(cache.stats().prefill_misses.load() > 0);

    // Decode resumes normal admission.
    cache.set_phase(false);
    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
}

static void test_prefill_admission_on_populates_during_prefill() {
    CASE("prefill admission on: prompt processing populates");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1, "lru",
                                 /*prefill_admit=*/true), nullptr));
    cache.set_phase(true);
    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
}

static void test_reset_clears_everything() {
    CASE("reset empties the cache and its counters");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));
    use_expert(cache, 0, 0);
    use_expert(cache, 0, 1);
    CHECK(cache.slots_used() > 0);

    cache.reset();
    CHECK_EQ(cache.slots_used(), 0);
    CHECK_EQ((long long)cache.stats().hits.load(), 0LL);
    CHECK_EQ((long long)cache.stats().misses.load(), 0LL);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) == nullptr);

    // Usable again after reset.
    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
}

static void test_out_of_range_keys_are_ignored() {
    CASE("out-of-range layer/expert/projection never corrupt state");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));
    const auto w = weights(0x42);

    cache.record_miss(-1, 0, GATE);
    cache.record_miss(0, -1, GATE);
    cache.record_miss(0, 0, 99);
    cache.record_miss(999, 0, GATE);
    CHECK(cache.promote_projection(999, 0, GATE, w.data(), PROJ_BYTES) == nullptr);
    CHECK(cache.promote_projection(0, 0, 99, w.data(), PROJ_BYTES) == nullptr);
    CHECK(cache.lookup(999, 0, GATE, PROJ_BYTES) == nullptr);
    CHECK_EQ(cache.slots_used(), 0);
}

static void test_metrics_distinguish_hits_from_fallbacks() {
    CASE("metrics separate cache-served projections from host fallbacks");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));

    use_expert(cache, 0, 0);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
    CHECK(cache.stats().cache_served_projections.load() > 0);

    CHECK_EQ((long long)cache.stats().host_weight_copy_fallbacks.load(), 0LL);
    cache.inc_host_weight_copy_fallback();
    CHECK_EQ((long long)cache.stats().host_weight_copy_fallbacks.load(), 1LL);

    const std::string json = cache.stats_json();
    CHECK(json.find("cache_served_projections") != std::string::npos);
    CHECK(json.find("host_weight_copy_fallbacks") != std::string::npos);
    // The old name described CPU expert execution, which does not exist.
    CHECK(json.find("cpu_expert_calls") == std::string::npos);
}

// ---------------------------------------------------------------------
// Tensor-name parsing (the naming rules the hook lives by)
// ---------------------------------------------------------------------

static void test_parse_recognizes_separate_expert_tensors() {
    CASE("parse: separate gate/up/down expert tensors");
    CHECK_EQ(moe_cache_parse_layer("blk.5.ffn_gate_exps.weight"), 5);
    CHECK_EQ(moe_cache_parse_layer("blk.17.ffn_down_exps.weight"), 17);
    CHECK_EQ(moe_cache_parse_projection("blk.5.ffn_gate_exps.weight"), GATE);
    CHECK_EQ(moe_cache_parse_projection("blk.5.ffn_up_exps.weight"), UP);
    CHECK_EQ(moe_cache_parse_projection("blk.5.ffn_down_exps.weight"), DOWN);
}

static void test_parse_maps_fused_gate_up_to_projection_zero() {
    // Fused layouts have no separate up tensor; the combined weights ride
    // in projection 0 and projection 1 simply never occurs.
    CASE("parse: fused gate_up maps to projection 0");
    CHECK_EQ(moe_cache_parse_layer("blk.3.ffn_gate_up_exps.weight"), 3);
    CHECK_EQ(moe_cache_parse_projection("blk.3.ffn_gate_up_exps.weight"), GATE);
}

static void test_parse_rejects_non_expert_tensors() {
    CASE("parse: shared-expert and non-expert tensors are rejected");
    // Shared experts ("shexp") are dense weights placed like any other --
    // caching them per routed expert would be wrong.
    CHECK_EQ(moe_cache_parse_layer("blk.5.ffn_gate_shexp.weight"), -1);
    CHECK_EQ(moe_cache_parse_layer("blk.5.attn_q.weight"), -1);
    CHECK_EQ(moe_cache_parse_layer("output.weight"), -1);
    CHECK_EQ(moe_cache_parse_layer(nullptr), -1);
    CHECK_EQ(moe_cache_parse_projection("blk.5.attn_q.weight"), -1);
    CHECK_EQ(moe_cache_parse_projection(nullptr), -1);
}

// ---------------------------------------------------------------------
// Learned geometry (deferred init)
// ---------------------------------------------------------------------

static moe_cache_config deferred_config(size_t budget) {
    moe_cache_config cfg;
    cfg.device_id = 0;
    cfg.n_layers  = 4;
    cfg.n_experts = 8;
    cfg.budget_bytes = budget;
    cfg.policy = "lru";
    cfg.admission_misses = 1;
    cfg.host_only_for_testing = true;
    return cfg;
}

// Distinct stable origins standing in for tensor base addresses.
static char g_origin_tag[16];
static const void * origin(int i) { return g_origin_tag + i; }

static void test_learned_geometry_honours_unequal_sizes() {
    // The bug this exists for: geometry used to be copied from whichever
    // tensor the hook saw first, so gate=128/up=256/down=512 became
    // 128/128/128 and two of three projections were never cacheable.
    CASE("learned geometry: unequal gate/up/down sizes from observation");
    moe_expert_cache cache;
    CHECK(cache.init_deferred(deferred_config((128 + 256 + 512) * 4), nullptr));
    CHECK(cache.is_learning());

    // Nothing caches while learning.
    std::vector<uint8_t> g(128, 0xA1), u(256, 0xB2), d(512, 0xC3);
    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, g.data(), 128) == nullptr);

    // One full pass: three tensors observed, then the first one revisits.
    CHECK(!cache.observe_geometry(0, GATE, 128, origin(0)));
    CHECK(!cache.observe_geometry(0, UP,   256, origin(1)));
    CHECK(!cache.observe_geometry(0, DOWN, 512, origin(2)));
    CHECK(cache.observe_geometry(0, GATE, 128, origin(0)));  // wraparound
    CHECK(!cache.is_learning());
    CHECK(cache.is_initialized());

    // Each projection now has a correctly sized region.
    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, g.data(), 128) != nullptr);
    cache.record_miss(0, 0, UP);
    CHECK(cache.promote_projection(0, 0, UP, u.data(), 256) != nullptr);
    cache.record_miss(0, 0, DOWN);
    CHECK(cache.promote_projection(0, 0, DOWN, d.data(), 512) != nullptr);
    CHECK_EQ(memcmp(cache.lookup(0, 0, DOWN, 512), d.data(), 512), 0);
}

static void test_learned_geometry_supports_fused_gate_up() {
    // A fused model has projections 0 (gate_up, double width) and 2 (down)
    // only. The old first-tensor guess sized all three regions to the
    // fused width and the guard then rejected down forever.
    CASE("learned geometry: fused gate_up layout (no separate up)");
    moe_expert_cache cache;
    CHECK(cache.init_deferred(deferred_config((1024 + 512) * 4), nullptr));

    std::vector<uint8_t> gu(1024, 0x11), d(512, 0x22);
    CHECK(!cache.observe_geometry(0, GATE, 1024, origin(0)));
    CHECK(!cache.observe_geometry(0, DOWN, 512,  origin(1)));
    // Wraparound on the first tensor finalizes.
    CHECK(cache.observe_geometry(0, GATE, 1024, origin(0)));
    CHECK(!cache.is_learning());

    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, gu.data(), 1024) != nullptr);
    cache.record_miss(0, 0, DOWN);
    CHECK(cache.promote_projection(0, 0, DOWN, d.data(), 512) != nullptr);
    // The projection that does not exist for this model is never served.
    CHECK(cache.lookup(0, 0, UP, 1024) == nullptr);
    CHECK(cache.lookup(0, 0, UP, 0) == nullptr);
}

static void test_learned_geometry_rejects_later_mismatches() {
    // Mixed quantization across layers: a layer whose projection size
    // differs from the learned one is simply not cached (partial
    // cacheability), never over-read.
    CASE("learned geometry: mismatched layer sizes are uncacheable, not corrupting");
    moe_expert_cache cache;
    CHECK(cache.init_deferred(deferred_config(256 * 3 * 4), nullptr));
    CHECK(!cache.observe_geometry(0, GATE, 256, origin(0)));
    CHECK(!cache.observe_geometry(0, UP,   256, origin(1)));
    CHECK(!cache.observe_geometry(0, DOWN, 256, origin(2)));
    CHECK(cache.observe_geometry(0, GATE, 256, origin(0)));

    std::vector<uint8_t> bigger(512, 0x99);
    cache.record_miss(1, 0, GATE);
    CHECK(cache.promote_projection(1, 0, GATE, bigger.data(), 512) == nullptr);
    CHECK(cache.lookup(1, 0, GATE, 512) == nullptr);
    CHECK_EQ(cache.slots_used(), 0);
}

static void test_no_observations_never_initializes() {
    CASE("learned geometry: zero observations means no pool, fail closed");
    moe_expert_cache cache;
    CHECK(cache.init_deferred(deferred_config(1 << 20), nullptr));
    CHECK(cache.is_learning());
    // Lookups and promotes are inert until geometry exists.
    CHECK(cache.lookup(0, 0, GATE, 256) == nullptr);
    std::vector<uint8_t> w(256, 0x42);
    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, w.data(), 256) == nullptr);
}

// ---------------------------------------------------------------------
// Tensor identity (two models sharing one device cache)
// ---------------------------------------------------------------------

static void test_same_key_different_model_is_a_miss() {
    // Two models loaded on one device (main + draft/MTP) share the cache
    // and use identical tensor names. blk.0 expert 0 of model B must not
    // be served model A's weights just because the names collide.
    CASE("tensor identity: same (layer,expert,proj) from another model misses");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1), nullptr));

    std::vector<uint8_t> model_a(PROJ_BYTES, 0xAA);
    std::vector<uint8_t> model_b(PROJ_BYTES, 0xBB);

    cache.record_miss(0, 0, GATE);
    CHECK(cache.promote_projection(0, 0, GATE, model_a.data(), PROJ_BYTES) != nullptr);

    // Model A's own lookup hits.
    void * hit = cache.lookup(0, 0, GATE, PROJ_BYTES, model_a.data());
    CHECK(hit != nullptr);
    CHECK_EQ(memcmp(hit, model_a.data(), PROJ_BYTES), 0);

    // Model B presents the same key with a different tensor: miss, never
    // model A's bytes.
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES, model_b.data()) == nullptr);
}

// ---------------------------------------------------------------------
// Hybrid staging plan
// ---------------------------------------------------------------------

static void test_hybrid_plan_roundtrip_and_single_take() {
    // The hook records skips at staging time; the op takes the plan
    // exactly once. A second take must come back empty -- a stale plan
    // surviving into the next graph run would route rows to weights that
    // WERE staged that time.
    CASE("hybrid plan: record, take once, then empty");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4), nullptr));

    char cpy_a[8], host_w[8];
    cache.hybrid_record_skip(cpy_a, 3, host_w, /*wtype=*/0);
    cache.hybrid_record_skip(cpy_a, 5, host_w + 4, 0);
    cache.hybrid_record_skip(cpy_a, 3, host_w, 0);  // duplicate: ignored
    CHECK(cache.hybrid_plan_pending(cpy_a));

    auto plan = cache.hybrid_take_plan(cpy_a);
    CHECK_EQ((long long)plan.skips.size(), 2LL);
    CHECK(plan.has_expert(3));
    CHECK(plan.has_expert(5));
    CHECK(!plan.has_expert(4));
    CHECK_EQ(plan.host_src_for(5), (const void *)(host_w + 4));

    CHECK(!cache.hybrid_plan_pending(cpy_a));
    CHECK(cache.hybrid_take_plan(cpy_a).empty());
}

static void test_hybrid_purge_drops_pending_plans() {
    // The scheduler's abandon hook fires when a graph dies mid-compute:
    // its ops never take their plans, and an orphaned plan taken later by
    // an unrelated tensor at a reused base would compute with the wrong
    // host weights.
    CASE("hybrid purge: abandoned plans do not survive");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4), nullptr));

    char cpy_a[8], cpy_b[8], host_w[8];
    cache.hybrid_record_skip(cpy_a, 3, host_w, 0);
    cache.hybrid_record_skip(cpy_b, 5, host_w + 4, 0);
    CHECK(cache.hybrid_plan_pending(cpy_a));
    CHECK(cache.hybrid_plan_pending(cpy_b));

    cache.hybrid_purge_plans();
    CHECK(!cache.hybrid_plan_pending(cpy_a));
    CHECK(!cache.hybrid_plan_pending(cpy_b));
    CHECK(cache.hybrid_take_plan(cpy_a).empty());
}

static void test_staged_bases_survive_reset() {
    // Reorder safety: a base the hook has ever staged into must stay
    // marked. Clearing on reset() could race a stage already in flight,
    // and a stale "staged" mark only costs the reorder optimization.
    CASE("staged bases: sticky, and reset does not clear them");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4), nullptr));

    char base[8];
    CHECK(!cache.is_staged_base(base));
    cache.note_staged_base(base);
    CHECK(cache.is_staged_base(base));

    cache.reset();
    CHECK(cache.is_staged_base(base));
}

static void test_admission_blocked_by_phase_tracks_prefill_policy() {
    // The hybrid skip must not fire when promotion was declined by phase
    // policy rather than pressure -- otherwise a cold prompt executes as
    // per-row scalar CPU GEMVs over every uncached expert.
    CASE("admission_blocked_by_phase: prefill with admission off, only");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4, /*admission=*/1, "lru",
                                 /*prefill_admit=*/false), nullptr));
    CHECK(!cache.admission_blocked_by_phase());
    cache.set_phase(true);
    CHECK(cache.admission_blocked_by_phase());
    cache.set_phase(false);
    CHECK(!cache.admission_blocked_by_phase());

    moe_expert_cache admit;
    CHECK(admit.init(make_config(4, /*admission=*/1, "lru",
                                 /*prefill_admit=*/true), nullptr));
    admit.set_phase(true);
    CHECK(!admit.admission_blocked_by_phase());
}

static void test_hybrid_plans_are_isolated_per_staged_tensor() {
    // Two tensors staged concurrently (main + draft context) must never
    // see each other's skips.
    CASE("hybrid plan: keys are per staged tensor");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(4), nullptr));

    char cpy_a[4], cpy_b[4], w[4];
    cache.hybrid_record_skip(cpy_a, 1, w, 0);
    cache.hybrid_record_skip(cpy_b, 2, w, 0);
    auto pa = cache.hybrid_take_plan(cpy_a);
    CHECK(pa.has_expert(1));
    CHECK(!pa.has_expert(2));
    auto pb = cache.hybrid_take_plan(cpy_b);
    CHECK(pb.has_expert(2));
}

// ---------------------------------------------------------------------
// Concurrent lifecycle (two compute threads + a metrics/reset thread)
// ---------------------------------------------------------------------

static void test_concurrent_use_and_reset_survive() {
    // Main + draft contexts each have a compute thread, and the server's
    // HTTP threads call stats/reset during unload. This cannot prove the
    // absence of races, but it exercises every locked path concurrently
    // and fails loudly (crash/deadlock) if the locking regresses.
    CASE("concurrency: two users plus stats/reset run to completion");
    moe_expert_cache cache;
    CHECK(cache.init(make_config(8, /*admission=*/1), nullptr));

    auto user = [&cache](int layer_base) {
        for (int round = 0; round < 200; round++) {
            for (int e = 0; e < 8; e++) {
                use_expert(cache, layer_base + (round % 2), e,
                           (uint8_t)(e * 7 + 1));
            }
        }
    };
    std::thread a(user, 0);
    std::thread b(user, 2);
    std::thread c([&cache]() {
        for (int i = 0; i < 100; i++) {
            (void)cache.stats_json();
            (void)cache.slots_used();
            if (i % 25 == 24) cache.reset();
        }
    });
    a.join();
    b.join();
    c.join();
    // Still consistent and usable afterwards.
    use_expert(cache, 0, 0, 0x5C);
    CHECK(cache.lookup(0, 0, GATE, PROJ_BYTES) != nullptr);
}

int main() {
    test_admission_threshold_one();
    test_admission_threshold_two_needs_a_second_use();
    test_admission_threshold_three();
    test_cached_projection_does_not_reset_others();
    test_all_projections_eventually_populate();

    test_cached_bytes_are_the_promoted_bytes();
    test_projections_occupy_distinct_regions();
    test_partial_fill_misses_on_the_absent_projection();
    test_wrong_geometry_is_rejected();
    test_unknown_geometry_refuses_to_initialize();
    test_unequal_projection_sizes_are_honoured();

    test_lru_evicts_the_least_recently_used();
    test_slru_keeps_a_reused_expert_over_a_one_shot();

    test_prefill_does_not_warm_the_cache_by_default();
    test_prefill_admission_on_populates_during_prefill();
    test_reset_clears_everything();
    test_out_of_range_keys_are_ignored();
    test_metrics_distinguish_hits_from_fallbacks();

    test_parse_recognizes_separate_expert_tensors();
    test_parse_maps_fused_gate_up_to_projection_zero();
    test_parse_rejects_non_expert_tensors();

    test_learned_geometry_honours_unequal_sizes();
    test_learned_geometry_supports_fused_gate_up();
    test_learned_geometry_rejects_later_mismatches();
    test_no_observations_never_initializes();

    test_same_key_different_model_is_a_miss();
    test_hybrid_plan_roundtrip_and_single_take();
    test_hybrid_plans_are_isolated_per_staged_tensor();
    test_hybrid_purge_drops_pending_plans();
    test_staged_bases_survive_reset();
    test_admission_blocked_by_phase_tracks_prefill_policy();
    test_concurrent_use_and_reset_survive();

    if (g_failures) {
        fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all MoE cache tests passed\n");
    return 0;
}
