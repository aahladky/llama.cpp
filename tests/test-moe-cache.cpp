// Deterministic unit tests for the SYCL MoE expert cache (roadmap Task F8).
//
// These run the real moe_expert_cache in host-only mode: the slot pool is
// ordinary memory and promotions are plain memcpy, so admission, eviction,
// phase and geometry logic are exercised with no GPU and no model. That is
// deliberate -- the admission bug these were written for (Task F3) is pure
// bookkeeping, and a test that needs a 16 GiB MoE and two Arc cards to run
// is a test that does not run.
//
// Cases requiring real devices (two contexts on one GPU, multi-GPU,
// dynamic backend loading) are not covered here; they need the end-to-end
// fixtures under scripts/moe-cache-correctness/.

#include "ggml-sycl/moe-cache.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
// Admission (Task F3)
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

    if (g_failures) {
        fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all MoE cache tests passed\n");
    return 0;
}
