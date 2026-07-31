// Deterministic unit tests for the hybrid MoE partition (roadmap Task G2).
//
// Host-only, in the shape Task F8 established: the partition and merge are
// plain bookkeeping, and a test that needs two Arc cards and a 31 GiB model
// is a test that does not get run. Execution (G3/G4) is not implemented, so
// nothing here claims to test it.
//
// The cases that matter most are the ones the previous representation got
// wrong: two experts contributing to one token, and routing coefficients
// being applied at all.

#include "ggml-sycl/moe-hybrid.hpp"

#include <cstdio>
#include <cstdlib>
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

#define CHECK_NEAR(a, b) do {                                               \
    const double d = (double)(a) - (double)(b);                             \
    if (d > 1e-5 || d < -1e-5) {                                            \
        fprintf(stderr, "FAIL [%s] %s:%d: %g != %g\n",                      \
                g_case.c_str(), __FILE__, __LINE__, (double)(a), (double)(b)); \
        g_failures++;                                                       \
    }                                                                       \
} while (0)

// The policy side of moe-cache links here; the device side does not.
// Host-only mode never reaches these, so they abort rather than no-op.
static void device_path_unreachable(const char * what) {
    fprintf(stderr, "moe_hybrid: host-only test reached the device path (%s)\n", what);
    abort();
}
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

// A host-only cache with room for a handful of experts.
static moe_cache_config host_cfg(int n_layers = 2, int n_experts = 8,
                                 size_t proj = 64) {
    moe_cache_config cfg;
    cfg.device_id = 0;
    cfg.n_layers = n_layers;
    cfg.n_experts = n_experts;
    cfg.gate_bytes = proj;
    cfg.up_bytes = proj;
    cfg.down_bytes = proj;
    cfg.expert_bytes = proj * 3;
    cfg.budget_bytes = proj * 3 * 4;   // 4 slots
    cfg.policy = "lru";
    cfg.admission_misses = 1;
    cfg.host_only_for_testing = true;
    return cfg;
}

// Promotion is gated on admission, so a projection only becomes resident
// after the configured number of misses. Skipping record_miss() here made
// every "resident" fixture silently empty.
static void fill_projection(moe_expert_cache & c, int layer, int expert,
                            int projection, size_t proj_bytes) {
    std::vector<char> src(proj_bytes, (char)(expert + 1));
    c.record_miss(layer, expert, projection);
    const void * p = c.promote_projection(layer, expert, projection,
                                          src.data(), proj_bytes);
    if (!p) {
        fprintf(stderr, "fixture: promotion of (%d,%d,%d) failed\n",
                layer, expert, projection);
        abort();
    }
}

// --- residency query -------------------------------------------------

static void test_contains_reports_per_projection_residency() {
    CASE("contains is per projection");
    moe_expert_cache c;
    CHECK(c.init(host_cfg(), nullptr));
    fill_projection(c, 0, 3, 0, 64);   // gate only
    CHECK(c.contains(0, 3, 0));
    CHECK(!c.contains(0, 3, 1));       // up not filled
    CHECK(!c.contains(0, 3, 2));
    CHECK(!c.contains(0, 4, 0));       // different expert
    CHECK(!c.contains(1, 3, 0));       // different layer
}

static void test_contains_does_not_disturb_the_cache() {
    // A partition builder must not change what it is measuring: lookup()
    // counts hits, moves SLRU recency and clears admission progress.
    CASE("contains has no side effects");
    moe_expert_cache c;
    CHECK(c.init(host_cfg(), nullptr));
    fill_projection(c, 0, 1, 0, 64);
    const uint64_t hits_before = c.stats().hits.load();
    const uint64_t misses_before = c.stats().misses.load();
    for (int i = 0; i < 50; i++) {
        (void)c.contains(0, 1, 0);
        (void)c.contains(0, 7, 0);     // absent
    }
    CHECK(c.stats().hits.load() == hits_before);
    CHECK(c.stats().misses.load() == misses_before);
}

static void test_contains_rejects_out_of_range() {
    CASE("contains rejects out-of-range keys");
    moe_expert_cache c;
    CHECK(c.init(host_cfg(), nullptr));
    CHECK(!c.contains(-1, 0, 0));
    CHECK(!c.contains(0, -1, 0));
    CHECK(!c.contains(0, 0, -1));
    CHECK(!c.contains(0, 0, 3));
    CHECK(!c.contains(99, 0, 0));
}

// --- partition -------------------------------------------------------

static void test_partition_classifies_by_residency() {
    CASE("partition classifies by residency");
    moe_expert_cache c;
    CHECK(c.init(host_cfg(), nullptr));
    fill_projection(c, 0, 2, 0, 64);
    fill_projection(c, 0, 5, 0, 64);

    const int32_t experts[] = {2, 3, 5, 6};
    auto p = moe_build_partition(experts, nullptr, nullptr, 4, 0, 0, &c);
    CHECK(p.contributions.size() == 4);
    CHECK(p.n_hits() == 2);
    CHECK(p.n_misses() == 2);
    CHECK(p.contributions[0].tier == MOE_TIER_GPU_HIT);
    CHECK(p.contributions[1].tier == MOE_TIER_CPU_MISS);
    CHECK(p.contributions[2].tier == MOE_TIER_GPU_HIT);
    CHECK(p.contributions[3].tier == MOE_TIER_CPU_MISS);
}

static void test_partition_without_a_cache_is_all_misses() {
    CASE("no cache means every row is a miss");
    const int32_t experts[] = {0, 1, 2};
    auto p = moe_build_partition(experts, nullptr, nullptr, 3, 0, 0, nullptr);
    CHECK(p.all_misses());
    CHECK(p.n_hits() == 0);
}

static void test_partition_preserves_routing_weights() {
    CASE("routing weights are preserved");
    const int32_t experts[] = {0, 1};
    const float   weights[] = {0.25f, 0.75f};
    auto p = moe_build_partition(experts, weights, nullptr, 2, 0, 0, nullptr);
    CHECK_NEAR(p.contributions[0].routing_weight, 0.25f);
    CHECK_NEAR(p.contributions[1].routing_weight, 0.75f);
}

static void test_partition_preserves_original_row_mapping() {
    // The routed batch is reordered by the counting sort; the destination
    // is the pre-sort row, and losing that scatters output to wrong tokens.
    CASE("original rows survive reordering");
    const int32_t experts[] = {0, 1, 2};
    const int32_t original[] = {7, 3, 5};
    auto p = moe_build_partition(experts, nullptr, original, 3, 0, 0, nullptr);
    CHECK(p.contributions[0].original_row == 7);
    CHECK(p.contributions[1].original_row == 3);
    CHECK(p.contributions[2].original_row == 5);
    CHECK(p.contributions[0].contiguous_row == 0);
    CHECK(p.contributions[2].contiguous_row == 2);
}

static void test_partition_records_the_projection() {
    CASE("projection identity is recorded");
    const int32_t experts[] = {0};
    auto p = moe_build_partition(experts, nullptr, nullptr, 1, 0, 2, nullptr);
    CHECK(p.contributions[0].projection == 2);
}

static void test_shared_destination_rows_are_counted_once() {
    // Two experts per token: four contributions, two destination rows.
    CASE("destination rows are distinct from contributions");
    const int32_t experts[]  = {0, 1, 2, 3};
    const int32_t original[] = {0, 0, 1, 1};
    auto p = moe_build_partition(experts, nullptr, original, 4, 0, 0, nullptr);
    CHECK(p.contributions.size() == 4);
    CHECK(p.n_destination_rows() == 2);
}

// --- merge -----------------------------------------------------------

static void test_merge_sums_experts_sharing_a_token() {
    // The defect this representation exists to fix: the previous merge
    // memcpy'd into dst[original_row], so the second expert overwrote the
    // first instead of adding to it.
    CASE("two experts on one token are summed");
    const int32_t experts[]  = {0, 1};
    const int32_t original[] = {0, 0};
    const float   weights[]  = {1.0f, 1.0f};
    auto p = moe_build_partition(experts, weights, original, 2, 0, 0, nullptr);

    const int64_t ne0 = 2;
    const float outputs[] = {1.0f, 2.0f,    10.0f, 20.0f};
    float dst[2] = {0};
    moe_merge_contributions(p, outputs, dst, ne0, true);
    CHECK_NEAR(dst[0], 11.0f);
    CHECK_NEAR(dst[1], 22.0f);
}

static void test_merge_applies_routing_weights() {
    CASE("routing weights are applied");
    const int32_t experts[]  = {0, 1};
    const int32_t original[] = {0, 0};
    const float   weights[]  = {0.25f, 0.75f};
    auto p = moe_build_partition(experts, weights, original, 2, 0, 0, nullptr);

    const float outputs[] = {4.0f,   8.0f};   // ne0 = 1
    float dst[1] = {0};
    moe_merge_contributions(p, outputs, dst, 1, true);
    CHECK_NEAR(dst[0], 0.25f * 4.0f + 0.75f * 8.0f);
}

static void test_merge_scatters_to_original_rows() {
    CASE("merge scatters to pre-sort rows");
    const int32_t experts[]  = {0, 1};
    const int32_t original[] = {2, 0};
    auto p = moe_build_partition(experts, nullptr, original, 2, 0, 0, nullptr);

    const float outputs[] = {5.0f, 9.0f};
    float dst[3] = {0};
    moe_merge_contributions(p, outputs, dst, 1, true);
    CHECK_NEAR(dst[0], 9.0f);
    CHECK_NEAR(dst[1], 0.0f);
    CHECK_NEAR(dst[2], 5.0f);
}

static void test_merge_is_tier_independent() {
    // The same contributions must merge identically whether they were
    // computed on GPU or CPU: results cannot depend on which tier ran.
    CASE("merge does not depend on tier");
    const int32_t experts[]  = {0, 1};
    const int32_t original[] = {0, 0};
    const float   weights[]  = {0.3f, 0.7f};
    auto a = moe_build_partition(experts, weights, original, 2, 0, 0, nullptr);
    auto b = a;
    b.contributions[0].tier = MOE_TIER_GPU_HIT;
    b.contributions[1].tier = MOE_TIER_CPU_MISS;

    const float outputs[] = {2.0f, 6.0f};
    float da[1] = {0}, db[1] = {0};
    moe_merge_contributions(a, outputs, da, 1, true);
    moe_merge_contributions(b, outputs, db, 1, true);
    CHECK_NEAR(da[0], db[0]);
}

static void test_merge_accumulates_when_not_zeroing() {
    CASE("merge can accumulate into existing output");
    const int32_t experts[]  = {0};
    const int32_t original[] = {0};
    auto p = moe_build_partition(experts, nullptr, original, 1, 0, 0, nullptr);
    const float outputs[] = {1.5f};
    float dst[1] = {10.0f};
    moe_merge_contributions(p, outputs, dst, 1, false);
    CHECK_NEAR(dst[0], 11.5f);
}

static void test_merge_tolerates_an_empty_partition() {
    CASE("empty partition is a no-op");
    moe_hybrid_partition p;
    float dst[1] = {3.0f};
    moe_merge_contributions(p, nullptr, dst, 1, false);
    CHECK_NEAR(dst[0], 3.0f);
    CHECK(p.empty());
    CHECK(p.n_destination_rows() == 0);
}

int main() {
    test_contains_reports_per_projection_residency();
    test_contains_does_not_disturb_the_cache();
    test_contains_rejects_out_of_range();

    test_partition_classifies_by_residency();
    test_partition_without_a_cache_is_all_misses();
    test_partition_preserves_routing_weights();
    test_partition_preserves_original_row_mapping();
    test_partition_records_the_projection();
    test_shared_destination_rows_are_counted_once();

    test_merge_sums_experts_sharing_a_token();
    test_merge_applies_routing_weights();
    test_merge_scatters_to_original_rows();
    test_merge_is_tier_independent();
    test_merge_accumulates_when_not_zeroing();
    test_merge_tolerates_an_empty_partition();

    if (g_failures) {
        fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all MoE hybrid partition tests passed\n");
    return 0;
}
