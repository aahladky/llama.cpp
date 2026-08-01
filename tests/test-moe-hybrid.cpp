// Deterministic unit tests for the hybrid MoE partition and its CPU
// miss executor.
//
// Host-only, like test-moe-cache.cpp: the partition, merge, and CPU-tier
// math are plain bookkeeping, and a test that needs two Arc cards and a
// 31 GiB model is a test that does not get run. GPU hit dispatch needs
// real devices and is covered by the fixtures under
// scripts/moe-cache-correctness/.
//
// The cases that matter most are the ones the previous representation got
// wrong: two experts contributing to one token, and routing coefficients
// being applied at all.

#include "ggml-sycl/moe-hybrid.hpp"

#ifdef GGML_MOE_HYBRID_CPU_KERNELS
#include "ggml-cpu.h"
#endif

#include <cmath>
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

// --- CPU miss execution ----------------------------------------------

#include "ggml.h"

// A small deterministic expert-weight fixture: n_expert experts of
// ne01 x ne00 f32 weights with knowable values.
static std::vector<float> f32_experts(int n_expert, int64_t ne01, int64_t ne00) {
    std::vector<float> w((size_t)n_expert * ne01 * ne00);
    for (size_t i = 0; i < w.size(); i++) {
        // Bounded, sign-alternating, expert-distinct.
        w[i] = ((int)(i % 17) - 8) * 0.125f + (float)(i / (ne01 * ne00)) * 0.5f;
    }
    return w;
}

static std::vector<float> ramp_activation(int64_t ne00) {
    std::vector<float> a((size_t)ne00);
    for (int64_t j = 0; j < ne00; j++) {
        a[(size_t)j] = 0.01f * (float)(j % 29) - 0.1f;
    }
    return a;
}

// Reference: plain double-accumulated dot over f32 weights.
static float ref_dot(const float * w_row, const float * act, int64_t ne00) {
    double acc = 0.0;
    for (int64_t j = 0; j < ne00; j++) {
        acc += (double)w_row[j] * (double)act[j];
    }
    return (float)acc;
}

static moe_hybrid_partition all_miss_partition(const int32_t * experts,
                                               int64_t n_rows) {
    return moe_build_partition(experts, nullptr, nullptr, n_rows,
                               /*layer=*/0, /*projection=*/0,
                               /*cache=*/nullptr);
}

static void test_cpu_misses_match_the_f32_reference() {
    CASE("G3: f32 miss execution matches the reference dot");
    const int64_t ne00 = 96, ne01 = 5;
    const int n_expert = 4;
    const auto w = f32_experts(n_expert, ne01, ne00);
    const auto act = ramp_activation(ne00);

    const int32_t experts[2] = {1, 3};
    auto part = all_miss_partition(experts, 2);
    std::vector<float> out(2 * (size_t)ne01, -777.0f);

    const int64_t n = moe_cpu_execute_misses(
        part, w.data(), (size_t)ne01 * ne00 * sizeof(float), GGML_TYPE_F32,
        ne00, ne01, act.data(), /*act_stride=*/0, out.data());
    CHECK(n == 2);
    for (int k = 0; k < 2; k++) {
        const float * exp_w = w.data() + (size_t)experts[k] * ne01 * ne00;
        for (int64_t o = 0; o < ne01; o++) {
            CHECK_NEAR(out[(size_t)k * ne01 + o],
                       ref_dot(exp_w + o * ne00, act.data(), ne00));
        }
    }
}

static void test_cpu_misses_dequantize_with_ggml_math() {
    // Quantize a known f32 fixture with ggml's own quantizer, run the
    // executor over the quantized blocks, and compare against a reference
    // computed independently here. This pins the plumbing (row size,
    // expert stride, partition order) AND that the executor does real
    // quantized math rather than reinterpreting blocks as floats --
    // defect 4 of the old scaffold.
    //
    // Which reference depends on what the executor is built to use. With
    // ggml-cpu's kernels linked it computes an integer dot against the
    // activation quantized to vec_dot_type, so the exact reference is
    // dequantized weights against the DEQUANTIZED-REQUANTIZED activation;
    // holding it to the f32 activation instead would only measure the
    // quantizer's error. Without them it dequantizes and dots in f64, and
    // the f32 activation is exact. Both bars are tight enough that block
    // reinterpretation fails them by orders of magnitude.
    CASE("G3: quantized miss execution goes through ggml's dequantizer");
    const int64_t ne00 = 256;  // one Q4_K / Q8_0 superblock multiple
    const int64_t ne01 = 3;
    const int n_expert = 3;
    const auto w = f32_experts(n_expert, ne01, ne00);
    const auto act = ramp_activation(ne00);

    for (const enum ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K,
                                      GGML_TYPE_Q6_K}) {
        const size_t row_bytes = ggml_row_size(type, ne00);
        const size_t expert_bytes = row_bytes * (size_t)ne01;
        std::vector<uint8_t> q(expert_bytes * n_expert);
        // Quantize per expert so rows stay row_bytes apart, as in a GGUF.
        for (int e = 0; e < n_expert; e++) {
            ggml_quantize_chunk(type, w.data() + (size_t)e * ne01 * ne00,
                                q.data() + (size_t)e * expert_bytes,
                                0, ne01, ne00, nullptr);
        }

        const int32_t experts[2] = {0, 2};
        auto part = all_miss_partition(experts, 2);
        std::vector<float> out(2 * (size_t)ne01, 0.0f);
        const int64_t n = moe_cpu_execute_misses(
            part, q.data(), expert_bytes, type, ne00, ne01,
            act.data(), 0, out.data());
        CHECK(n == 2);

        // Independent reference: dequantize with the public traits and dot.
        const auto * traits = ggml_get_type_traits(type);
        std::vector<float> row((size_t)ne00);
#ifdef GGML_MOE_HYBRID_CPU_KERNELS
        // The kernel dots the weights against the activation quantized to
        // vec_dot_type, so it cannot equal the f32 reference. The gap is
        // bounded, though: an int8 quantizer with a per-block scale puts
        // every element within (max|act| / 254) of the original, and the
        // dot's error is at most that times sum|w|. That bound is a few
        // parts in ten thousand of the value here -- reinterpreting the
        // blocks as floats misses it by three orders of magnitude, which
        // is the thing this case exists to catch.
        double act_max = 0.0;
        for (int64_t j = 0; j < ne00; j++) {
            act_max = std::max(act_max, (double)std::fabs(act[(size_t)j]));
        }
        const double act_step = act_max / 254.0;
#endif
        for (int k = 0; k < 2; k++) {
            for (int64_t o = 0; o < ne01; o++) {
                const uint8_t * wrow = q.data()
                    + (size_t)experts[k] * expert_bytes + (size_t)o * row_bytes;
                traits->to_float(wrow, row.data(), ne00);
                const float ref = ref_dot(row.data(), act.data(), ne00);
                const float got = out[(size_t)k * ne01 + o];
#ifdef GGML_MOE_HYBRID_CPU_KERNELS
                double w_abs = 0.0;
                for (int64_t j = 0; j < ne00; j++) {
                    w_abs += (double)std::fabs(row[(size_t)j]);
                }
                // x2 for rounding of the block scale itself.
                const double tol = 2.0 * act_step * w_abs + 1e-5;
                if (!(std::fabs((double)got - (double)ref) <= tol)) {
                    fprintf(stderr, "FAIL [%s] %s:%d: %g != %g (tol %g)\n",
                            g_case.c_str(), __FILE__, __LINE__,
                            (double)got, (double)ref, tol);
                    g_failures++;
                }
#else
                CHECK_NEAR(got, ref);
#endif
            }
        }
    }
}

static void test_cpu_misses_leave_hit_outputs_untouched() {
    CASE("G3: GPU-hit entries in the output buffer are not written");
    const int64_t ne00 = 32, ne01 = 2;
    const auto w = f32_experts(2, ne01, ne00);
    const auto act = ramp_activation(ne00);

    const int32_t experts[2] = {0, 1};
    auto part = all_miss_partition(experts, 2);
    part.contributions[0].tier = MOE_TIER_GPU_HIT;  // as if resident

    std::vector<float> out(2 * (size_t)ne01, -5.0f);
    const int64_t n = moe_cpu_execute_misses(
        part, w.data(), (size_t)ne01 * ne00 * sizeof(float), GGML_TYPE_F32,
        ne00, ne01, act.data(), 0, out.data());
    CHECK(n == 1);
    CHECK_NEAR(out[0], -5.0f);  // hit entry untouched
    CHECK_NEAR(out[1], -5.0f);
    CHECK(out[(size_t)ne01] != -5.0f);  // miss entry computed
}

static void test_cpu_misses_fail_closed_on_unknown_types() {
    CASE("G3: unsupported weight types fail closed, never guess");
    const int32_t experts[1] = {0};
    auto part = all_miss_partition(experts, 1);
    float out[4] = {0};
    float act[4] = {0};
    uint8_t bogus[64] = {0};
    CHECK(moe_cpu_execute_misses(part, bogus, 64, /*wtype=*/-1,
                                 4, 1, act, 0, out) == -1);
    CHECK(moe_cpu_execute_misses(part, bogus, 64, /*wtype=*/GGML_TYPE_COUNT,
                                 4, 1, act, 0, out) == -1);
    CHECK(moe_cpu_execute_misses(part, nullptr, 64, GGML_TYPE_F32,
                                 4, 1, act, 0, out) == -1);
}

static void test_parallel_gemv_batch_matches_sequential() {
    // The threaded batch must be bit-identical to sequential execution:
    // chunks only partition WHICH rows a worker computes, never the math
    // or the order within a row.
    CASE("G4: parallel gemv batch is bit-identical to sequential");
    const int64_t ne00 = 128, ne01 = 6;
    const int n_expert = 5;
    const auto w = f32_experts(n_expert, ne01, ne00);
    const auto act = ramp_activation(ne00);

    const size_t n_jobs = 12;
    std::vector<float> seq(n_jobs * (size_t) ne01), par(n_jobs * (size_t) ne01);
    std::vector<moe_cpu_gemv_job> js(n_jobs), jp(n_jobs);
    for (size_t k = 0; k < n_jobs; k++) {
        const float * ew = w.data() + (k % n_expert) * (size_t)(ne01 * ne00);
        js[k] = {ew, act.data(), seq.data() + k * (size_t) ne01};
        jp[k] = {ew, act.data(), par.data() + k * (size_t) ne01};
    }
    CHECK(moe_cpu_execute_gemvs(js.data(), n_jobs, GGML_TYPE_F32, ne00, ne01, 1));
    CHECK(moe_cpu_execute_gemvs(jp.data(), n_jobs, GGML_TYPE_F32, ne00, ne01, 8));
    CHECK(memcmp(seq.data(), par.data(), seq.size() * sizeof(float)) == 0);

    // Unsupported type fails closed without touching outputs.
    par.assign(par.size(), -3.0f);
    CHECK(!moe_cpu_execute_gemvs(jp.data(), n_jobs, /*wtype=*/-1, ne00, ne01, 8));
    CHECK_NEAR(par[0], -3.0f);
}

static void test_cpu_misses_then_merge_produce_the_routed_sum() {
    // End to end for the pieces that exist: partition (all miss) ->
    // CPU execution -> weighted merge equals the directly computed
    // routed-MoE output for a two-experts-per-token batch.
    CASE("G3+G5: executed misses merge into the correct routed sum");
    const int64_t ne00 = 64, ne01 = 4;
    const int n_expert = 4;
    const auto w = f32_experts(n_expert, ne01, ne00);
    const auto act = ramp_activation(ne00);

    // One token, two experts, distinct routing weights.
    const int32_t experts[2] = {1, 2};
    const float   weights_r[2] = {0.7f, 0.3f};
    const int32_t rows[2] = {0, 0};
    auto part = moe_build_partition(experts, weights_r, rows, 2, 0, 0, nullptr);

    std::vector<float> tier_out(2 * (size_t)ne01, 0.0f);
    CHECK(moe_cpu_execute_misses(part, w.data(),
                                 (size_t)ne01 * ne00 * sizeof(float),
                                 GGML_TYPE_F32, ne00, ne01,
                                 act.data(), 0, tier_out.data()) == 2);

    std::vector<float> dst((size_t)ne01, 123.0f);
    moe_merge_contributions(part, tier_out.data(), dst.data(), ne01, true);

    for (int64_t o = 0; o < ne01; o++) {
        const float a = ref_dot(w.data() + (size_t)1 * ne01 * ne00 + o * ne00,
                                act.data(), ne00);
        const float b = ref_dot(w.data() + (size_t)2 * ne01 * ne00 + o * ne00,
                                act.data(), ne00);
        CHECK_NEAR(dst[(size_t)o], 0.7f * a + 0.3f * b);
    }
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

    test_cpu_misses_match_the_f32_reference();
    test_cpu_misses_dequantize_with_ggml_math();
    test_cpu_misses_leave_hit_outputs_untouched();
    test_cpu_misses_fail_closed_on_unknown_types();
    test_parallel_gemv_batch_matches_sequential();
    test_cpu_misses_then_merge_produce_the_routed_sum();

    if (g_failures) {
        fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all MoE hybrid partition tests passed\n");
    return 0;
}
