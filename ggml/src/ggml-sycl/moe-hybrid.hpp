// Hybrid MoE Execution — partition representation.
//
// A routed MoE output is the weighted sum of n_expert_used expert outputs
// per token. The partition therefore is NOT a two-way split of rows: it is
// a list of *contributions*, each one expert's share of one token's
// output, tagged with the tier that will compute it.
//
// A row split with a memcpy merge is wrong output, not an optimization
// gap: with n_expert_used = 2 the second expert overwrites the first,
// and the routing coefficient is unrepresented. Merge must accumulate
// every contribution for a token.
//
// This header is deliberately free of SYCL, like moe-cache.hpp, so the
// partition logic can be unit-tested on any machine.

#ifndef GGML_SYCL_MOE_HYBRID_HPP
#define GGML_SYCL_MOE_HYBRID_HPP

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "moe-cache.hpp"

// Which tier computes a contribution.
enum moe_exec_tier {
    MOE_TIER_GPU_HIT  = 0,  // expert weights resident in the GPU cache
    MOE_TIER_CPU_MISS = 1,  // computed on CPU over host/mmap weights
};

// One expert's share of one token's output.
struct moe_contribution {
    int32_t original_row   = -1;  // destination row in dst, pre-sort order
    int32_t contiguous_row = -1;  // source row in the reordered src1
    int32_t expert_id      = -1;
    // The router's coefficient for this (token, expert). The merge is a
    // weighted sum; a merge without it is wrong output.
    float   routing_weight = 0.0f;
    int32_t projection     = -1;  // 0=gate, 1=up, 2=down
    moe_exec_tier tier     = MOE_TIER_CPU_MISS;
    // For hits, the resident region, so execution does not repeat the
    // residency query the partition already did.
    void *  slot_ptr       = nullptr;
};

// A batch's worth of contributions, in a fixed order.
struct moe_hybrid_partition {
    std::vector<moe_contribution> contributions;

    int n_hits() const {
        int n = 0;
        for (const auto & c : contributions) {
            if (c.tier == MOE_TIER_GPU_HIT) n++;
        }
        return n;
    }
    int n_misses() const { return (int)contributions.size() - n_hits(); }
    bool empty() const { return contributions.empty(); }
    bool all_hits() const { return !empty() && n_misses() == 0; }
    bool all_misses() const { return !empty() && n_hits() == 0; }

    // Distinct destination rows. Not contributions.size(): n_expert_used
    // contributions share one destination row, which is exactly why the
    // merge accumulates rather than copies.
    int n_destination_rows() const;
};

// Build the partition for one (layer, projection) over a routed batch.
//
// expert_ids[i]      -- expert assigned to contiguous row i
// routing_weights[i] -- that expert's coefficient for the row; may be null,
//                       in which case weights are recorded as 1.0
// original_rows[i]   -- destination row for contiguous row i; may be null
//                       when the batch was not reordered, in which case the
//                       contiguous index is used
//
// Residency is queried with moe_expert_cache::contains(), which does not
// disturb cache statistics or eviction order -- a partition must not
// change what it is measuring.
moe_hybrid_partition moe_build_partition(
    const int32_t * expert_ids,
    const float *   routing_weights,
    const int32_t * original_rows,
    int64_t         n_rows,
    int32_t         layer_id,
    int             projection,
    const moe_expert_cache * cache);

// Accumulate contributions into dst:
//   dst[row] = sum over that row's contributions of weight * expert_output
//
// tier_outputs supplies each contribution's computed expert output in the
// partition's own order, ne0 floats each. Ordering is fixed by the
// partition rather than by which tier finished first, so the result does
// not depend on scheduling.
//
// Set zero_dst unless the caller has already zeroed dst; this accumulates.
void moe_merge_contributions(
    const moe_hybrid_partition & partition,
    const float * tier_outputs,
    float *       dst,
    int64_t       ne0,
    bool          zero_dst);

// Compute the CPU tier of one projection's partition at batch 1.
//
// For every MOE_TIER_CPU_MISS contribution k, computes that expert's
// output over the host-resident (typically mmap-backed) weights:
//
//     outputs[k*ne01 + o] = dot(dequant(expert_row_o), activation_row)
//
// weights_base        host base of this projection's expert tensor
//                     (all experts; expert e's blocks start at
//                     weights_base + e * expert_stride_bytes)
// expert_stride_bytes bytes between consecutive experts
// wtype               the weights' ggml_type (int to keep this header
//                     ggml-free; moe-hybrid.cpp validates it)
// ne00                input dim = elements per weight row
// ne01                output rows per expert
// activations         f32 activations; contribution k's row is
//                     activations + contiguous_row * act_stride_floats
// outputs             partition-ordered tier outputs: the
//                     k-th contribution's ne01 floats live at k*ne01.
//                     GPU-hit entries are left untouched.
//
// Returns the number of miss contributions computed, or -1 when the
// weight type has no dequantizer -- the caller must then fail closed to
// the non-hybrid path rather than guess.
//
// Quantized math is ggml's own (ggml_get_type_traits()->to_float per
// row, then an f32 dot). Calling the CPU backend's mul_mat_id
// machinery instead is not reachable from the SYCL backend in
// shared/dynamic builds without a new cross-backend
// interface; the dequantizer in ggml-base is the same reference math
// every CPU kernel is tested against, and the miss path is bound by
// reading the weights (page faults/storage), not by FLOPs, at batch 1.
int64_t moe_cpu_execute_misses(
    const moe_hybrid_partition & partition,
    const void *  weights_base,
    size_t        expert_stride_bytes,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    const float * activations,
    size_t        act_stride_floats,
    float *       outputs);

// Can moe_cpu_execute_misses handle this weight type? The staging hook
// asks BEFORE committing to skip an expert's device copy -- a skip whose
// CPU tier then fails closed would leave garbage rows.
bool moe_cpu_can_execute(int wtype);

// One expert's gemv over host weights: out[o] = dot(dequant(row_o), act)
// for o in [0, ne01). The single-expert core of moe_cpu_execute_misses,
// exposed for the in-op hybrid path, which addresses experts directly
// rather than through a partition. Returns false when the type is
// unsupported (caller must have checked moe_cpu_can_execute).
bool moe_cpu_expert_gemv(
    const void *  expert_weights,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    const float * activation,
    float *       out);

// A batch of independent expert gemvs, executed across threads. This is
// what the in-op CPU tier calls: dequantizing an 800 KB projection per
// routed row is compute-heavy, and single-threaded it measurably LOSES
// to the transfer it replaced (1.9 vs 4.3 t/s on the 122B target).
// Jobs are split into contiguous chunks over at most n_threads workers;
// each job is one expert's full gemv, so rows never interleave and the
// output is bit-identical to the sequential order. Returns false (and
// computes nothing) if any job's type would be unsupported.
struct moe_cpu_gemv_job {
    const void *  weights;    // one expert's projection weights (host)
    const float * activation; // ne00 floats
    float *       out;        // ne01 floats
};

// Miss-path counters, accumulated by moe_cpu_execute_gemvs into a caller-
// owned block. Per-device rather than global: the SYCL metrics collector
// emits one object per device, and a process-global counter repeated in
// each one double-counts as soon as Prometheus sum()s over the label.
//
// The ns breakdown costs two clock reads per weight row and is recorded
// only when GGML_MOE_HYBRID_PROFILE=1; the row/byte/thread counters are
// one atomic add per call and are always on. profiled_rows says how many
// rows the ns figures actually cover, so a mixed run cannot be read as if
// the whole of it were timed.
struct moe_cpu_tier_counters {
    std::atomic<int64_t> calls{0};          // moe_cpu_execute_gemvs calls
    std::atomic<int64_t> jobs{0};           // (expert, activation) pairs
    std::atomic<int64_t> weight_rows{0};    // output rows computed
    // Of those, the ones ggml-cpu's quantized kernels computed. Equal to
    // weight_rows on a build that links them and a type they cover; zero
    // says every row went the dequantize-and-dot way, which is roughly
    // 25x slower and is otherwise invisible from outside.
    std::atomic<int64_t> kernel_rows{0};
    std::atomic<int64_t> weight_bytes{0};   // quantized weight bytes read
    std::atomic<int64_t> threads_used{0};   // high-water worker count
    std::atomic<int64_t> wall_ns{0};        // in-call wall time
    std::atomic<int64_t> dispatch_ns{0};    // getting workers running
    std::atomic<int64_t> dequant_ns{0};     // weight dequantization
    std::atomic<int64_t> matmul_ns{0};      // the dot products
    std::atomic<int64_t> quant_act_ns{0};   // activation quantization
    std::atomic<int64_t> profiled_rows{0};  // rows the ns figures cover
};

// Plain snapshot of the above, for callers that only read.
struct moe_cpu_tier_stats {
    int64_t calls;
    int64_t jobs;
    int64_t weight_rows;
    int64_t kernel_rows;
    int64_t weight_bytes;
    int64_t threads_used;
    int64_t wall_ns;
    int64_t dispatch_ns;
    int64_t dequant_ns;
    int64_t matmul_ns;
    int64_t quant_act_ns;
    int64_t profiled_rows;
};
moe_cpu_tier_stats moe_cpu_tier_snapshot(const moe_cpu_tier_counters & c);

// Is the per-row ns breakdown being recorded? Read once from
// GGML_MOE_HYBRID_PROFILE at first call.
bool moe_cpu_profile_enabled();

// Workers the tier can put on one batch, counting the caller. This is
// the worker pool's own size -- it already leaves cores for the SYCL
// runtime and the server, and already honours GGML_MOE_HYBRID_THREADS --
// so callers should ask rather than compute their own limit.
int moe_cpu_tier_max_threads();

// stats may be null, in which case nothing is counted.
bool moe_cpu_execute_gemvs(
    const moe_cpu_gemv_job * jobs,
    size_t        n_jobs,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    int           n_threads,
    moe_cpu_tier_counters * stats = nullptr);

// Hybrid metrics for Prometheus export.
struct moe_hybrid_metrics {
    int64_t hit_rows = 0;
    int64_t miss_rows = 0;
    double  cpu_miss_time_ms = 0.0;
    double  gpu_hit_time_ms = 0.0;
    double  merge_time_ms = 0.0;
    int64_t promotion_bytes = 0;
    int64_t h2d_bytes_avoided = 0;
};

#endif // GGML_SYCL_MOE_HYBRID_HPP
