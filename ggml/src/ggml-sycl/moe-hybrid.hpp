// Hybrid MoE Execution — partition representation (roadmap Task G2).
//
// A routed MoE output is the weighted sum of n_expert_used expert outputs
// per token. The partition therefore is NOT a two-way split of rows: it is
// a list of *contributions*, each one expert's share of one token's
// output, tagged with the tier that will compute it.
//
// The earlier version of this file modelled it as a row split and merged
// with memcpy into dst[original_row], so with n_expert_used = 2 the second
// expert silently overwrote the first, and the routing coefficient was
// never represented at all. See modelctl/docs/moe-hybrid-execution-design.md
// §0 for that and the other defects this replaces.
//
// This header is deliberately free of SYCL, like moe-cache.hpp, so the
// partition logic can be unit-tested on any machine (Task F8's pattern).

#ifndef GGML_SYCL_MOE_HYBRID_HPP
#define GGML_SYCL_MOE_HYBRID_HPP

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
    // weighted sum; omitting this made the previous merge wrong
    // independently of the overwrite bug.
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
// not depend on scheduling (design §3.6).
//
// Set zero_dst unless the caller has already zeroed dst; this accumulates.
void moe_merge_contributions(
    const moe_hybrid_partition & partition,
    const float * tier_outputs,
    float *       dst,
    int64_t       ne0,
    bool          zero_dst);

// Hybrid metrics for Prometheus export (Task G6).
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
