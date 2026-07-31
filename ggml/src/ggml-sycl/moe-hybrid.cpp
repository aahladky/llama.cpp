// Hybrid MoE Execution — partition representation (roadmap Task G2).
//
// No SYCL here: this is the partition and merge logic, and keeping it
// plain C++ is what lets tests/test-moe-hybrid.cpp run without a GPU.
// Execution (G3/G4) and the real merge into device buffers (G5) are not
// implemented; see modelctl/docs/moe-hybrid-execution-design.md §4.

#include "moe-hybrid.hpp"

#include <algorithm>
#include <cstring>

int moe_hybrid_partition::n_destination_rows() const {
    if (contributions.empty()) {
        return 0;
    }
    int32_t max_row = -1;
    for (const auto & c : contributions) {
        max_row = std::max(max_row, c.original_row);
    }
    if (max_row < 0) {
        return 0;
    }
    // Count distinct rows rather than assuming they are dense: a partition
    // built for one projection of a sparse batch need not cover every row.
    std::vector<bool> seen((size_t)max_row + 1, false);
    int n = 0;
    for (const auto & c : contributions) {
        if (c.original_row >= 0 && !seen[(size_t)c.original_row]) {
            seen[(size_t)c.original_row] = true;
            n++;
        }
    }
    return n;
}

moe_hybrid_partition moe_build_partition(
    const int32_t * expert_ids,
    const float *   routing_weights,
    const int32_t * original_rows,
    int64_t         n_rows,
    int32_t         layer_id,
    int             projection,
    const moe_expert_cache * cache)
{
    moe_hybrid_partition part;
    if (!expert_ids || n_rows <= 0) {
        return part;
    }
    part.contributions.reserve((size_t)n_rows);

    for (int64_t i = 0; i < n_rows; i++) {
        moe_contribution c;
        c.contiguous_row = (int32_t)i;
        c.original_row   = original_rows ? original_rows[i] : (int32_t)i;
        c.expert_id      = expert_ids[i];
        c.routing_weight = routing_weights ? routing_weights[i] : 1.0f;
        c.projection     = projection;

        // No cache, or the projection is not resident, means the CPU tier
        // computes it. contains() is used rather than lookup() so that
        // building a partition does not count hits, move SLRU recency or
        // clear admission progress -- classifying work must not change
        // what is cached.
        const bool resident = cache && cache->contains(layer_id, c.expert_id,
                                                       projection);
        c.tier = resident ? MOE_TIER_GPU_HIT : MOE_TIER_CPU_MISS;
        part.contributions.push_back(c);
    }
    return part;
}

void moe_merge_contributions(
    const moe_hybrid_partition & partition,
    const float * tier_outputs,
    float *       dst,
    int64_t       ne0,
    bool          zero_dst)
{
    if (!dst || ne0 <= 0) {
        return;
    }
    if (zero_dst) {
        const int rows = partition.n_destination_rows();
        if (rows > 0) {
            // Rows are addressed by original_row, so the span to clear is
            // bounded by the largest one, not by the count.
            int32_t max_row = -1;
            for (const auto & c : partition.contributions) {
                max_row = std::max(max_row, c.original_row);
            }
            if (max_row >= 0) {
                std::memset(dst, 0, (size_t)(max_row + 1) * (size_t)ne0
                            * sizeof(float));
            }
        }
    }
    if (!tier_outputs) {
        return;
    }

    // Fixed order: the partition's own. Accumulating in completion order
    // would make the same input produce different output run to run, since
    // float addition is not associative (design §3.6).
    for (size_t k = 0; k < partition.contributions.size(); k++) {
        const moe_contribution & c = partition.contributions[k];
        if (c.original_row < 0) {
            continue;
        }
        const float * src = tier_outputs + k * (size_t)ne0;
        float * out = dst + (size_t)c.original_row * (size_t)ne0;
        const float w = c.routing_weight;
        for (int64_t j = 0; j < ne0; j++) {
            out[j] += w * src[j];
        }
    }
}
