// Hybrid MoE Execution — partition representation (roadmap Task G2) and
// CPU miss execution (Task G3).
//
// No SYCL here: partition, merge and the CPU tier are plain C++ (plus
// ggml-base for the quant dequantizers), which is what lets
// tests/test-moe-hybrid.cpp run without a GPU. GPU hit dispatch (G4),
// the in-op merge (G5) and async promotion (G6) are not wired; see
// modelctl/docs/moe-hybrid-execution-design.md §4.

#include "moe-hybrid.hpp"

#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <vector>

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

int64_t moe_cpu_execute_misses(
    const moe_hybrid_partition & partition,
    const void *  weights_base,
    size_t        expert_stride_bytes,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    const float * activations,
    size_t        act_stride_floats,
    float *       outputs)
{
    if (!weights_base || !activations || !outputs ||
        ne00 <= 0 || ne01 <= 0 || expert_stride_bytes == 0) {
        return -1;
    }
    const enum ggml_type type = (enum ggml_type) wtype;
    if (wtype < 0 || wtype >= GGML_TYPE_COUNT) {
        return -1;
    }

    const bool is_f32 = (type == GGML_TYPE_F32);
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!is_f32 && (!traits || !traits->to_float)) {
        // No dequantizer for this type: the caller fails closed to the
        // non-hybrid path. Guessing here is defect 4 of the old scaffold
        // (quant blocks reinterpreted as floats) all over again.
        return -1;
    }
    const size_t row_bytes = ggml_row_size(type, ne00);

    // One dequantized row at a time: the working set stays one row
    // (ne00 floats) regardless of expert count, and the quantized row is
    // read exactly once -- for a cold mmap expert the page faults on that
    // read ARE the cost, and nothing here reads a weight byte twice.
    std::vector<float> row_f32;
    if (!is_f32) {
        row_f32.resize((size_t)ne00);
    }

    int64_t computed = 0;
    for (size_t k = 0; k < partition.contributions.size(); k++) {
        const moe_contribution & c = partition.contributions[k];
        if (c.tier != MOE_TIER_CPU_MISS) {
            continue;
        }
        if (c.expert_id < 0 || c.contiguous_row < 0) {
            continue;
        }
        const uint8_t * expert = (const uint8_t *) weights_base
                               + (size_t) c.expert_id * expert_stride_bytes;
        const float * act = activations
                          + (size_t) c.contiguous_row * act_stride_floats;
        float * out = outputs + k * (size_t) ne01;

        for (int64_t o = 0; o < ne01; o++) {
            const void * wrow = expert + (size_t) o * row_bytes;
            const float * frow;
            if (is_f32) {
                frow = (const float *) wrow;
            } else {
                traits->to_float(wrow, row_f32.data(), ne00);
                frow = row_f32.data();
            }
            // Plain f32 dot; accumulate in double for a batch-1 row so
            // the CPU tier does not drift from the GPU tier's f32
            // accumulation more than the quantization already does.
            double acc = 0.0;
            for (int64_t j = 0; j < ne00; j++) {
                acc += (double) frow[j] * (double) act[j];
            }
            out[o] = (float) acc;
        }
        computed++;
    }
    return computed;
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
