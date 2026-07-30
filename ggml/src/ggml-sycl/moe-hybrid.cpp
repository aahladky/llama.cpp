// Hybrid MoE Execution — implementation.

#include "moe-hybrid.hpp"
#include <cstring>
#include <chrono>

moe_hybrid_partition moe_build_partition(
    const int32_t * expert_ids,
    int64_t n_rows,
    int32_t layer_id,
    moe_expert_cache * cache)
{
    moe_hybrid_partition part;
    part.gpu_hits.reserve(n_rows);
    part.cpu_misses.reserve(n_rows);

    if (!cache || !cache->is_initialized()) {
        // No cache: everything is a miss.
        for (int64_t i = 0; i < n_rows; i++) {
            part.cpu_misses.push_back({(int32_t)i, (int32_t)i, expert_ids[i]});
        }
        return part;
    }

    for (int64_t i = 0; i < n_rows; i++) {
        int32_t expert = expert_ids[i];
        // Check if this expert is fully cached (gate+up+down).
        // We check gate projection (projection 0) as a proxy — if gate
        // is cached, the slot exists and will be filled for other projections.
        void * cached = cache->lookup(layer_id, expert, /*projection=*/0, /*proj_bytes=*/0);
        if (cached) {
            part.gpu_hits.push_back({(int32_t)i, (int32_t)i, expert});
        } else {
            part.cpu_misses.push_back({(int32_t)i, (int32_t)i, expert});
        }
    }

    return part;
}

void moe_hybrid_context::init(int64_t ne10, int64_t ne0, int max_miss_rows,
                              size_t expert_bytes) {
    cpu_src1.resize(max_miss_rows * ne10);
    cpu_dst.resize(max_miss_rows * ne0);
    cpu_expert.resize(expert_bytes / sizeof(float));
    initialized = true;
}

void moe_cpu_miss_execute(
    const void * src0_host,
    const float * src1_host,
    float * dst_host,
    const moe_hybrid_partition & partition,
    size_t expert_stride,
    int64_t ne0,
    int64_t ne10,
    int32_t n_experts)
{
    if (!partition.has_misses()) return;

    for (const auto & entry : partition.cpu_misses) {
        int32_t expert = entry.expert_id;
        if (expert < 0 || expert >= n_experts) continue;

        const float * expert_weights = (const float *)(
            (const char *)src0_host + expert * expert_stride);
        const float * activation = src1_host + entry.contiguous_row * ne10;
        float * output = dst_host + entry.contiguous_row * ne0;

        // Simple CPU matmul: output = activation @ expert_weights^T
        // expert_weights is [ne0 x ne10] (row-major)
        // activation is [ne10]
        // output is [ne0]
        for (int64_t o = 0; o < ne0; o++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < ne10; i++) {
                sum += activation[i] * expert_weights[o * ne10 + i];
            }
            output[o] = sum;
        }
    }
}

void moe_merge_outputs(
    const float * gpu_dst,
    const float * cpu_dst,
    const moe_hybrid_partition & partition,
    float * dst,
    int64_t ne0,
    size_t row_bytes,
    void * stream)
{
    // Copy GPU hit results to final dst.
    for (const auto & entry : partition.gpu_hits) {
        // gpu_dst is indexed by contiguous row within the hit subset.
        // We need to find which index in gpu_hits this entry is.
        // For simplicity, we use the contiguous_row as offset into gpu_dst.
        const float * src = gpu_dst + entry.contiguous_row * ne0;
        float * d = dst + entry.original_row * ne0;
        memcpy(d, src, row_bytes);
    }

    // Copy CPU miss results to final dst.
    for (const auto & entry : partition.cpu_misses) {
        const float * src = cpu_dst + entry.contiguous_row * ne0;
        float * d = dst + entry.original_row * ne0;
        memcpy(d, src, row_bytes);
    }
}
