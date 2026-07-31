// Hybrid MoE Execution — partition representation and CPU miss
// execution.
//
// No SYCL here: partition, merge and the CPU tier are plain C++ (plus
// ggml-base for the quant dequantizers), which is what lets
// tests/test-moe-hybrid.cpp run without a GPU. GPU hit dispatch and the
// in-op hybrid integration live in ggml-sycl.cpp; promotion-delay and
// eviction-before-reuse counters are not recorded.

#include "moe-hybrid.hpp"

#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <thread>
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

bool moe_cpu_can_execute(int wtype) {
    if (wtype < 0 || wtype >= GGML_TYPE_COUNT) {
        return false;
    }
    const enum ggml_type type = (enum ggml_type) wtype;
    if (type == GGML_TYPE_F32) {
        return true;
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits && traits->to_float;
}

bool moe_cpu_expert_gemv(
    const void *  expert_weights,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    const float * activation,
    float *       out)
{
    if (!expert_weights || !activation || !out || ne00 <= 0 || ne01 <= 0 ||
        !moe_cpu_can_execute(wtype)) {
        return false;
    }
    const enum ggml_type type = (enum ggml_type) wtype;
    const bool is_f32 = (type == GGML_TYPE_F32);
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    const size_t row_bytes = ggml_row_size(type, ne00);

    std::vector<float> row_f32;
    if (!is_f32) {
        row_f32.resize((size_t) ne00);
    }
    for (int64_t o = 0; o < ne01; o++) {
        const void * wrow = (const uint8_t *) expert_weights + (size_t) o * row_bytes;
        const float * frow;
        if (is_f32) {
            frow = (const float *) wrow;
        } else {
            traits->to_float(wrow, row_f32.data(), ne00);
            frow = row_f32.data();
        }
        double acc = 0.0;
        for (int64_t j = 0; j < ne00; j++) {
            acc += (double) frow[j] * (double) activation[j];
        }
        out[o] = (float) acc;
    }
    return true;
}

bool moe_cpu_execute_gemvs(
    const moe_cpu_gemv_job * jobs,
    size_t        n_jobs,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    int           n_threads)
{
    if (!jobs || n_jobs == 0 || !moe_cpu_can_execute(wtype)) {
        return n_jobs == 0 && moe_cpu_can_execute(wtype);
    }
    for (size_t k = 0; k < n_jobs; k++) {
        if (!jobs[k].weights || !jobs[k].activation || !jobs[k].out) {
            return false;
        }
    }

    const int workers = std::max(1, std::min<int>(n_threads, (int) n_jobs));
    if (workers == 1) {
        for (size_t k = 0; k < n_jobs; k++) {
            if (!moe_cpu_expert_gemv(jobs[k].weights, wtype, ne00, ne01,
                                     jobs[k].activation, jobs[k].out)) {
                return false;
            }
        }
        return true;
    }

    // Contiguous chunks: each output row has exactly one writer, so the
    // result is identical to sequential execution regardless of how the
    // chunks are scheduled.
    std::vector<std::thread> pool;
    pool.reserve((size_t) workers - 1);
    std::atomic<bool> ok{true};
    auto worker = [&](size_t begin, size_t end) {
        for (size_t k = begin; k < end && ok.load(std::memory_order_relaxed); k++) {
            if (!moe_cpu_expert_gemv(jobs[k].weights, wtype, ne00, ne01,
                                     jobs[k].activation, jobs[k].out)) {
                ok.store(false, std::memory_order_relaxed);
            }
        }
    };
    const size_t chunk = (n_jobs + (size_t) workers - 1) / (size_t) workers;
    for (int w = 1; w < workers; w++) {
        const size_t begin = (size_t) w * chunk;
        if (begin >= n_jobs) break;
        pool.emplace_back(worker, begin, std::min(n_jobs, begin + chunk));
    }
    worker(0, std::min(n_jobs, chunk));
    for (auto & t : pool) {
        t.join();
    }
    return ok.load();
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

    if (!moe_cpu_can_execute(wtype)) {
        // No dequantizer for this type: the caller fails closed to the
        // non-hybrid path. Guessing here is defect 4 of the old scaffold
        // (quant blocks reinterpreted as floats) all over again.
        return -1;
    }

    // moe_cpu_expert_gemv dequantizes one row at a time: the working set
    // stays one row (ne00 floats) regardless of expert count, and every
    // quantized row is read exactly once -- for a cold mmap expert the
    // page faults on that read ARE the cost.
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
        if (moe_cpu_expert_gemv(expert, wtype, ne00, ne01, act, out)) {
            computed++;
        }
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
    // float addition is not associative.
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
