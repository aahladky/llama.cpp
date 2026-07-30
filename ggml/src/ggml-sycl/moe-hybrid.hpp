// Hybrid MoE Execution — partition builder and dispatch.
//
// Splits expert computation into GPU cache hits and CPU misses.
// GPU hits use persistent cached expert tensors.
// CPU misses use mmap-backed host expert weights.
// Outputs are merged per-token.

#ifndef GGML_SYCL_MOE_HYBRID_HPP
#define GGML_SYCL_MOE_HYBRID_HPP

#include <cstdint>
#include <vector>
#include "moe-cache.hpp"

// Partition of (token_row, expert) pairs into GPU hits and CPU misses.
struct moe_hybrid_partition {
    // Indices into the contiguous routed-row buffer.
    // These are NOT original token indices — they're offsets into the
    // reordered src1/dst buffers produced by mmid_counting_sort_rows.
    struct row_entry {
        int32_t contiguous_row;  // index into reordered src1/dst
        int32_t original_row;    // index into original src1/dst
        int32_t expert_id;
    };

    std::vector<row_entry> gpu_hits;   // experts in cache -> GPU compute
    std::vector<row_entry> cpu_misses; // experts not in cache -> CPU compute

    bool has_hits() const { return !gpu_hits.empty(); }
    bool has_misses() const { return !cpu_misses.empty(); }
    bool is_all_hits() const { return cpu_misses.empty(); }
    bool is_all_misses() const { return gpu_hits.empty(); }
    int total_rows() const { return (int)(gpu_hits.size() + cpu_misses.size()); }
};

// Build a partition by checking which experts are in the GPU cache.
// expert_ids[i] is the expert assigned to contiguous row i.
// Returns a partition with rows classified as hits or misses.
moe_hybrid_partition moe_build_partition(
    const int32_t * expert_ids,
    int64_t n_rows,
    int32_t layer_id,
    moe_expert_cache * cache);

// Hybrid execution context: holds CPU buffers for miss computation.
struct moe_hybrid_context {
    // CPU-side buffers for miss rows.
    std::vector<float> cpu_src1;     // [n_miss_rows * ne10]  activations
    std::vector<float> cpu_dst;      // [n_miss_rows * ne0]   outputs
    std::vector<float> cpu_expert;   // [expert_bytes/4]      workspace

    bool initialized = false;

    void init(int64_t ne10, int64_t ne0, int max_miss_rows, size_t expert_bytes);
};

// Execute CPU miss path: compute expert outputs on CPU using
// host-resident (mmap-backed) expert weights.
// src0_host: pointer to the expert weight tensor in host memory
//            (indexed by expert_id * expert_stride)
// src1_host: pointer to the contiguous activation rows on host
// dst_host: pointer to the contiguous output rows on host
// partition: the partition (only cpu_misses are used)
// expert_stride: bytes between consecutive expert weights
// ne0, ne10: output and input dimensions
void moe_cpu_miss_execute(
    const void * src0_host,
    const float * src1_host,
    float * dst_host,
    const moe_hybrid_partition & partition,
    size_t expert_stride,
    int64_t ne0,
    int64_t ne10,
    int32_t n_experts);

// Merge GPU hit outputs and CPU miss outputs into the final dst buffer.
// gpu_dst: GPU buffer with computed hit rows
// cpu_dst: CPU buffer with computed miss rows
// partition: the partition
// dst: final output buffer (GPU)
// ne0: output dimension (per row)
// row_bytes: bytes per output row
// stream: SYCL queue for the final copy
void moe_merge_outputs(
    const float * gpu_dst,
    const float * cpu_dst,
    const moe_hybrid_partition & partition,
    float * dst,
    int64_t ne0,
    size_t row_bytes,
    void * stream);

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
