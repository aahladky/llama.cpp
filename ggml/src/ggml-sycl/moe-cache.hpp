// MoE Expert Cache — persistent GPU slot pool for routed expert weights.
//
// Each cache instance lives on one SYCL device.  It manages a fixed number
// of slots, each large enough to hold one routed expert's weights for one
// layer.  The cache is queried by (layer_id, expert_id) and returns a
// device pointer on hit or nullptr on miss.
//
// Policy: LRU with optional second-miss admission (SLRU in a later milestone).
//
// Thread safety: the cache is accessed from the SYCL compute thread only;
// no concurrent access is expected during a single forward pass.

#ifndef GGML_SYCL_MOE_CACHE_HPP
#define GGML_SYCL_MOE_CACHE_HPP

#include <cstdint>
#include <atomic>
#include <vector>
#include <string>
#include "common.hpp"

// Unique key for a cached expert.
struct moe_expert_key {
    int32_t layer  = -1;
    int32_t expert = -1;

    bool valid() const { return layer >= 0 && expert >= 0; }
    bool operator==(const moe_expert_key & o) const {
        return layer == o.layer && expert == o.expert;
    }
};

// One cache slot: holds device memory for a single expert's tensors.
struct moe_cache_slot {
    moe_expert_key key;
    bool occupied       = false;
    bool protected_seg  = false;  // SLRU protected segment (future)
    uint64_t last_access = 0;
    uint64_t access_count = 0;
    void * device_ptr   = nullptr;  // SYCL USM allocation
    size_t bytes        = 0;

    // Per-tensor sub-allocations within the slot (gate, up, down).
    // Stored as offsets from device_ptr.
    size_t gate_offset  = 0;
    size_t up_offset    = 0;
    size_t down_offset  = 0;
    size_t gate_bytes   = 0;
    size_t up_bytes     = 0;
    size_t down_bytes   = 0;
};

// Per-layer index: maps expert_id -> slot_id, or -1 if not cached.
struct moe_cache_layer_index {
    std::vector<int32_t> expert_to_slot;  // [n_experts_per_layer]

    void init(int n_experts) {
        expert_to_slot.assign(n_experts, -1);
    }
};

// Atomic counters for cache metrics.
struct moe_cache_stats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> promotions{0};
    std::atomic<uint64_t> h2d_bytes{0};
    std::atomic<uint64_t> cpu_expert_calls{0};
    std::atomic<uint64_t> gpu_expert_calls{0};
    std::atomic<uint64_t> sync_wait_ns{0};

    void reset() {
        hits = misses = evictions = promotions = 0;
        h2d_bytes = cpu_expert_calls = gpu_expert_calls = sync_wait_ns = 0;
    }
};

// Configuration for cache creation.
struct moe_cache_config {
    int device_id       = 0;
    size_t budget_bytes = 0;
    int n_layers        = 0;
    int n_experts       = 0;
    size_t expert_bytes = 0;      // total bytes per expert
    size_t gate_bytes   = 0;
    size_t up_bytes     = 0;
    size_t down_bytes   = 0;
    std::string policy  = "lru";
    int admission_misses = 1;
    bool prefill_admit   = false;
};

// The cache itself.
class moe_expert_cache {
public:
    moe_expert_cache() = default;
    ~moe_expert_cache();

    bool init(const moe_cache_config & cfg, queue_ptr queue);

    // Look up (layer, expert).  Returns slot pointer on hit, nullptr on miss.
    void * lookup(int32_t layer, int32_t expert);

    // Record a miss for admission tracking.
    void record_miss(int32_t layer, int32_t expert);

    // Promote a missed expert into a cache slot.
    void * promote(int32_t layer, int32_t expert,
                   const void * host_src_gate,
                   const void * host_src_up,
                   const void * host_src_down);

    void reset();

    const moe_cache_stats & stats() const { return m_stats; }
    int slot_count() const { return (int)m_slots.size(); }
    int slots_used() const;
    size_t budget_bytes() const { return m_budget_bytes; }
    bool is_initialized() const { return m_initialized; }
    std::string stats_json() const;

private:
    int evict_lru();
    void touch(int slot_id);

    queue_ptr m_queue = nullptr;
    bool m_initialized = false;
    size_t m_budget_bytes = 0;
    int m_n_layers = 0;
    int m_n_experts = 0;
    size_t m_expert_bytes = 0;
    int m_admission_misses = 1;
    uint64_t m_tick = 0;

    std::vector<moe_cache_slot> m_slots;
    std::vector<moe_cache_layer_index> m_layer_index;
    std::vector<int> m_miss_counts;
    moe_cache_stats m_stats;
};

#endif // GGML_SYCL_MOE_CACHE_HPP
