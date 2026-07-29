// MoE Expert Cache — persistent GPU slot pool for routed expert weights.
//
// Each cache instance lives on one SYCL device.  It manages a fixed number
// of slots, each large enough to hold one routed expert's weights for one
// layer.  The cache is queried by (layer_id, expert_id) and returns a
// device pointer on hit or nullptr on miss.
//
// Policy: SLRU (probationary + protected segments) with second-miss
// admission and prefill protection.  Falls back to plain LRU when the
// policy is "lru".
//
// SLRU segments:
//   probationary (20%): new entries; one-off experts don't evict hot ones.
//   protected (80%): promoted from probationary on second access; only
//     evicted when the protected segment is full (demoted to probationary
//     rather than freed).
//
// Prefill protection: during prefill (set_phase(true)), miss counts are
// NOT incremented, so long prompts don't flood the cache with one-use
// experts.  Decode (set_phase(false)) resumes normal admission.
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
// Supports partial fill: each projection (gate/up/down) is tracked separately
// and the slot is only fully usable once all three are present.
struct moe_cache_slot {
    moe_expert_key key;
    bool occupied       = false;
    bool protected_seg  = false;  // SLRU protected segment
    uint8_t filled_mask = 0;      // bit0=gate, bit1=up, bit2=down
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

    bool is_full() const { return filled_mask == 0x7; }  // gate+up+down
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
    std::atomic<uint64_t> prefill_misses{0};
    std::atomic<uint64_t> decode_misses{0};
    std::atomic<uint64_t> prefill_hits{0};
    std::atomic<uint64_t> decode_hits{0};
    std::atomic<uint64_t> slru_promotions{0};    // probationary -> protected
    std::atomic<uint64_t> slru_demotions{0};      // protected -> probationary

    void reset() {
        hits = misses = evictions = promotions = 0;
        h2d_bytes = cpu_expert_calls = gpu_expert_calls = sync_wait_ns = 0;
        prefill_misses = decode_misses = prefill_hits = decode_hits = 0;
        slru_promotions = slru_demotions = 0;
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

    // Look up (layer, expert) for a specific projection (0=gate, 1=up, 2=down).
    // Returns slot pointer on hit, nullptr on miss.
    void * lookup(int32_t layer, int32_t expert, int projection);

    // Record a miss for admission tracking.  During prefill, miss counts
    // are NOT incremented (prefill protection).
    void record_miss(int32_t layer, int32_t expert);

    // Promote one projection of an expert into the cache.
    // If the expert isn't cached yet, allocates a slot and copies this
    // projection.  If the expert is cached but this projection is missing,
    // fills it.  Returns the slot pointer.
    void * promote_projection(int32_t layer, int32_t expert,
                               int projection, const void * host_src);

    // Set inference phase: true=prefill, false=decode.
    // During prefill, miss counts are not incremented.
    void set_phase(bool is_prefill) { m_is_prefill = is_prefill; }

    void reset();

    const moe_cache_stats & stats() const { return m_stats; }
    int slot_count() const { return (int)m_slots.size(); }
    int slots_used() const;
    size_t budget_bytes() const { return m_budget_bytes; }
    bool is_initialized() const { return m_initialized; }
    std::string stats_json() const;

private:
    int  evict_lru();
    int  evict_slru();
    void touch(int slot_id);
    void promote_to_protected(int slot_id);

    queue_ptr m_queue = nullptr;
    bool m_initialized = false;
    size_t m_budget_bytes = 0;
    int m_n_layers = 0;
    int m_n_experts = 0;
    size_t m_expert_bytes = 0;
    int m_admission_misses = 1;
    bool m_prefill_admit = false;
    bool m_is_prefill = false;
    bool m_use_slru = false;
    int m_protected_limit = 0;   // max protected-segment slots
    uint64_t m_tick = 0;

    std::vector<moe_cache_slot> m_slots;
    std::vector<moe_cache_layer_index> m_layer_index;
    std::vector<int> m_miss_counts;
    moe_cache_stats m_stats;
};

#endif // GGML_SYCL_MOE_CACHE_HPP
