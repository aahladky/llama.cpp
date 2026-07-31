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
// Thread safety: the compute path (lookup/record_miss/promote) runs on the
// SYCL compute thread, while reset/stats/phase can be invoked from the
// server HTTP threads.  All shared state is protected by m_mutex.

#ifndef GGML_SYCL_MOE_CACHE_HPP
#define GGML_SYCL_MOE_CACHE_HPP

// Deliberately free of SYCL headers.  The only device concern this header
// ever had was the queue type, and including ggml-sycl/common.hpp to get it
// dragged the SYCL headers and the ggml-sycl target's private compile
// definitions into every consumer -- which is what kept
// tests/test-moe-cache.cpp out of the build.  The queue is opaque here and
// cast back to queue_ptr in moe-cache.cpp, the one place that talks to the
// device.  ctx.stream() converts implicitly, so callers are unaffected.
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

// Projections held per expert: gate, up, down.
//
// Fused layouts: architectures with a combined gate+up tensor
// ("ffn_gate_up_exps") map it to projection 0; projection 1 then simply
// never occurs for that model. The cache learns which projections exist
// and how large each one is by OBSERVING real staged copies (see
// observe_geometry) instead of assuming three equal components.
#define MOE_CACHE_N_PROJECTIONS 3

// Parse the layer index from a routed-expert tensor name like
// "blk.5.ffn_gate_exps". Returns -1 for anything that is not a routed
// MoE expert tensor (note: shared-expert tensors are named "*_shexp",
// which does not contain "exps" and is correctly rejected).
int moe_cache_parse_layer(const char * name);

// Projection index from a tensor name: 0=gate (also fused gate_up),
// 1=up, 2=down, -1=unknown. Lives here (not in the SYCL TU) so the
// host-only tests can pin the naming rules per architecture.
int moe_cache_parse_projection(const char * name);

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

    // The host source each filled projection was promoted from. Two
    // models loaded on one device share this cache and use identical
    // tensor names ("blk.5.ffn_gate_exps"), so (layer, expert) alone
    // would let model B's lookup be served model A's weights. The host
    // pointer is the tensor identity that names cannot provide.
    const void * proj_origin[MOE_CACHE_N_PROJECTIONS] = {nullptr, nullptr, nullptr};

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
    // Projection copies the cache declined to serve, so the weights went
    // host->device as they would with no cache at all.  This was named
    // cpu_expert_calls, which described CPU expert *execution* -- something
    // no shipping backend implements (that is Phase G).  Nothing computed
    // on the CPU here; the copy simply bypassed the cache.
    std::atomic<uint64_t> host_weight_copy_fallbacks{0};
    // Projection lookups served from a cache slot.  Counted per projection,
    // not per expert: one expert use touches gate, up and down separately.
    std::atomic<uint64_t> cache_served_projections{0};
    std::atomic<uint64_t> prefill_misses{0};
    std::atomic<uint64_t> decode_misses{0};
    std::atomic<uint64_t> prefill_hits{0};
    std::atomic<uint64_t> decode_hits{0};
    std::atomic<uint64_t> slru_promotions{0};    // probationary -> protected
    std::atomic<uint64_t> slru_demotions{0};      // protected -> probationary

    void reset() {
        hits = misses = evictions = promotions = 0;
        h2d_bytes = host_weight_copy_fallbacks = cache_served_projections = 0;
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
    // Host-only mode for unit tests: slots are allocated with malloc and
    // no device copies are issued, so the admission, eviction, phase and
    // reset logic can be exercised on a machine with no GPU. Requires a
    // null queue; a non-null queue always takes the real device path, so
    // this cannot be reached by accident in production.
    bool host_only_for_testing = false;
};

// Device operations, defined in moe-cache-device.cpp -- the only translation
// unit here that includes the SYCL headers.  Splitting them out is what lets
// the policy implementation (admission, eviction, phase, reset) compile as
// plain C++, and so lets tests/test-moe-cache.cpp build without a GPU or the
// ggml-sycl target's private compile definitions.
//
// `queue` is a sycl::queue *.  All three report failure by return value
// rather than throwing, so the policy code needs no SYCL exception handling.
void * moe_cache_device_alloc(size_t bytes, void * queue);
void   moe_cache_device_free(void * ptr, void * queue);
bool   moe_cache_device_copy(void * dst, const void * src, size_t bytes, void * queue);

// The cache itself.
class moe_expert_cache {
public:
    moe_expert_cache() = default;
    ~moe_expert_cache();

    // queue is a sycl::queue * (queue_ptr); see the note on the includes.
    // Null is only accepted together with cfg.host_only_for_testing.
    //
    // Queue ownership contract: the queue passed here must be the DEVICE's
    // queue (in practice the dpct per-device default in-order queue, which
    // every ggml-sycl context's stream() resolves to). It is owned by the
    // device manager, not by any backend context, so it outlives every
    // context that shares this cache; and because it is in-order, promote
    // copy-ins and lookup copy-outs stay serialized no matter which
    // context's compute thread submitted them. All cache device work --
    // pool alloc, pool free, promote copies -- goes through THIS queue;
    // callers must not submit cache transfers on a queue of their own.
    bool init(const moe_cache_config & cfg, void * queue);

    // Deferred variant: accepts a config with NO projection geometry.
    // The cache starts in learning mode (is_learning() true); feed it
    // observe_geometry() calls until it reports ready, at which point the
    // pool is allocated from the OBSERVED per-projection sizes. This is
    // how fused gate_up layouts and unequal gate/up/down sizes get
    // correctly sized regions instead of first-tensor guesses.
    bool init_deferred(const moe_cache_config & cfg, void * queue);

    // Record one staged-copy observation while learning. `origin` is the
    // host base address of the staged tensor (host_src minus the expert
    // offset): revisiting an origin that was already staged and left
    // means the graph has wrapped into a second pass, so every projection
    // that exists has been seen and geometry can be finalized. Returns
    // true once the cache has just become (or already is) ready.
    //
    // If the scheduler stages one tensor for several splits in a single
    // pass, finalization can fire before the later projections appear;
    // those projections are then simply never cached (fail-safe, not
    // fail-wrong).
    bool observe_geometry(int32_t layer, int projection, size_t proj_bytes,
                          const void * origin);

    // True while in deferred mode with geometry not yet finalized.
    bool is_learning() const;

    // The device queue every cache submission uses (see the init contract).
    void * queue() const { return m_queue; }

    // Look up (layer, expert) for a specific projection (0=gate, 1=up, 2=down).
    // proj_bytes must match the learned projection size AND host_src must
    // match the pointer the projection was promoted from (tensor identity
    // -- two models on one device reuse the same names). Returns a device
    // pointer to the projection region within the slot on hit, nullptr on
    // miss. host_src=nullptr skips the identity check; the production hook
    // always passes it, the default exists for policy tests that have no
    // stable tensor addresses.
    void * lookup(int32_t layer, int32_t expert, int projection, size_t proj_bytes,
                  const void * host_src = nullptr);

    // Is this projection resident right now?  Task G2's partition builder
    // needs to ask without changing the answer: lookup() counts a hit or a
    // miss, updates SLRU recency and clears admission progress, all of
    // which are correct for the transfer-cache path and wrong for
    // classifying rows. This touches nothing.
    bool contains(int32_t layer, int32_t expert, int projection) const;

    // Record a miss for admission tracking.  During prefill, miss counts
    // are NOT incremented (prefill protection).
    //
    // Admission state is per (layer, expert, projection).  One use of an
    // expert misses on gate, up and down separately, so a counter shared
    // across the three reached an admission threshold of 2 within a single
    // use -- making "admit on the second use" behave like "admit on the
    // first".  Worse, a hit on an already-cached projection reset the
    // shared counter, so whether the remaining projections were ever
    // admitted depended on the order they happened to be visited in.
    void record_miss(int32_t layer, int32_t expert, int projection);

    // Promote one projection of an expert into the cache.
    // If the expert isn't cached yet, allocates a slot and copies this
    // projection.  If the expert is cached but this projection is missing,
    // fills it.  proj_bytes must match the learned projection size,
    // otherwise no copy is made.  The slot records host_src as the
    // projection's origin; lookups only serve requests with the same
    // origin.  Returns a device pointer to the projection region within
    // the slot.
    void * promote_projection(int32_t layer, int32_t expert,
                               int projection, const void * host_src, size_t proj_bytes);

    // Count one projection copy that bypassed the cache and went
    // host->device directly.
    void inc_host_weight_copy_fallback() { m_stats.host_weight_copy_fallbacks++; }

    // Set inference phase: true=prefill, false=decode.
    // During prefill, miss counts are not incremented.
    void set_phase(bool is_prefill);

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
    // Carve the slot pool from m_geom_bytes. Caller holds m_mutex (or is
    // in single-threaded init).
    bool finalize_geometry_locked();

    // Index into m_miss_counts for one (layer, expert, projection).
    // Returns -1 when any component is out of range.
    int  miss_index(int32_t layer, int32_t expert, int projection) const {
        if (layer < 0 || layer >= m_n_layers ||
            expert < 0 || expert >= m_n_experts ||
            projection < 0 || projection >= MOE_CACHE_N_PROJECTIONS) {
            return -1;
        }
        return (layer * m_n_experts + expert) * MOE_CACHE_N_PROJECTIONS + projection;
    }

    void * m_queue = nullptr;   // sycl::queue *, opaque here
    bool m_initialized = false;
    bool m_host_only = false;
    // Single contiguous allocation backing every slot. One reservation is
    // cheaper for the allocator than n_slots separate ones and cannot end
    // up fragmented across the device's address space; slots point into
    // it at fixed offsets.
    void * m_pool = nullptr;
    size_t m_pool_bytes = 0;
    size_t m_budget_bytes = 0;
    int m_n_layers = 0;
    int m_n_experts = 0;
    size_t m_expert_bytes = 0;
    // Learned (or configured) byte size per projection index. 0 means
    // "this projection does not exist for this model" (fused layouts have
    // no separate up), and requests against it are rejected.
    size_t m_geom_bytes[MOE_CACHE_N_PROJECTIONS] = {0, 0, 0};
    // Geometry-learning state (deferred init). m_obs_completed holds the
    // origins whose staging we have seen finish (we moved on to another
    // tensor); revisiting one of them means a second pass has begun.
    bool m_deferred = false;
    const void * m_obs_current_origin = nullptr;
    std::vector<const void *> m_obs_completed;
    int m_admission_misses = 1;
    bool m_prefill_admit = false;
    bool m_is_prefill = false;
    bool m_use_slru = false;
    int m_protected_limit = 0;   // max protected-segment slots
    uint64_t m_tick = 0;

    mutable std::mutex m_mutex;  // protects slots, layer index, miss counts, phase
    std::vector<moe_cache_slot> m_slots;
    std::vector<moe_cache_layer_index> m_layer_index;
    std::vector<int> m_miss_counts;
    moe_cache_stats m_stats;
};

#endif // GGML_SYCL_MOE_CACHE_HPP
