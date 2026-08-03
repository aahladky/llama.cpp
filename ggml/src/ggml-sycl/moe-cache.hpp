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

// Deliberately free of SYCL headers, so policy tests build without the
// SYCL toolchain: queue pointers are opaque here and cast back to
// queue_ptr in the device translation unit, the one place that talks to
// the device.  ctx.stream() converts implicitly, so callers are
// unaffected.
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <utility>
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

// Advice values passed to moe_cache_config::mmap_advise. Plain ints (not
// the ggml-backend constants) so this header stays free of ggml includes
// for the host-only tests; ggml-sycl.cpp static_asserts they match the
// GGML_MOE_MMAP_ADVISE_* values.
#define MOE_CACHE_MMAP_ADVISE_WILLNEED 1
#define MOE_CACHE_MMAP_ADVISE_DONTNEED 2
typedef void (*moe_cache_mmap_advise_fn)(const void * ptr, size_t len, int advice);

// Per-step advice batches are bounded; entries past the cap are dropped
// and counted (advise_dropped), never silently.
#define MOE_CACHE_ADVISE_MAX_RANGES 4096

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
    // Async-fill reservations: projections whose copy is queued but not
    // yet complete. Invisible to lookup(), immune to eviction.
    uint8_t pending_mask = 0;
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
    // host->device as they would with no cache at all.  Nothing computes
    // on the CPU for these; the copy simply bypassed the cache.
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
    // mmap-tier advice calls issued (WILLNEED ranges are coalesced first,
    // so willneed counts madvise calls, not miss events) and batch
    // entries dropped at the per-step cap.
    std::atomic<uint64_t> advise_willneed{0};
    std::atomic<uint64_t> advise_dontneed{0};
    std::atomic<uint64_t> advise_dropped{0};
    // Plans that survived a completed graph and were purged at step end
    // (each one is a leak of the class the C3 hang grew from).
    std::atomic<uint64_t> hybrid_stale_plans_purged{0};
    // Async admission fills (reserve at miss, copy at step end).
    std::atomic<uint64_t> async_fills_enqueued{0};
    std::atomic<uint64_t> async_fills_completed{0};
    std::atomic<uint64_t> async_fills_discarded{0};
    std::atomic<uint64_t> async_fill_bytes{0};

    void reset() {
        hits = misses = evictions = promotions = 0;
        h2d_bytes = host_weight_copy_fallbacks = cache_served_projections = 0;
        prefill_misses = decode_misses = prefill_hits = decode_hits = 0;
        slru_promotions = slru_demotions = 0;
        advise_willneed = advise_dontneed = advise_dropped = 0;
        hybrid_stale_plans_purged = 0;
        async_fills_enqueued = async_fills_completed = 0;
        async_fills_discarded = async_fill_bytes = 0;
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
    // SSD/mmap tier (P1): when non-null, the cache batches the host
    // ranges miss-path staged copies read (WILLNEED at step end -- likely
    // needed again next step) and the origins of evicted projections
    // (DONTNEED at step end -- cache pressure judged them cold).  The fn
    // must no-op for pointers outside live model mappings; llama-mmap's
    // bridge enforces that, which is what keeps a RAM/VRAM-resident model
    // untouched.  Production additionally gates wiring this behind
    // GGML_MOE_CACHE_MMAP_ADVISE=1.
    moe_cache_mmap_advise_fn mmap_advise = nullptr;
    // Async admission fills: promote_projection only RESERVES a slot; the
    // H2D copy runs on a dedicated transfer queue at step end, ordered
    // after the graph's compute by a barrier event, and the projection
    // stays invisible until the copy completes. Removes every fill from
    // the compute stream. Default off here; the backend turns it on
    // unless GGML_MOE_CACHE_ASYNC_FILL=0.
    bool async_fill = false;
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

// Async-fill support (moe-cache-device.cpp). The transfer queue is a
// second in-order queue on the compute queue's device+context. A fill
// submitted through moe_cache_device_copy_async_after runs on the
// transfer queue but only after everything already submitted on the
// compute queue (barrier dependency) -- so a refill of a reused slot can
// never overwrite bytes an in-flight kernel or D2D hit copy still reads.
// Returns an opaque event (freed with moe_cache_device_event_free), or
// null on submission failure.
void * moe_cache_device_create_transfer_queue(void * compute_queue);
void   moe_cache_device_destroy_transfer_queue(void * transfer_queue);
void * moe_cache_device_copy_async_after(void * dst, const void * src, size_t bytes,
                                         void * transfer_queue, void * compute_queue);
bool   moe_cache_device_event_complete(void * ev);
void   moe_cache_device_event_free(void * ev);

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

    // ---- hybrid staging plan ----------------------------------------
    // Under hybrid mode the scheduler hook SKIPS the host->device copy
    // for a miss expert the cache declined to admit -- avoiding that
    // transfer is the entire saving -- and records it here. The op that
    // consumes the staged tensor then computes those experts' rows on
    // CPU over the original host weights. The plan is keyed by the
    // STAGED tensor's device base address: it is what both sides can
    // see (the hook gets dst, the op gets src0->data), it is unique per
    // staged tensor, and it cannot collide across two models the way
    // tensor names do.

    struct hybrid_skip {
        int32_t      expert_id = -1;
        const void * host_src  = nullptr;  // this expert's weights on host
    };

    struct hybrid_plan {
        std::vector<hybrid_skip> skips;
        int          wtype = -1;           // ggml_type of the weights
        bool empty() const { return skips.empty(); }
        bool has_expert(int32_t eid) const {
            for (const auto & s : skips) {
                if (s.expert_id == eid) return true;
            }
            return false;
        }
        const void * host_src_for(int32_t eid) const {
            for (const auto & s : skips) {
                if (s.expert_id == eid) return s.host_src;
            }
            return nullptr;
        }
    };

    // Record one skipped staging copy for the tensor staged at cpy_base.
    void hybrid_record_skip(const void * cpy_base, int32_t expert_id,
                            const void * host_src, int wtype);

    // True when a plan is pending for this staged tensor (the op must
    // then avoid fused kernels that would read the unstaged regions).
    bool hybrid_plan_pending(const void * cpy_base) const;

    // Remove and return the plan for this staged tensor (empty plan when
    // none). The op takes it exactly once per execution; a stale plan
    // must never survive into the next graph run.
    hybrid_plan hybrid_take_plan(const void * cpy_base);

    // Drop every pending plan.  Called (via the scheduler's abandon hook)
    // when a graph is abandoned mid-compute: its ops will never take
    // their plans, and a later tensor reusing a staged base must not
    // inherit one and compute with the wrong host weights.
    void hybrid_purge_plans();

    // Purge plans that survived a COMPLETED graph. Any plan not taken by
    // its op by the time the graph finishes is leaked -- and because
    // hybrid_record_skip merges into an existing plan at the same staged
    // base, a leaked plan poisons the next graph that reuses the base
    // (stale skips join fresh ones and the CPU tier's row bookkeeping
    // waits on work that never exists: the C3 hang). Called from the
    // scheduler's step-end hook, the one point where no op can be
    // mid-dispatch. Returns the number purged; counts them in stats.
    size_t hybrid_purge_stale();

    // Async admission fills: retire completed copies (making their
    // projections servable) and enqueue this step's reservations on the
    // transfer queue. Called from the scheduler's step-end hook. No-op
    // unless cfg.async_fill was set.
    void async_fill_flush_step();

    // The staging hook refills this device base with raw GGUF-layout
    // bytes on every graph run.  opt_for_reorder_id must never in-place
    // reorder such a tensor: the reorder happens once but its flag is
    // sticky, so from the second run on a fused kernel would read raw
    // restaged bytes as if they were reordered.  The set is monotonic for
    // the cache's lifetime -- clearing it (e.g. on reset()) could race a
    // stage already in flight, and a false positive after address reuse
    // only costs the reorder optimization, never correctness.
    void note_staged_base(const void * cpy_base);
    bool is_staged_base(const void * cpy_base) const;

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

    // Is this projection resident right now?  The hybrid partition
    // builder needs to ask without changing the answer: lookup() counts a hit or a
    // miss, updates SLRU recency and clears admission progress, all of
    // which are correct for the transfer-cache path and wrong for
    // classifying rows. This touches nothing.
    bool contains(int32_t layer, int32_t expert, int projection) const;

    // Record a miss for admission tracking.  During prefill, miss counts
    // are NOT incremented (prefill protection).
    //
    // Admission state is per (layer, expert, projection).  One use of an
    // expert misses on gate, up and down separately, so a counter shared
    // across the three would reach an admission threshold of 2 within a
    // single use -- "admit on the second use" must mean the second USE,
    // not the second projection touched.  test-moe-cache.cpp pins this.
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

    // ---- mmap-tier advice (P1) --------------------------------------
    // Record the host range a miss-path staged copy just read; the whole
    // batch gets WILLNEED at step end.  No-op unless cfg.mmap_advise was
    // set.  Hits are deliberately not recorded: their bytes are
    // device-resident, and warming host pages nobody will read fights
    // the page cache on exactly the machines this feature targets.
    void advise_note_use(const void * host_src, size_t bytes);

    // Fire the step's advice: DONTNEED for evicted origins (skipping any
    // that overlap a range also used this step -- it is about to be
    // re-read), then WILLNEED for the coalesced used ranges.  Called from
    // the scheduler's step-end hook.  In-flight async H2D reads from an
    // advised range are safe: the mappings are read-only and file-backed,
    // so a dropped page refaults with identical contents.
    void advise_flush_step();

    // Drop the batches without advising (graph abandoned mid-compute).
    void advise_abandon_step();

    // Set inference phase: true=prefill, false=decode.
    // During prefill, miss counts are not incremented.
    void set_phase(bool is_prefill);

    // True while the current phase forbids admission (prefill with
    // prefill-admission off).  A promotion declined for this reason is
    // not cache pressure: the hybrid path must stage-and-execute
    // normally, or a cold prompt turns into per-row scalar CPU GEMVs
    // over every uncached expert.
    bool admission_blocked_by_phase() const;

    void reset();

    const moe_cache_stats & stats() const { return m_stats; }
    int slot_count() const { return (int)m_slots.size(); }

    // What the learning window decided.  Read-only observers for the
    // execution fingerprint: the learned per-projection size is what
    // decides, for the rest of the process, which projections can be
    // cached at all, so a fingerprint that omits it cannot tell a
    // different geometry from a different access pattern.  Unlocked
    // deliberately -- these are written once under m_mutex during
    // finalization and never again, and the fingerprint must not be able
    // to serialize the compute path against the HTTP threads.
    size_t geometry_bytes(int projection) const {
        return (projection >= 0 && projection < MOE_CACHE_N_PROJECTIONS)
             ? m_geom_bytes[projection] : 0;
    }
    size_t slot_bytes() const { return m_expert_bytes; }
    const void * pool_base() const { return m_pool; }

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

    // Async-fill state. m_pending_fills holds reservations made by
    // promote_projection this step; flush moves them to m_inflight_fills
    // with a device event. m_fill_seq stamps reservations so a reset()
    // in between invalidates them (the copy still lands in the pool --
    // harmlessly -- but the slot is never marked filled).
    bool m_async_fill = false;
    void * m_transfer_queue = nullptr;
    struct pending_fill {
        int slot_id = -1;
        int projection = -1;
        const void * host_src = nullptr;
        size_t bytes = 0;
        uint64_t seq = 0;
    };
    struct inflight_fill {
        pending_fill req;
        void * event = nullptr;
    };
    std::vector<pending_fill>  m_pending_fills;
    std::vector<inflight_fill> m_inflight_fills;
    uint64_t m_fill_seq = 0;

    mutable std::mutex m_mutex;  // protects slots, layer index, miss counts, phase
    std::vector<moe_cache_slot> m_slots;
    std::vector<moe_cache_layer_index> m_layer_index;
    std::vector<int> m_miss_counts;
    moe_cache_stats m_stats;

    // mmap-tier advice state (P1).  The batches live under m_mutex; the
    // syscalls fire in advise_flush_step AFTER swapping them out, so the
    // compute path never blocks on madvise.
    moe_cache_mmap_advise_fn m_mmap_advise = nullptr;
    std::vector<std::pair<const void *, size_t>> m_step_willneed;
    std::vector<std::pair<const void *, size_t>> m_step_dontneed;

    // Pending hybrid plans, keyed by staged-tensor device base. Written
    // at staging time, taken at op time within the same split execution;
    // ordinarily holds at most a couple of entries (interleaved main +
    // draft contexts).
    mutable std::mutex m_hybrid_mutex;
    std::map<const void *, hybrid_plan> m_hybrid_plans;
    // Device bases the hook has ever staged into (see note_staged_base).
    std::set<const void *> m_staged_bases;
};

#endif // GGML_SYCL_MOE_CACHE_HPP
