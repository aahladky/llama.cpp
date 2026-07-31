// MoE Expert Cache — implementation.

// Policy only: admission, eviction, phase, reset, stats.  No SYCL headers --
// every device operation goes through moe_cache_device_* in
// moe-cache-device.cpp.  That separation is what lets tests/test-moe-cache.cpp
// link this file and exercise the policy without a GPU.
#include "moe-cache.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sstream>

int moe_cache_parse_layer(const char * name) {
    if (!name) return -1;
    if (strncmp(name, "blk.", 4) != 0) return -1;
    if (!strstr(name, "exps")) return -1;
    return atoi(name + 4);
}

int moe_cache_parse_projection(const char * name) {
    if (!name) return -1;
    if (strstr(name, "gate_exps") || strstr(name, "gate_up_exps")) return 0;
    if (strstr(name, "up_exps"))   return 1;
    if (strstr(name, "down_exps")) return 2;
    return -1;
}

moe_expert_cache::~moe_expert_cache() {
    // Slots point into one pool allocation; freeing the pool frees them all.
    if (m_pool) {
        if (m_host_only) {
            free(m_pool);
        } else if (m_queue) {
            moe_cache_device_free(m_pool, m_queue);
        }
        m_pool = nullptr;
    }
    for (auto & slot : m_slots) {
        slot.device_ptr = nullptr;
    }
}

bool moe_expert_cache::init(const moe_cache_config & cfg, void * queue) {
    // Host-only is opt-in AND requires a null queue, so a production caller
    // that accidentally passes null still gets a hard failure rather than a
    // silently device-less cache.
    m_host_only = (queue == nullptr) && cfg.host_only_for_testing;
    if ((!queue && !m_host_only) || cfg.budget_bytes == 0 || cfg.n_layers == 0 ||
        cfg.n_experts == 0 || cfg.expert_bytes == 0) {
        return false;
    }

    m_queue        = queue;
    m_budget_bytes = cfg.budget_bytes;
    m_n_layers     = cfg.n_layers;
    m_n_experts    = cfg.n_experts;
    m_expert_bytes = cfg.expert_bytes;
    m_geom_bytes[0] = cfg.gate_bytes;
    m_geom_bytes[1] = cfg.up_bytes;
    m_geom_bytes[2] = cfg.down_bytes;
    m_admission_misses = std::max(1, cfg.admission_misses);
    m_prefill_admit   = cfg.prefill_admit;
    m_is_prefill      = false;
    m_use_slru        = (cfg.policy == "slru");
    m_deferred        = false;

    // Projection sizes are taken individually rather than assumed equal, so
    // a model whose gate/up/down differ still gets correctly sized regions.
    // A config that reports no per-projection sizes at all is not cacheable
    // through this entry point: guessing would mean over-reading the host
    // weights. (Learned geometry goes through init_deferred instead.)
    if (cfg.gate_bytes == 0 || cfg.up_bytes == 0 || cfg.down_bytes == 0) {
        fprintf(stderr, "moe_cache: refusing to initialize -- projection geometry "
                        "unknown (gate=%zu up=%zu down=%zu)\n",
                cfg.gate_bytes, cfg.up_bytes, cfg.down_bytes);
        return false;
    }
    return finalize_geometry_locked();
}

bool moe_expert_cache::init_deferred(const moe_cache_config & cfg, void * queue) {
    // Same validity rules as init(), minus the geometry: that is exactly
    // what deferred mode exists to learn. The pool is allocated later, in
    // finalize_geometry_locked(), from the sizes real staged copies
    // reported -- never from a first-tensor guess that a fused gate_up
    // layout or unequal projections would falsify.
    m_host_only = (queue == nullptr) && cfg.host_only_for_testing;
    if ((!queue && !m_host_only) || cfg.budget_bytes == 0 || cfg.n_layers == 0 ||
        cfg.n_experts == 0) {
        return false;
    }
    m_queue        = queue;
    m_budget_bytes = cfg.budget_bytes;
    m_n_layers     = cfg.n_layers;
    m_n_experts    = cfg.n_experts;
    m_admission_misses = std::max(1, cfg.admission_misses);
    m_prefill_admit   = cfg.prefill_admit;
    m_is_prefill      = false;
    m_use_slru        = (cfg.policy == "slru");
    m_deferred        = true;
    m_initialized     = false;
    return true;
}

bool moe_expert_cache::is_learning() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_deferred && !m_initialized;
}

bool moe_expert_cache::observe_geometry(int32_t layer, int projection,
                                        size_t proj_bytes, const void * origin) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_deferred) return m_initialized;
    if (m_initialized) return true;
    if (layer < 0 || projection < 0 || projection >= MOE_CACHE_N_PROJECTIONS ||
        proj_bytes == 0 || origin == nullptr) {
        return false;
    }

    // First sighting of this projection index fixes its size. A later
    // tensor of the same projection with a different size (mixed quants
    // across layers, or a second model) keeps the first size and is simply
    // never cached -- the guards in lookup/promote enforce that.
    if (m_geom_bytes[projection] == 0) {
        m_geom_bytes[projection] = proj_bytes;
    }

    // The hook fires once per routed expert, so one tensor produces a run
    // of identical origins. A CHANGE of origin means the previous tensor's
    // staging finished; seeing an origin from that finished set again means
    // the graph wrapped into a second pass -- every projection that exists
    // has now been seen at least once, and the pool can be carved.
    if (origin == m_obs_current_origin) {
        return false;
    }
    if (m_obs_current_origin != nullptr &&
        std::find(m_obs_completed.begin(), m_obs_completed.end(),
                  m_obs_current_origin) == m_obs_completed.end()) {
        m_obs_completed.push_back(m_obs_current_origin);
    }
    const bool revisit = std::find(m_obs_completed.begin(), m_obs_completed.end(),
                                   origin) != m_obs_completed.end();
    m_obs_current_origin = origin;
    if (!revisit) {
        return false;
    }
    if (!finalize_geometry_locked()) {
        // Leave m_deferred set so is_learning() keeps reporting true and
        // the caller keeps falling back to plain host copies.
        return false;
    }
    // The one line hardware validation looks for: what the cache decided
    // the model's projections look like, and what that bought.
    fprintf(stderr, "moe_cache: geometry learned (gate=%zu up=%zu down=%zu), "
                    "%d slots of %zu bytes\n",
            m_geom_bytes[0], m_geom_bytes[1], m_geom_bytes[2],
            (int)m_slots.size(), m_expert_bytes);
    m_obs_completed.clear();
    m_obs_completed.shrink_to_fit();
    m_obs_current_origin = nullptr;
    return true;
}

bool moe_expert_cache::finalize_geometry_locked() {
    // Slot layout from the per-projection sizes actually present. A
    // projection with size 0 does not exist for this model (fused gate_up
    // has no separate up) and gets a zero-width region that the guards
    // reject.
    size_t slot_bytes = 0;
    for (int p = 0; p < MOE_CACHE_N_PROJECTIONS; p++) {
        slot_bytes += m_geom_bytes[p];
    }
    if (slot_bytes == 0) {
        fprintf(stderr, "moe_cache: no projection geometry observed -- not initializing\n");
        return false;
    }
    m_expert_bytes = slot_bytes;

    int n_slots = (int)(m_budget_bytes / slot_bytes);
    if (n_slots <= 0) {
        fprintf(stderr, "moe_cache: budget %zu below one slot (%zu bytes)\n",
                m_budget_bytes, slot_bytes);
        return false;
    }
    m_protected_limit = m_use_slru ? std::max(1, (int)(n_slots * 0.8)) : 0;

    // One contiguous pool for every slot rather than n_slots separate
    // device allocations: fewer allocator round-trips, and the cache cannot
    // end up scattered across the device address space.
    m_pool_bytes = slot_bytes * (size_t)n_slots;
    if (m_host_only) {
        m_pool = malloc(m_pool_bytes);
    } else {
        m_pool = moe_cache_device_alloc(m_pool_bytes, m_queue);
    }
    // The allocator reports both a throw and a null OOM return as null, so
    // this check covers every failure path.
    if (!m_pool) {
        fprintf(stderr, "moe_cache: pool allocation of %zu bytes returned null\n",
                m_pool_bytes);
        m_pool_bytes = 0;
        return false;
    }

    m_slots.resize(n_slots);
    for (int i = 0; i < n_slots; i++) {
        m_slots[i].device_ptr = (char *)m_pool + (size_t)i * slot_bytes;
        m_slots[i].bytes = slot_bytes;
        m_slots[i].gate_bytes = m_geom_bytes[0];
        m_slots[i].up_bytes   = m_geom_bytes[1];
        m_slots[i].down_bytes = m_geom_bytes[2];
        m_slots[i].gate_offset = 0;
        m_slots[i].up_offset   = m_geom_bytes[0];
        m_slots[i].down_offset = m_geom_bytes[0] + m_geom_bytes[1];
    }

    m_layer_index.resize(m_n_layers);
    for (int l = 0; l < m_n_layers; l++) {
        m_layer_index[l].init(m_n_experts);
    }

    m_miss_counts.assign((size_t)m_n_layers * m_n_experts * MOE_CACHE_N_PROJECTIONS, 0);
    m_initialized = true;
    m_stats.reset();
    return true;
}

void * moe_expert_cache::lookup(int32_t layer, int32_t expert, int projection,
                                size_t proj_bytes, const void * host_src) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized || layer < 0 || layer >= m_n_layers ||
        expert < 0 || expert >= m_n_experts ||
        projection < 0 || projection >= MOE_CACHE_N_PROJECTIONS) {
        m_stats.misses++;
        return nullptr;
    }

    // Geometry guard: only serve from cache when the caller's projection
    // size matches the learned size (otherwise the slot region would be
    // over-read).  Not counted as a miss -- this expert is not cacheable.
    const size_t expect = m_geom_bytes[projection];
    if (expect == 0 || expect != proj_bytes) {
        return nullptr;
    }

    m_tick++;
    int slot_id = m_layer_index[layer].expert_to_slot[expert];
    if (slot_id >= 0 && slot_id < (int)m_slots.size() &&
        m_slots[slot_id].occupied &&
        m_slots[slot_id].key.layer == layer &&
        m_slots[slot_id].key.expert == expert) {
        // Check that the requested projection is filled, and that it was
        // filled FROM THIS TENSOR: two models on one device share the
        // cache and reuse identical names, so (layer, expert) alone would
        // serve model B a slot holding model A's weights.
        const moe_cache_slot & s = m_slots[slot_id];
        bool filled = false;
        size_t offset = 0;
        switch (projection) {
            case 0: filled = (s.filled_mask & 1) != 0; offset = s.gate_offset; break;  // gate
            case 1: filled = (s.filled_mask & 2) != 0; offset = s.up_offset;   break;  // up
            case 2: filled = (s.filled_mask & 4) != 0; offset = s.down_offset; break;  // down
        }
        if (filled && host_src != nullptr &&
            s.proj_origin[projection] != host_src) {
            filled = false;
        }
        if (!filled) {
            m_stats.misses++;
            if (m_is_prefill) m_stats.prefill_misses++;
            else              m_stats.decode_misses++;
            return nullptr;
        }
        touch(slot_id);
        m_stats.hits++;
        m_stats.cache_served_projections++;
        if (m_is_prefill) m_stats.prefill_hits++;
        else              m_stats.decode_hits++;
        // Clear only THIS projection's admission progress.  Clearing the
        // whole expert's would let a cached gate reset the admission the
        // still-uncached up and down projections had accumulated.
        if (int mi = miss_index(layer, expert, projection); mi >= 0) {
            m_miss_counts[mi] = 0;
        }

        // SLRU: promote probationary slot to protected on second access.
        if (m_use_slru && !m_slots[slot_id].protected_seg &&
            m_slots[slot_id].access_count >= 2) {
            promote_to_protected(slot_id);
        }
        // Return the projection region within the slot, not the slot base.
        return (char *)m_slots[slot_id].device_ptr + offset;
    }

    m_stats.misses++;
    if (m_is_prefill) m_stats.prefill_misses++;
    else              m_stats.decode_misses++;
    return nullptr;
}

void moe_expert_cache::record_miss(int32_t layer, int32_t expert, int projection) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized) return;
    const int idx = miss_index(layer, expert, projection);
    if (idx < 0) return;

    // Prefill protection: don't increment miss counts during prefill
    // unless prefill admission is explicitly enabled.
    if (m_is_prefill && !m_prefill_admit) {
        return;
    }
    m_miss_counts[idx]++;
}

void * moe_expert_cache::promote_projection(int32_t layer, int32_t expert,
                                             int projection, const void * host_src,
                                             size_t proj_bytes) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized || !host_src) return nullptr;
    if (layer < 0 || layer >= m_n_layers || expert < 0 || expert >= m_n_experts) return nullptr;

    // Geometry guard: bail out of caching when the caller's projection size
    // does not match the learned size (avoids over-reading host_src or
    // overflowing the slot region).
    if (projection < 0 || projection >= MOE_CACHE_N_PROJECTIONS) return nullptr;
    const size_t expect = m_geom_bytes[projection];
    if (expect == 0 || expect != proj_bytes) return nullptr;

    const int idx = miss_index(layer, expert, projection);
    if (idx < 0) return nullptr;
    if (m_miss_counts[idx] < m_admission_misses) {
        return nullptr;
    }

    // Find existing slot for this expert, or allocate a new one.
    int slot_id = m_layer_index[layer].expert_to_slot[expert];
    if (slot_id < 0) {
        // Allocate a new slot.
        for (int i = 0; i < (int)m_slots.size(); i++) {
            if (!m_slots[i].occupied) {
                slot_id = i;
                break;
            }
        }
        if (slot_id < 0) {
            slot_id = m_use_slru ? evict_slru() : evict_lru();
        }
        if (slot_id < 0) return nullptr;

        moe_cache_slot & slot = m_slots[slot_id];
        if (slot.occupied && slot.key.valid()) {
            m_layer_index[slot.key.layer].expert_to_slot[slot.key.expert] = -1;
            m_stats.evictions++;
        }
        slot.key = {layer, expert};
        slot.occupied = true;
        slot.filled_mask = 0;
        slot.last_access = m_tick;
        slot.access_count = 1;
        slot.protected_seg = false;
        for (int p = 0; p < MOE_CACHE_N_PROJECTIONS; p++) {
            slot.proj_origin[p] = nullptr;
        }
        m_layer_index[layer].expert_to_slot[expert] = slot_id;
    }

    moe_cache_slot & slot = m_slots[slot_id];

    // Copy this projection from host to device.
    size_t offset = 0;
    size_t bytes = 0;
    uint8_t bit = 0;
    switch (projection) {
        case 0: offset = slot.gate_offset; bytes = slot.gate_bytes; bit = 1; break;
        case 1: offset = slot.up_offset;   bytes = slot.up_bytes;   bit = 2; break;
        case 2: offset = slot.down_offset; bytes = slot.down_bytes; bit = 4; break;
        default: return nullptr;
    }
    if (bytes == 0) return nullptr;

    if (m_host_only) {
        // Unit-test mode: the "device" pool is host memory, so a plain
        // copy keeps the contents checkable without a GPU.
        memcpy((char *)slot.device_ptr + offset, host_src, bytes);
    } else if (!moe_cache_device_copy((char *)slot.device_ptr + offset,
                                      host_src, bytes, m_queue)) {
        return nullptr;
    }

    m_stats.h2d_bytes += bytes;
    slot.filled_mask |= bit;
    // The origin is the projection's tensor identity; lookups only serve
    // requests that present the same host pointer.
    slot.proj_origin[projection] = host_src;
    m_miss_counts[idx] = 0;
    m_stats.promotions++;

    // Return the projection region within the slot, not the slot base.
    return (char *)slot.device_ptr + offset;
}

bool moe_expert_cache::contains(int32_t layer, int32_t expert,
                                int projection) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_initialized || layer < 0 || layer >= m_n_layers ||
        expert < 0 || expert >= m_n_experts ||
        projection < 0 || projection >= MOE_CACHE_N_PROJECTIONS) {
        return false;
    }
    const int slot_id = m_layer_index[layer].expert_to_slot[expert];
    if (slot_id < 0 || slot_id >= (int)m_slots.size()) {
        return false;
    }
    const moe_cache_slot & s = m_slots[slot_id];
    if (!s.occupied || s.key.layer != layer || s.key.expert != expert) {
        return false;
    }
    // Residency is per projection: a slot can hold gate while up and down
    // are still absent, and a partition that treated the expert as one
    // unit would send two thirds of the work to the wrong tier.
    return (s.filled_mask & (1u << projection)) != 0;
}

void moe_expert_cache::set_phase(bool is_prefill) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_is_prefill = is_prefill;
}

void moe_expert_cache::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);

    for (auto & slot : m_slots) {
        slot.occupied = false;
        slot.key = {-1, -1};
        slot.last_access = 0;
        slot.access_count = 0;
        slot.protected_seg = false;
        slot.filled_mask = 0;
        for (int p = 0; p < MOE_CACHE_N_PROJECTIONS; p++) {
            slot.proj_origin[p] = nullptr;
        }
    }
    for (auto & idx : m_layer_index) {
        std::fill(idx.expert_to_slot.begin(), idx.expert_to_slot.end(), -1);
    }
    std::fill(m_miss_counts.begin(), m_miss_counts.end(), 0);
    m_tick = 0;
    m_stats.reset();
}

int moe_expert_cache::slots_used() const {
    std::lock_guard<std::mutex> lock(m_mutex);

    int count = 0;
    for (const auto & s : m_slots) {
        if (s.occupied) count++;
    }
    return count;
}

std::string moe_expert_cache::stats_json() const {
    std::lock_guard<std::mutex> lock(m_mutex);

    int used = 0;
    for (const auto & s : m_slots) {
        if (s.occupied) used++;
    }

    std::ostringstream ss;
    ss << "{";
    ss << "\"hits\":" << m_stats.hits.load() << ",";
    ss << "\"misses\":" << m_stats.misses.load() << ",";
    ss << "\"evictions\":" << m_stats.evictions.load() << ",";
    ss << "\"promotions\":" << m_stats.promotions.load() << ",";
    ss << "\"h2d_bytes\":" << m_stats.h2d_bytes.load() << ",";
    ss << "\"cache_served_projections\":" << m_stats.cache_served_projections.load() << ",";
    ss << "\"host_weight_copy_fallbacks\":" << m_stats.host_weight_copy_fallbacks.load() << ",";
    ss << "\"prefill_hits\":" << m_stats.prefill_hits.load() << ",";
    ss << "\"prefill_misses\":" << m_stats.prefill_misses.load() << ",";
    ss << "\"decode_hits\":" << m_stats.decode_hits.load() << ",";
    ss << "\"decode_misses\":" << m_stats.decode_misses.load() << ",";
    ss << "\"slru_promotions\":" << m_stats.slru_promotions.load() << ",";
    ss << "\"slru_demotions\":" << m_stats.slru_demotions.load() << ",";
    ss << "\"slot_count\":" << m_slots.size() << ",";
    ss << "\"slots_used\":" << used << ",";
    // Learned geometry, so a support bundle can see what the cache decided
    // the model's projections look like (0 = projection absent, e.g. no
    // separate up for a fused gate_up layout).
    ss << "\"geometry\":{\"gate\":" << m_geom_bytes[0]
       << ",\"up\":" << m_geom_bytes[1]
       << ",\"down\":" << m_geom_bytes[2] << "},";
    ss << "\"learning\":" << ((m_deferred && !m_initialized) ? "true" : "false") << ",";
    ss << "\"budget_bytes\":" << m_budget_bytes;
    uint64_t total = m_stats.hits.load() + m_stats.misses.load();
    ss << ",\"hit_rate\":" << (total > 0 ? (double)m_stats.hits.load() / total : 0.0);
    uint64_t d_total = m_stats.decode_hits.load() + m_stats.decode_misses.load();
    ss << ",\"decode_hit_rate\":" << (d_total > 0 ? (double)m_stats.decode_hits.load() / d_total : 0.0);
    ss << "}";
    return ss.str();
}

int moe_expert_cache::evict_lru() {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < (int)m_slots.size(); i++) {
        if (m_slots[i].occupied && !m_slots[i].protected_seg &&
            m_slots[i].last_access < oldest) {
            oldest = m_slots[i].last_access;
            victim = i;
        }
    }
    return victim;
}

void moe_expert_cache::touch(int slot_id) {
    m_slots[slot_id].last_access = m_tick;
    m_slots[slot_id].access_count++;
}

int moe_expert_cache::evict_slru() {
    // SLRU eviction: try probationary segment first.
    // Only evict from protected if probationary is empty.
    int victim = -1;
    uint64_t oldest = UINT64_MAX;

    // First pass: probationary (non-protected) slots.
    for (int i = 0; i < (int)m_slots.size(); i++) {
        if (m_slots[i].occupied && !m_slots[i].protected_seg &&
            m_slots[i].last_access < oldest) {
            oldest = m_slots[i].last_access;
            victim = i;
        }
    }
    if (victim >= 0) return victim;

    // Second pass: protected slots (demote LRU protected to probationary,
    // then evict the now-probationary slot).
    oldest = UINT64_MAX;
    for (int i = 0; i < (int)m_slots.size(); i++) {
        if (m_slots[i].occupied && m_slots[i].protected_seg &&
            m_slots[i].last_access < oldest) {
            oldest = m_slots[i].last_access;
            victim = i;
        }
    }
    if (victim >= 0) {
        m_slots[victim].protected_seg = false;
        m_stats.slru_demotions++;
    }
    return victim;
}

void moe_expert_cache::promote_to_protected(int slot_id) {
    // Count current protected slots.
    int protected_count = 0;
    for (const auto & s : m_slots) {
        if (s.occupied && s.protected_seg) protected_count++;
    }

    if (protected_count < m_protected_limit) {
        // Room in protected segment -- promote directly.
        m_slots[slot_id].protected_seg = true;
        m_stats.slru_promotions++;
        return;
    }

    // Protected segment is full: demote LRU protected slot to probationary,
    // then promote the new hot slot.
    int demote = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < (int)m_slots.size(); i++) {
        if (m_slots[i].occupied && m_slots[i].protected_seg &&
            i != slot_id &&
            m_slots[i].last_access < oldest) {
            oldest = m_slots[i].last_access;
            demote = i;
        }
    }
    if (demote >= 0) {
        m_slots[demote].protected_seg = false;
        m_stats.slru_demotions++;
    }
    m_slots[slot_id].protected_seg = true;
    m_stats.slru_promotions++;
}
