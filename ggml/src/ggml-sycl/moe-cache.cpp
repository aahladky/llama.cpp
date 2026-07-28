// MoE Expert Cache — implementation.

#include "moe-cache.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sstream>

moe_expert_cache::~moe_expert_cache() {
    if (m_queue) {
        for (auto & slot : m_slots) {
            if (slot.device_ptr) {
                try {
                    sycl::free(slot.device_ptr, *m_queue);
                } catch (...) {}
                slot.device_ptr = nullptr;
            }
        }
    }
}

bool moe_expert_cache::init(const moe_cache_config & cfg, queue_ptr queue) {
    if (!queue || cfg.budget_bytes == 0 || cfg.n_layers == 0 ||
        cfg.n_experts == 0 || cfg.expert_bytes == 0) {
        return false;
    }

    m_queue        = queue;
    m_budget_bytes = cfg.budget_bytes;
    m_n_layers     = cfg.n_layers;
    m_n_experts    = cfg.n_experts;
    m_expert_bytes = cfg.expert_bytes;
    m_admission_misses = std::max(1, cfg.admission_misses);

    size_t slot_bytes = cfg.gate_bytes + cfg.up_bytes + cfg.down_bytes;
    if (slot_bytes == 0) {
        slot_bytes = cfg.expert_bytes;
    }
    int n_slots = (int)(cfg.budget_bytes / slot_bytes);
    if (n_slots <= 0) {
        return false;
    }

    m_slots.resize(n_slots);
    for (int i = 0; i < n_slots; i++) {
        try {
            m_slots[i].device_ptr = sycl::malloc_device(slot_bytes, *queue);
        } catch (const sycl::exception & e) {
            fprintf(stderr, "moe_cache: failed to allocate slot %d (%zu bytes): %s\n",
                    i, slot_bytes, e.what());
            for (int j = 0; j < i; j++) {
                sycl::free(m_slots[j].device_ptr, *queue);
                m_slots[j].device_ptr = nullptr;
            }
            m_slots.clear();
            return false;
        }
        m_slots[i].bytes = slot_bytes;
        m_slots[i].gate_bytes = cfg.gate_bytes;
        m_slots[i].up_bytes   = cfg.up_bytes;
        m_slots[i].down_bytes = cfg.down_bytes;
        m_slots[i].gate_offset = 0;
        m_slots[i].up_offset   = cfg.gate_bytes;
        m_slots[i].down_offset = cfg.gate_bytes + cfg.up_bytes;
    }

    m_layer_index.resize(m_n_layers);
    for (int l = 0; l < m_n_layers; l++) {
        m_layer_index[l].init(m_n_experts);
    }

    m_miss_counts.assign(m_n_layers * m_n_experts, 0);
    m_initialized = true;
    m_stats.reset();
    return true;
}

void * moe_expert_cache::lookup(int32_t layer, int32_t expert) {
    if (!m_initialized || layer < 0 || layer >= m_n_layers ||
        expert < 0 || expert >= m_n_experts) {
        m_stats.misses++;
        return nullptr;
    }

    m_tick++;
    int slot_id = m_layer_index[layer].expert_to_slot[expert];
    if (slot_id >= 0 && slot_id < (int)m_slots.size() &&
        m_slots[slot_id].occupied &&
        m_slots[slot_id].key.layer == layer &&
        m_slots[slot_id].key.expert == expert) {
        touch(slot_id);
        m_stats.hits++;
        m_stats.gpu_expert_calls++;
        m_miss_counts[layer * m_n_experts + expert] = 0;
        return m_slots[slot_id].device_ptr;
    }

    m_stats.misses++;
    return nullptr;
}

void moe_expert_cache::record_miss(int32_t layer, int32_t expert) {
    if (!m_initialized || layer < 0 || layer >= m_n_layers ||
        expert < 0 || expert >= m_n_experts) {
        return;
    }
    m_miss_counts[layer * m_n_experts + expert]++;
}

void * moe_expert_cache::promote(int32_t layer, int32_t expert,
                                  const void * host_src_gate,
                                  const void * host_src_up,
                                  const void * host_src_down) {
    if (!m_initialized) return nullptr;

    int idx = layer * m_n_experts + expert;
    if (m_miss_counts[idx] < m_admission_misses) {
        return nullptr;
    }

    int slot_id = -1;
    for (int i = 0; i < (int)m_slots.size(); i++) {
        if (!m_slots[i].occupied) {
            slot_id = i;
            break;
        }
    }
    if (slot_id < 0) {
        slot_id = evict_lru();
    }
    if (slot_id < 0) {
        return nullptr;
    }

    moe_cache_slot & slot = m_slots[slot_id];

    if (slot.occupied && slot.key.valid()) {
        m_layer_index[slot.key.layer].expert_to_slot[slot.key.expert] = -1;
        m_stats.evictions++;
    }

    size_t total_bytes = 0;
    try {
        if (host_src_gate && slot.gate_bytes > 0) {
            m_queue->memcpy(
                (char *)slot.device_ptr + slot.gate_offset,
                host_src_gate, slot.gate_bytes).wait();
            total_bytes += slot.gate_bytes;
        }
        if (host_src_up && slot.up_bytes > 0) {
            m_queue->memcpy(
                (char *)slot.device_ptr + slot.up_offset,
                host_src_up, slot.up_bytes).wait();
            total_bytes += slot.up_bytes;
        }
        if (host_src_down && slot.down_bytes > 0) {
            m_queue->memcpy(
                (char *)slot.device_ptr + slot.down_offset,
                host_src_down, slot.down_bytes).wait();
            total_bytes += slot.down_bytes;
        }
    } catch (const sycl::exception & e) {
        fprintf(stderr, "moe_cache: promote failed: %s\n", e.what());
        return nullptr;
    }

    m_stats.h2d_bytes += total_bytes;

    slot.key = {layer, expert};
    slot.occupied = true;
    slot.last_access = m_tick;
    slot.access_count = 1;

    m_layer_index[layer].expert_to_slot[expert] = slot_id;
    m_miss_counts[idx] = 0;
    m_stats.promotions++;

    return slot.device_ptr;
}

void moe_expert_cache::reset() {
    for (auto & slot : m_slots) {
        slot.occupied = false;
        slot.key = {-1, -1};
        slot.last_access = 0;
        slot.access_count = 0;
    }
    for (auto & idx : m_layer_index) {
        std::fill(idx.expert_to_slot.begin(), idx.expert_to_slot.end(), -1);
    }
    std::fill(m_miss_counts.begin(), m_miss_counts.end(), 0);
    m_tick = 0;
    m_stats.reset();
}

int moe_expert_cache::slots_used() const {
    int count = 0;
    for (const auto & s : m_slots) {
        if (s.occupied) count++;
    }
    return count;
}

std::string moe_expert_cache::stats_json() const {
    std::ostringstream ss;
    ss << "{";
    ss << "\"hits\":" << m_stats.hits.load() << ",";
    ss << "\"misses\":" << m_stats.misses.load() << ",";
    ss << "\"evictions\":" << m_stats.evictions.load() << ",";
    ss << "\"promotions\":" << m_stats.promotions.load() << ",";
    ss << "\"h2d_bytes\":" << m_stats.h2d_bytes.load() << ",";
    ss << "\"gpu_expert_calls\":" << m_stats.gpu_expert_calls.load() << ",";
    ss << "\"cpu_expert_calls\":" << m_stats.cpu_expert_calls.load() << ",";
    ss << "\"sync_wait_ns\":" << m_stats.sync_wait_ns.load() << ",";
    ss << "\"slot_count\":" << m_slots.size() << ",";
    ss << "\"slots_used\":" << slots_used() << ",";
    ss << "\"budget_bytes\":" << m_budget_bytes;
    uint64_t total = m_stats.hits.load() + m_stats.misses.load();
    ss << ",\"hit_rate\":" << (total > 0 ? (double)m_stats.hits.load() / total : 0.0);
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
