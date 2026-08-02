#include "moe-fingerprint.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unistd.h>

// One writer, one lock.  The staging hook and the op already take the cache's
// mutex on the same thread, so an extra uncontended lock here is the same
// order of cost; and correctness under two device compute threads matters
// more than shaving it.  The FILE gets a large buffer so the hot path
// memcpy's into userspace and syscalls rarely.
namespace {

struct fp_state {
    FILE *     f       = nullptr;
    bool       raw     = false;
    uint64_t   step    = 0;
    std::mutex mu;
    // Pointer -> ordinal of first appearance.  See the header: this is what
    // makes two runs comparable without the allocator's addresses in the way.
    std::unordered_map<const void *, uint32_t> ids;
    std::string buf;

    uint32_t id_of(const void * p) {
        if (p == nullptr) return 0;
        auto it = ids.find(p);
        if (it != ids.end()) return it->second;
        const uint32_t next = (uint32_t) ids.size() + 1;
        ids.emplace(p, next);
        return next;
    }
};

fp_state * fp_get() {
    static fp_state * s = [] () -> fp_state * {
        const char * path = getenv("GGML_MOE_FINGERPRINT");
        if (!path || !*path) return nullptr;
        FILE * f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "moe_fingerprint: cannot open %s for writing\n", path);
            return nullptr;
        }
        auto * st = new fp_state();
        st->f = f;
        setvbuf(f, nullptr, _IOFBF, 4u << 20);
        const char * raw = getenv("GGML_MOE_FINGERPRINT_RAW");
        st->raw = raw && *raw && *raw != '0';
        st->ids.reserve(1u << 16);
        fprintf(f, "V 1 %d\n", (int) getpid());
        // Flushed at exit rather than per record: a crash loses the tail, and
        // losing the tail of a run that crashed is not the failure mode this
        // is built for.
        atexit([] () {
            // fp_get() is re-entered here, but the static is already
            // constructed, so this is just a load.
            if (fp_state * s2 = fp_get()) {
                std::lock_guard<std::mutex> lock(s2->mu);
                fflush(s2->f);
            }
        });
        return st;
    }();
    return s;
}

} // namespace

bool moe_fp_enabled() {
    return fp_get() != nullptr;
}

uint64_t moe_fp_step() {
    fp_state * s = fp_get();
    return s ? s->step : 0;
}

void moe_fp_stage(int dev, int layer, int proj, int32_t expert, char disp,
                  const void * cpy_base, const void * host_src) {
    fp_state * s = fp_get();
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mu);
    if (s->raw) {
        fprintf(s->f, "S %llu %d %d %d %d %c %u %u %p %p\n",
                (unsigned long long) s->step, dev, layer, proj, (int) expert, disp,
                s->id_of(cpy_base), s->id_of(host_src), cpy_base, host_src);
    } else {
        fprintf(s->f, "S %llu %d %d %d %d %c %u %u\n",
                (unsigned long long) s->step, dev, layer, proj, (int) expert, disp,
                s->id_of(cpy_base), s->id_of(host_src));
    }
}

void moe_fp_geom(int dev, size_t gate, size_t up, size_t down,
                 int slots, size_t slot_bytes, const void * pool) {
    fp_state * s = fp_get();
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mu);
    fprintf(s->f, "G %llu %d %zu %zu %zu %d %zu %u\n",
            (unsigned long long) s->step, dev, gate, up, down, slots, slot_bytes,
            s->id_of(pool));
}

void moe_fp_op(int dev, int layer, int proj,
               int64_t ne00, int64_t ne01, int64_t ne11, int64_t ne12,
               const char * path, int plan_n, int cpu_rows, int gpu_rows,
               int reorder, const void * src0_data, uint64_t ids_hash) {
    fp_state * s = fp_get();
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mu);
    fprintf(s->f, "O %llu %d %d %d %lld %lld %lld %lld %s %d %d %d %d %u %016llx\n",
            (unsigned long long) s->step, dev, layer, proj,
            (long long) ne00, (long long) ne01, (long long) ne11, (long long) ne12,
            path, plan_n, cpu_rows, gpu_rows, reorder, s->id_of(src0_data),
            (unsigned long long) ids_hash);
}

uint64_t moe_fp_hash(const void * data, size_t bytes) {
    // FNV-1a: no table, no allocation, and good enough to make two differing
    // routing tensors collide only by accident we would notice as a
    // non-monotonic result.
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

bool moe_fp_dst_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_MOE_FINGERPRINT_DST");
        return v && *v && *v != '0';
    }();
    return on && fp_get() != nullptr;
}

void moe_fp_dst(int dev, int layer, int proj, uint64_t dst_hash) {
    fp_state * s = fp_get();
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mu);
    fprintf(s->f, "D %llu %d %d %d %016llx\n",
            (unsigned long long) s->step, dev, layer, proj,
            (unsigned long long) dst_hash);
}

void moe_fp_step_end(int dev, uint64_t hits, uint64_t misses, uint64_t evictions,
                     uint64_t promotions, uint64_t fallbacks, uint64_t served,
                     uint64_t staging_skips, int slots_used) {
    fp_state * s = fp_get();
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mu);
    fprintf(s->f, "E %llu %d %llu %llu %llu %llu %llu %llu %llu %d\n",
            (unsigned long long) s->step, dev,
            (unsigned long long) hits, (unsigned long long) misses,
            (unsigned long long) evictions, (unsigned long long) promotions,
            (unsigned long long) fallbacks, (unsigned long long) served,
            (unsigned long long) staging_skips, slots_used);
}

void moe_fp_step_advance() {
    fp_state * s = fp_get();
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mu);
    s->step++;
}

void moe_fp_event(const char * text) {
    fp_state * s = fp_get();
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mu);
    fprintf(s->f, "X %llu %s\n", (unsigned long long) s->step, text ? text : "");
}
