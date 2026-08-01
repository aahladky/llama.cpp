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

#ifdef GGML_MOE_HYBRID_CPU_KERNELS
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Miss-path profiling. The row/byte/thread counters are one atomic add
// per call and always on; the ns breakdown needs a clock read on either
// side of every dequantized weight row, which at this row size is a few
// percent of the row's own cost, so it is opt-in.
bool moe_cpu_profile_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_MOE_HYBRID_PROFILE");
        return v && v[0] && v[0] != '0';
    }();
    return on;
}

static inline int64_t moe_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline void moe_counter_add(std::atomic<int64_t> & a, int64_t v) {
    if (v) {
        a.fetch_add(v, std::memory_order_relaxed);
    }
}

// High-water mark, not a sum: "how wide did the tier ever get" is the
// question a thread count answers; summing it across calls answers none.
static inline void moe_counter_max(std::atomic<int64_t> & a, int64_t v) {
    int64_t cur = a.load(std::memory_order_relaxed);
    while (v > cur && !a.compare_exchange_weak(cur, v,
                                               std::memory_order_relaxed)) {
    }
}

moe_cpu_tier_stats moe_cpu_tier_snapshot(const moe_cpu_tier_counters & c) {
    moe_cpu_tier_stats s;
    s.calls         = c.calls.load(std::memory_order_relaxed);
    s.jobs          = c.jobs.load(std::memory_order_relaxed);
    s.weight_rows   = c.weight_rows.load(std::memory_order_relaxed);
    s.kernel_rows   = c.kernel_rows.load(std::memory_order_relaxed);
    s.weight_bytes  = c.weight_bytes.load(std::memory_order_relaxed);
    s.threads_used  = c.threads_used.load(std::memory_order_relaxed);
    s.wall_ns       = c.wall_ns.load(std::memory_order_relaxed);
    s.dispatch_ns   = c.dispatch_ns.load(std::memory_order_relaxed);
    s.dequant_ns    = c.dequant_ns.load(std::memory_order_relaxed);
    s.matmul_ns     = c.matmul_ns.load(std::memory_order_relaxed);
    s.quant_act_ns  = c.quant_act_ns.load(std::memory_order_relaxed);
    s.profiled_rows = c.profiled_rows.load(std::memory_order_relaxed);
    return s;
}

// Per-worker ns accumulators, folded into the shared counters once at the
// end of a worker's slice rather than per row: an atomic add per row on a
// line every worker touches costs more than the timing it records.
struct moe_cpu_row_timing {
    int64_t dequant_ns = 0;
    int64_t matmul_ns  = 0;
    int64_t rows       = 0;
};

// A persistent set of workers for the miss path.
//
// The tier used to std::thread::create its workers per call. At decode
// that is one spawn set per projection per layer -- 144 of them per token
// on the 122B target -- and once the kernels made the work itself cheap,
// spawning cost more than 10% of the op. The threads now live across
// calls and are woken by a generation counter.
//
// One batch at a time (the mutex): two contexts running miss work at once
// would be competing for the same cores anyway, and serializing them is
// simpler than sharing the pool between them.
class moe_cpu_pool {
public:
    static moe_cpu_pool & instance() {
        static moe_cpu_pool pool;
        return pool;
    }

    // Runs body(0..n_workers-1) with the caller as worker 0, and returns
    // once every worker has finished. n_workers is clamped to the pool.
    void run(int n_workers, const std::function<void(int)> & body) {
        std::lock_guard<std::mutex> batch(m_batch);
        n_workers = std::max(1, std::min(n_workers, m_size + 1));
        if (n_workers == 1) {
            body(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_body     = &body;
            m_helpers  = n_workers - 1;
            m_pending.store(m_helpers, std::memory_order_relaxed);
            m_generation.fetch_add(1, std::memory_order_release);
        }
        m_wake.notify_all();
        // The helpers hold a pointer to body and are running it right
        // now, so an escaping exception must not unwind past the join --
        // it would leave them dereferencing a dead closure. Rethrow after
        // the batch has drained.
        std::exception_ptr failure;
        try {
            body(0);
        } catch (...) {
            failure = std::current_exception();
        }
        // Spin briefly before blocking: at decode the helpers finish
        // within microseconds of the caller, and a condition-variable
        // round trip is the same order as the work itself.
        for (int i = 0; i < 4096 && m_pending.load(std::memory_order_acquire); i++) {
            std::this_thread::yield();
        }
        if (m_pending.load(std::memory_order_acquire) != 0) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_done.wait(lock, [&] { return m_pending.load(std::memory_order_acquire) == 0; });
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_body = nullptr;
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

    int size() const { return m_size + 1; }

private:
    moe_cpu_pool() {
        // Two cores back for the SYCL runtime and the server threads; the
        // caller is one of the workers, so the pool holds one fewer.
        int n = (int) std::thread::hardware_concurrency();
        const char * env = getenv("GGML_MOE_HYBRID_THREADS");
        if (env && env[0]) {
            n = atoi(env);
        } else {
            n = std::max(1, n - 2);
        }
        m_size = std::max(0, n - 1);
        m_workers.reserve((size_t) m_size);
        for (int i = 0; i < m_size; i++) {
            m_workers.emplace_back(&moe_cpu_pool::worker_main, this, i + 1);
        }
    }

    ~moe_cpu_pool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop.store(true, std::memory_order_release);
            m_generation.fetch_add(1, std::memory_order_release);
        }
        m_wake.notify_all();
        for (auto & t : m_workers) {
            if (t.joinable()) t.join();
        }
    }

    void worker_main(int id) {
        // Starts at 0, not at the current generation. A worker whose
        // thread had not begun running by the time the first batch was
        // published would otherwise adopt that batch's generation as
        // already-seen and sleep through it -- and since run() waits for
        // every helper to acknowledge, the batch would never complete.
        // Batches cannot be skipped either way: run() holds m_batch until
        // the previous one has drained, so the generation only advances
        // once every worker has taken the one before it.
        uint64_t seen = 0;
        for (;;) {
            uint64_t gen = m_generation.load(std::memory_order_acquire);
            for (int i = 0; i < 4096 && gen == seen; i++) {
                std::this_thread::yield();
                gen = m_generation.load(std::memory_order_acquire);
            }
            if (gen == seen) {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait(lock, [&] {
                    return m_generation.load(std::memory_order_acquire) != seen;
                });
                gen = m_generation.load(std::memory_order_acquire);
            }
            seen = gen;
            if (m_stop.load(std::memory_order_acquire)) {
                return;
            }
            // Read under the lock: run() publishes m_body and m_helpers
            // holding it, and a worker for a batch it is not part of must
            // not touch either.
            const std::function<void(int)> * body = nullptr;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_body && id <= m_helpers) {
                    body = m_body;
                }
            }
            if (!body) {
                continue;
            }
            (*body)(id);
            if (m_pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_done.notify_all();
            }
        }
    }

    std::mutex               m_batch;   // one batch at a time
    std::mutex               m_mutex;
    std::condition_variable  m_wake;
    std::condition_variable  m_done;
    std::vector<std::thread> m_workers;
    const std::function<void(int)> * m_body = nullptr;
    std::atomic<uint64_t>    m_generation{0};
    std::atomic<int>         m_pending{0};
    int                      m_helpers = 0;
    int                      m_size    = 0;
    // Atomic because workers read it outside m_mutex, right after seeing
    // a new generation. One plain bool written at teardown would still be
    // a race, and a race the sanitizers in ci/checks.sh do not look for.
    std::atomic<bool>        m_stop{false};
};

int moe_cpu_tier_max_threads() {
    return moe_cpu_pool::instance().size();
}

// The quantized kernel, when this build and this type have one.
//
// ggml-cpu's vec_dot reads a quantized weight row against a quantized
// activation and never materializes either as f32. That is the whole
// point: the profile of the dequantize-then-dot path on the 122B target
// (IQ2_XXS, 3072x1024) spent 67% of its thread time in to_float alone,
// making the miss tier cost more than the PCIe transfer it exists to
// avoid.
//
// The activation is quantized once per (activation, projection) into
// vec_dot_type -- Q8_K for the IQ2 family -- and reused across every
// output row of every expert that shares it.
struct moe_cpu_kernel {
    bool           usable       = false;
    bool           quantize_act = false;  // false when vec_dot eats f32 directly
    enum ggml_type act_type     = GGML_TYPE_F32;
    size_t         act_bytes    = 0;
#ifdef GGML_MOE_HYBRID_CPU_KERNELS
    ggml_vec_dot_t    vec_dot   = nullptr;
    ggml_from_float_t from_float = nullptr;
#endif
};

// Opt-out for A/B measurement and for bisecting a numerical difference
// back to this change; the dequantize path stays compiled either way.
static bool moe_cpu_kernels_disabled() {
    static const bool off = [] {
        const char * v = getenv("GGML_MOE_HYBRID_NO_VEC_DOT");
        return v && v[0] && v[0] != '0';
    }();
    return off;
}

static moe_cpu_kernel moe_cpu_kernel_for(enum ggml_type wtype, int64_t ne00) {
    moe_cpu_kernel k;
#ifdef GGML_MOE_HYBRID_CPU_KERNELS
    if (moe_cpu_kernels_disabled()) {
        return k;
    }
    // The kernels are unusable until ggml-cpu has detected its feature
    // set: before that every quantized vec_dot returns exactly 0.0, and
    // silently, so nothing downstream can tell wrong output from a token
    // the model happened to pick. Normally llama.cpp has already called
    // this by way of registering the CPU backend; a host-only test or
    // bench that links the tier directly has not.
    static const bool cpu_ready = [] { ggml_cpu_init(); return true; }();
    (void) cpu_ready;

    const struct ggml_type_traits_cpu * wt = ggml_get_type_traits_cpu(wtype);
    if (!wt || !wt->vec_dot) {
        return k;
    }
    const enum ggml_type vdt = wt->vec_dot_type;
    // A row that is not a whole number of blocks would make from_float
    // write past the buffer and vec_dot read past the row.
    if (ne00 % ggml_blck_size(wtype) != 0 || ne00 % ggml_blck_size(vdt) != 0) {
        return k;
    }
    k.vec_dot   = wt->vec_dot;
    k.act_type  = vdt;
    k.act_bytes = ggml_row_size(vdt, ne00);
    if (vdt == GGML_TYPE_F32) {
        // vec_dot takes the activation as-is; quantizing would be a copy.
        k.quantize_act = false;
        k.usable = true;
        return k;
    }
    const struct ggml_type_traits_cpu * at = ggml_get_type_traits_cpu(vdt);
    if (!at || !at->from_float) {
        return k;
    }
    k.from_float   = at->from_float;
    k.quantize_act = true;
    k.usable       = true;
#else
    (void) wtype;
    (void) ne00;
#endif
    return k;
}

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

// The gemv core. Computes output rows [row_begin, row_end) only, so a
// caller can split one expert across workers; timing, when non-null, is
// the calling worker's own accumulator.
//
// activation is the f32 row; prepared_act is that row in vec_dot_type,
// already quantized by the caller (which does it once for every expert
// sharing the activation). One of the two is used, never both.
static bool moe_cpu_expert_gemv_rows(
    const void *  expert_weights,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    const float * activation,
    const void *  prepared_act,
    const moe_cpu_kernel & kernel,
    float *       out,
    int64_t       row_begin,
    int64_t       row_end,
    std::vector<float> & row_f32,
    moe_cpu_row_timing * timing)
{
    if (!expert_weights || !activation || !out || ne00 <= 0 || ne01 <= 0 ||
        !moe_cpu_can_execute(wtype)) {
        return false;
    }
    row_begin = std::max<int64_t>(row_begin, 0);
    row_end   = std::min<int64_t>(row_end, ne01);

    const enum ggml_type type = (enum ggml_type) wtype;
    const size_t row_bytes = ggml_row_size(type, ne00);

#ifdef GGML_MOE_HYBRID_CPU_KERNELS
    if (kernel.usable) {
        const void * act = kernel.quantize_act ? prepared_act : (const void *) activation;
        if (!act) {
            return false;
        }
        // No dequantization happens here at all, so there is no dequant
        // time to attribute: the whole row is the kernel.
        const int64_t t0 = timing ? moe_now_ns() : 0;
        for (int64_t o = row_begin; o < row_end; o++) {
            const void * wrow = (const uint8_t *) expert_weights + (size_t) o * row_bytes;
            kernel.vec_dot((int) ne00, &out[o], 0, wrow, 0, act, 0, 1);
        }
        if (timing) {
            timing->matmul_ns += moe_now_ns() - t0;
            timing->rows      += row_end - row_begin;
        }
        return true;
    }
#else
    (void) prepared_act;
    (void) kernel;
#endif

    const bool is_f32 = (type == GGML_TYPE_F32);
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!is_f32 && (int64_t) row_f32.size() < ne00) {
        row_f32.resize((size_t) ne00);
    }
    for (int64_t o = row_begin; o < row_end; o++) {
        const void * wrow = (const uint8_t *) expert_weights + (size_t) o * row_bytes;
        const float * frow;
        const int64_t t0 = timing ? moe_now_ns() : 0;
        if (is_f32) {
            frow = (const float *) wrow;
        } else {
            traits->to_float(wrow, row_f32.data(), ne00);
            frow = row_f32.data();
        }
        const int64_t t1 = timing ? moe_now_ns() : 0;
        double acc = 0.0;
        for (int64_t j = 0; j < ne00; j++) {
            acc += (double) frow[j] * (double) activation[j];
        }
        out[o] = (float) acc;
        if (timing) {
            const int64_t t2 = moe_now_ns();
            timing->dequant_ns += t1 - t0;
            timing->matmul_ns  += t2 - t1;
            timing->rows++;
        }
    }
    return true;
}

bool moe_cpu_expert_gemv(
    const void *  expert_weights,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    const float * activation,
    float *       out)
{
    if (!activation || ne00 <= 0 || !moe_cpu_can_execute(wtype)) {
        return false;
    }
    const moe_cpu_kernel kernel =
        moe_cpu_kernel_for((enum ggml_type) wtype, ne00);
    std::vector<uint8_t> qact;
    const void * prepared = nullptr;
#ifdef GGML_MOE_HYBRID_CPU_KERNELS
    if (kernel.usable && kernel.quantize_act) {
        qact.resize(kernel.act_bytes);
        kernel.from_float(activation, qact.data(), ne00);
        prepared = qact.data();
    }
#endif
    std::vector<float> row_f32;
    return moe_cpu_expert_gemv_rows(expert_weights, wtype, ne00, ne01,
                                    activation, prepared, kernel, out,
                                    0, ne01, row_f32, nullptr);
}

bool moe_cpu_execute_gemvs(
    const moe_cpu_gemv_job * jobs,
    size_t        n_jobs,
    int           wtype,
    int64_t       ne00,
    int64_t       ne01,
    int           n_threads,
    moe_cpu_tier_counters * stats)
{
    if (!jobs || n_jobs == 0 || !moe_cpu_can_execute(wtype)) {
        return n_jobs == 0 && moe_cpu_can_execute(wtype);
    }
    for (size_t k = 0; k < n_jobs; k++) {
        if (!jobs[k].weights || !jobs[k].activation || !jobs[k].out) {
            return false;
        }
    }

    const bool profile = stats && moe_cpu_profile_enabled();
    const int64_t t_call = stats ? moe_now_ns() : 0;

    const moe_cpu_kernel kernel = moe_cpu_kernel_for((enum ggml_type) wtype, ne00);

    // Quantize each distinct activation once. At decode every expert in
    // the batch shares one activation, so the dedup is the difference
    // between one from_float and n_jobs of them; the pointers come
    // straight from the caller's packed buffer, so equal pointer means
    // equal data by construction.
    std::vector<uint8_t> act_store;
    std::vector<const void *> job_act((size_t) n_jobs, nullptr);
    int64_t quant_act_ns = 0;
#ifdef GGML_MOE_HYBRID_CPU_KERNELS
    if (kernel.usable && kernel.quantize_act) {
        const int64_t t_q0 = stats ? moe_now_ns() : 0;
        std::vector<const float *> seen;
        std::vector<size_t> seen_slot;
        seen.reserve(n_jobs);
        seen_slot.reserve(n_jobs);
        act_store.resize(kernel.act_bytes * n_jobs);
        for (size_t k = 0; k < n_jobs; k++) {
            size_t slot = seen.size();
            for (size_t s = 0; s < seen.size(); s++) {
                if (seen[s] == jobs[k].activation) { slot = seen_slot[s]; break; }
            }
            if (slot == seen.size()) {
                slot = seen.size();
                kernel.from_float(jobs[k].activation,
                                  act_store.data() + slot * kernel.act_bytes, ne00);
                seen.push_back(jobs[k].activation);
                seen_slot.push_back(slot);
            }
            job_act[k] = act_store.data() + slot * kernel.act_bytes;
        }
        if (stats) {
            quant_act_ns = moe_now_ns() - t_q0;
        }
    }
#endif

    // Slice by output row, not by job. One job per worker capped the tier
    // at n_jobs threads -- four, on a 26-thread machine, for a decode step
    // that missed four experts. Every output row still has exactly one
    // writer, so the result stays identical to sequential execution
    // whatever order the slices run in.
    //
    // Slices are at least a few rows wide so a worker amortizes its share
    // of the weight stream rather than ping-ponging cache lines with its
    // neighbours.
    const int pool_size = std::max(1, std::min(n_threads, moe_cpu_pool::instance().size()));
    struct row_slice { size_t job; int64_t begin; int64_t end; };
    std::vector<row_slice> slices;
    {
        const int64_t min_rows = 32;
        int64_t per_job = (ne01 * (int64_t) n_jobs + pool_size - 1) / pool_size;
        per_job = std::max(per_job, min_rows);
        slices.reserve(n_jobs * (size_t) ((ne01 + per_job - 1) / per_job));
        for (size_t k = 0; k < n_jobs; k++) {
            for (int64_t b = 0; b < ne01; b += per_job) {
                slices.push_back({k, b, std::min(ne01, b + per_job)});
            }
        }
    }
    const int workers = std::max(1, std::min<int>(pool_size, (int) slices.size()));

    std::atomic<bool> ok{true};
    std::atomic<size_t> next_slice{0};
    std::vector<moe_cpu_row_timing> timing_workers((size_t) workers);
    // Dispatch is timed from here, not from entry: activation
    // quantization is real work and has its own counter.
    const int64_t t_prep = stats ? moe_now_ns() : 0;
    std::atomic<int64_t> t_first_helper{0};

    auto worker = [&](int id) {
        if (stats && id == 1) {
            int64_t expected = 0;
            t_first_helper.compare_exchange_strong(expected, moe_now_ns());
        }
        moe_cpu_row_timing * timing =
            profile ? &timing_workers[(size_t) std::min(id, workers - 1)] : nullptr;
        std::vector<float> row_f32;
        // Claim slices dynamically: an expert whose rows are already
        // resident in page cache finishes far faster than one still
        // faulting from storage, and a static split would leave whoever
        // drew the cold one running alone.
        for (;;) {
            const size_t s = next_slice.fetch_add(1, std::memory_order_relaxed);
            if (s >= slices.size() || !ok.load(std::memory_order_relaxed)) {
                break;
            }
            const row_slice & sl = slices[s];
            if (!moe_cpu_expert_gemv_rows(jobs[sl.job].weights, wtype, ne00, ne01,
                                          jobs[sl.job].activation, job_act[sl.job],
                                          kernel, jobs[sl.job].out,
                                          sl.begin, sl.end, row_f32, timing)) {
                ok.store(false, std::memory_order_relaxed);
            }
        }
    };

    if (workers == 1) {
        worker(0);
    } else {
        moe_cpu_pool::instance().run(workers, worker);
    }
    const int64_t t_dispatched = t_first_helper.load(std::memory_order_relaxed);

    if (stats) {
        int64_t weight_rows = (int64_t) n_jobs * ne01;
        moe_counter_add(stats->calls, 1);
        moe_counter_add(stats->jobs, (int64_t) n_jobs);
        moe_counter_add(stats->weight_rows, weight_rows);
        moe_counter_add(stats->kernel_rows, kernel.usable ? weight_rows : 0);
        moe_counter_add(stats->weight_bytes,
                        weight_rows * (int64_t) ggml_row_size((enum ggml_type) wtype, ne00));
        moe_counter_max(stats->threads_used, workers);
        moe_counter_add(stats->wall_ns, moe_now_ns() - t_call);
        // Zero when the batch ran on the caller alone; otherwise how long
        // the first helper took to pick the work up.
        moe_counter_add(stats->dispatch_ns, t_dispatched ? t_dispatched - t_prep : 0);
        moe_counter_add(stats->quant_act_ns, quant_act_ns);
        if (profile) {
            moe_cpu_row_timing total;
            for (const auto & t : timing_workers) {
                total.dequant_ns += t.dequant_ns;
                total.matmul_ns  += t.matmul_ns;
                total.rows       += t.rows;
            }
            moe_counter_add(stats->dequant_ns, total.dequant_ns);
            moe_counter_add(stats->matmul_ns, total.matmul_ns);
            moe_counter_add(stats->profiled_rows, total.rows);
        }
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
