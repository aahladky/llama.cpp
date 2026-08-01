// Microbenchmark for the hybrid MoE CPU miss path.
//
// The miss path is the one piece of hybrid execution whose cost decides
// the feature: it replaces an expert's host-to-device transfer with host
// compute, so it wins only if computing is cheaper than transferring.
// Measuring it inside a 34 GiB server run takes ten minutes per data
// point; this takes a second, over the same geometry and -- when a GGUF
// is supplied -- the same bytes.
//
// Default geometry is Qwen3.5-122B-A10B (UD-IQ1_M file, IQ2_XS expert
// tensors): 3072 x 1024 gate/up and 1024 x 3072 down, 256 experts, 8
// routed. Weights come from the model file if --model is given, and from
// a quantization of random f32 data otherwise, so the bench runs on any
// machine.
//
// Usage:
//   bench-moe-hybrid [--model FILE] [--tensor NAME] [--experts N]
//                    [--threads N] [--reps N] [--check]

#include "ggml-sycl/moe-hybrid.hpp"

#include "ggml.h"

#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// The policy side of moe-cache links here because moe-hybrid.cpp calls
// it; the device side does not, and the bench never builds a cache.
// Same aborting stubs as test-moe-hybrid.cpp, for the same reason.
static void device_path_unreachable(const char * what) {
    fprintf(stderr, "bench: reached the device path (%s)\n", what);
    abort();
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif
void * moe_cache_device_alloc(size_t, void *) { device_path_unreachable("alloc"); return nullptr; }
void   moe_cache_device_free(void *, void *) { device_path_unreachable("free"); }
bool   moe_cache_device_copy(void *, const void *, size_t, void *) { device_path_unreachable("copy"); return false; }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// ---------------------------------------------------------------------
// GGUF: just enough of the header to find one tensor's bytes. Reading the
// real weights matters because quantized kernels are data-dependent (an
// all-zero or uniform-random block is not the same work as a real one).

struct gguf_lite {
    int          fd    = -1;
    const uint8_t * map = nullptr;
    size_t       size  = 0;
    size_t       data_offset = 0;
    struct entry { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
    std::vector<entry> tensors;

    ~gguf_lite() {
        if (map) munmap(const_cast<uint8_t *>(map), size);
        if (fd >= 0) close(fd);
    }
};

namespace {

struct reader {
    const uint8_t * p;
    const uint8_t * end;
    bool ok = true;

    bool take(size_t n, const uint8_t ** out) {
        if (!ok || (size_t)(end - p) < n) { ok = false; return false; }
        *out = p; p += n; return true;
    }
    template <typename T> bool u(T * out) {
        const uint8_t * q;
        if (!take(sizeof(T), &q)) return false;
        memcpy(out, q, sizeof(T)); return true;
    }
    bool str(std::string * out) {
        uint64_t n;
        if (!u(&n)) return false;
        const uint8_t * q;
        if (n > (uint64_t)(end - p) || !take((size_t) n, &q)) { ok = false; return false; }
        if (out) out->assign((const char *) q, (size_t) n);
        return true;
    }
};

// Scalar sizes for GGUF metadata value types, indexed by type id.
// Type 8 is string and type 9 is array; both are handled separately.
const size_t kv_scalar_size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

bool skip_value(reader & r, uint32_t type) {
    if (type == 8) return r.str(nullptr);
    if (type == 9) {
        uint32_t et; uint64_t n;
        if (!r.u(&et) || !r.u(&n)) return false;
        if (et == 8) {
            for (uint64_t i = 0; i < n && r.ok; i++) r.str(nullptr);
            return r.ok;
        }
        if (et > 12 || kv_scalar_size[et] == 0) { r.ok = false; return false; }
        const uint8_t * q;
        return r.take(kv_scalar_size[et] * (size_t) n, &q);
    }
    if (type > 12 || kv_scalar_size[type] == 0) { r.ok = false; return false; }
    const uint8_t * q;
    return r.take(kv_scalar_size[type], &q);
}

bool gguf_open(const char * path, gguf_lite & g) {
    g.fd = open(path, O_RDONLY);
    if (g.fd < 0) { fprintf(stderr, "bench: cannot open %s\n", path); return false; }
    struct stat st;
    if (fstat(g.fd, &st) != 0) { fprintf(stderr, "bench: cannot stat %s\n", path); return false; }
    g.size = (size_t) st.st_size;
    void * m = mmap(nullptr, g.size, PROT_READ, MAP_SHARED, g.fd, 0);
    if (m == MAP_FAILED) { fprintf(stderr, "bench: cannot mmap %s\n", path); return false; }
    g.map = (const uint8_t *) m;

    reader r{g.map, g.map + g.size};
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;
    if (!r.u(&magic) || !r.u(&version) || !r.u(&n_tensors) || !r.u(&n_kv)) return false;
    if (memcmp(&magic, "GGUF", 4) != 0) { fprintf(stderr, "bench: not a GGUF file\n"); return false; }

    for (uint64_t i = 0; i < n_kv; i++) {
        uint32_t type;
        if (!r.str(nullptr) || !r.u(&type) || !skip_value(r, type)) {
            fprintf(stderr, "bench: GGUF metadata parse failed\n"); return false;
        }
    }
    g.tensors.reserve((size_t) n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_lite::entry e;
        uint32_t nd;
        if (!r.str(&e.name) || !r.u(&nd) || nd > 4) { r.ok = false; break; }
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t v;
            if (!r.u(&v)) break;
            e.dims.push_back(v);
        }
        if (!r.u(&e.type) || !r.u(&e.off)) { r.ok = false; break; }
        g.tensors.push_back(std::move(e));
    }
    if (!r.ok) { fprintf(stderr, "bench: GGUF tensor table parse failed\n"); return false; }

    // Tensor data starts at the next general.alignment boundary. The
    // default alignment is 32 and every model here uses it; a file that
    // set something else would land the reads off the block boundary,
    // so check rather than assume.
    const size_t align = 32;
    const size_t here = (size_t)(r.p - g.map);
    g.data_offset = (here + align - 1) / align * align;
    return true;
}

double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char * type_name(enum ggml_type t) {
    const struct ggml_type_traits * tr = ggml_get_type_traits(t);
    return tr && tr->type_name ? tr->type_name : "?";
}

} // namespace

int main(int argc, char ** argv) {
    const char * model  = nullptr;
    const char * tensor = "blk.0.ffn_gate_exps.weight";
    int  n_experts = 4;     // misses per op at the measured ~50% hit ratio
    int  n_threads = 0;     // 0 = the tier's own default
    int  reps      = 20;
    bool check     = false;
    int64_t ne00 = 3072, ne01 = 1024;
    enum ggml_type wtype = GGML_TYPE_IQ2_XS;

    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "bench: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if      (!strcmp(a, "--model"))   model     = next(a);
        else if (!strcmp(a, "--tensor"))  tensor    = next(a);
        else if (!strcmp(a, "--experts")) n_experts = atoi(next(a));
        else if (!strcmp(a, "--threads")) n_threads = atoi(next(a));
        else if (!strcmp(a, "--reps"))    reps      = atoi(next(a));
        else if (!strcmp(a, "--ne00"))    ne00      = atoll(next(a));
        else if (!strcmp(a, "--ne01"))    ne01      = atoll(next(a));
        else if (!strcmp(a, "--check"))   check     = true;
        else { fprintf(stderr, "bench: unknown argument %s\n", a); return 2; }
    }
    if (n_experts <= 0 || reps <= 0 || ne00 <= 0 || ne01 <= 0) {
        fprintf(stderr, "bench: --experts/--reps/--ne00/--ne01 must be positive\n");
        return 2;
    }
    if (n_threads <= 0) {
        n_threads = std::max(1, (int) std::thread::hardware_concurrency() - 2);
    }

    // ---------------- weights ----------------
    gguf_lite g;
    std::vector<uint8_t> synth;
    const uint8_t * weights_base = nullptr;
    size_t expert_stride = 0;
    std::string source;

    if (model) {
        if (!gguf_open(model, g)) return 1;
        const gguf_lite::entry * e = nullptr;
        for (const auto & t : g.tensors) {
            if (t.name == tensor) { e = &t; break; }
        }
        if (!e) { fprintf(stderr, "bench: tensor %s not found\n", tensor); return 1; }
        if (e->dims.size() != 3) {
            fprintf(stderr, "bench: %s is not a 3-D expert tensor\n", tensor); return 1;
        }
        wtype = (enum ggml_type) e->type;
        ne00  = (int64_t) e->dims[0];
        ne01  = (int64_t) e->dims[1];
        const int64_t n_expert_total = (int64_t) e->dims[2];
        if (n_experts > n_expert_total) n_experts = (int) n_expert_total;
        expert_stride = (size_t) ne01 * ggml_row_size(wtype, ne00);
        const size_t need = g.data_offset + e->off + expert_stride * (size_t) n_experts;
        if (need > g.size) {
            fprintf(stderr, "bench: %s extends past end of file\n", tensor); return 1;
        }
        weights_base = g.map + g.data_offset + e->off;
        source = std::string("model ") + tensor;
    } else {
        // Quantize random f32 rows. IQ2_XS needs an importance matrix; a
        // flat one is not what a real quantizer would use, but the output
        // is a valid block stream of the right size, which is what the
        // timing depends on.
        expert_stride = (size_t) ne01 * ggml_row_size(wtype, ne00);
        synth.resize(expert_stride * (size_t) n_experts);
        std::vector<float> src((size_t) ne00 * (size_t) ne01);
        std::vector<float> imatrix((size_t) ne00, 1.0f);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (int e = 0; e < n_experts; e++) {
            for (auto & v : src) v = dist(rng);
            ggml_quantize_chunk(wtype, src.data(), synth.data() + (size_t) e * expert_stride,
                                0, ne01, ne00, imatrix.data());
        }
        weights_base = synth.data();
        source = "synthetic";
    }

    if (!moe_cpu_can_execute((int) wtype)) {
        fprintf(stderr, "bench: no CPU path for %s\n", type_name(wtype));
        return 1;
    }

    // ---------------- activations and jobs ----------------
    // One activation, every expert: that is decode. The tier still gets
    // one job per expert, which is what the real caller passes.
    std::vector<float> act((size_t) ne00);
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto & v : act) v = dist(rng);
    }
    std::vector<float> out((size_t) n_experts * (size_t) ne01);
    std::vector<moe_cpu_gemv_job> jobs((size_t) n_experts);
    for (int e = 0; e < n_experts; e++) {
        jobs[(size_t) e] = { weights_base + (size_t) e * expert_stride,
                             act.data(),
                             out.data() + (size_t) e * (size_t) ne01 };
    }

    const size_t bytes_per_call = (size_t) n_experts * expert_stride;

    printf("geometry     %s, ne00=%" PRId64 " ne01=%" PRId64 ", %d experts/op\n",
           type_name(wtype), ne00, ne01, n_experts);
    printf("weights      %s (%.2f MiB per op)\n", source.c_str(),
           bytes_per_call / 1048576.0);
    printf("threads      %d   reps %d   profile %s\n", n_threads, reps,
           moe_cpu_profile_enabled() ? "on" : "off");

    // Warm the pages: the first pass over a fresh mmap measures the
    // storage, not the kernel. Storage cost is real and is measured in
    // the server runs; here we want the compute.
    moe_cpu_tier_counters warm;
    if (!moe_cpu_execute_gemvs(jobs.data(), jobs.size(), (int) wtype, ne00, ne01,
                               n_threads, &warm)) {
        fprintf(stderr, "bench: CPU tier refused the batch\n");
        return 1;
    }

    std::vector<float> reference;
    if (check) {
        reference = out;
    }

    moe_cpu_tier_counters c;
    const double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        if (!moe_cpu_execute_gemvs(jobs.data(), jobs.size(), (int) wtype, ne00, ne01,
                                   n_threads, &c)) {
            fprintf(stderr, "bench: CPU tier refused the batch\n");
            return 1;
        }
    }
    const double t1 = now_s();

    const moe_cpu_tier_stats s = moe_cpu_tier_snapshot(c);
    const double per_call_ms = (t1 - t0) * 1000.0 / reps;
    const double rows = (double) s.weight_rows;

    printf("\n");
    printf("per op       %.3f ms  (%d experts, %" PRId64 " weight rows)\n",
           per_call_ms, n_experts, (int64_t) (rows / reps));
    printf("throughput   %.2f GiB/s of quantized weights\n",
           (double) bytes_per_call * reps / (t1 - t0) / 1073741824.0);
    printf("per row      %.0f ns\n", (t1 - t0) * 1e9 / rows);
    printf("dispatch     %.3f ms/op  (%.1f%% of the op)\n",
           (double) s.dispatch_ns / 1e6 / reps,
           100.0 * (double) s.dispatch_ns / ((t1 - t0) * 1e9));
    printf("threads used %" PRId64 "\n", s.threads_used);
    printf("path         %s (%" PRId64 "/%" PRId64 " rows through ggml-cpu kernels)\n",
           s.kernel_rows == s.weight_rows ? "quantized vec_dot"
                                          : (s.kernel_rows ? "mixed" : "dequantize-and-dot"),
           s.kernel_rows, s.weight_rows);

    // A decode step is every layer's gate, up and down projection.
    printf("\nprojected    %.1f ms per decode token at 48 layers x 3 projections\n",
           per_call_ms * 48 * 3);

    if (s.profiled_rows > 0) {
        const double p = (double) s.profiled_rows;
        printf("\nprofile      %" PRId64 " rows timed (%.0f%% of rows)\n",
               s.profiled_rows, 100.0 * p / rows);
        printf("  dequant    %8.0f ns/row   %5.1f%% of thread time\n",
               (double) s.dequant_ns / p,
               100.0 * (double) s.dequant_ns / (double)(s.dequant_ns + s.matmul_ns));
        printf("  matmul     %8.0f ns/row   %5.1f%% of thread time\n",
               (double) s.matmul_ns / p,
               100.0 * (double) s.matmul_ns / (double)(s.dequant_ns + s.matmul_ns));
        if (s.quant_act_ns > 0) {
            printf("  quant act  %8.3f ms total\n", (double) s.quant_act_ns / 1e6);
        }
    } else {
        printf("\nprofile      off (set GGML_MOE_HYBRID_PROFILE=1 for the ns breakdown)\n");
    }

    if (check) {
        // Against the warm-up pass, which ran the same code: this catches
        // a threading change that made results depend on the schedule,
        // not a change in the math itself.
        double max_abs = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < out.size(); i++) {
            const double d = std::fabs((double) out[i] - (double) reference[i]);
            max_abs = std::max(max_abs, d);
            const double m = std::fabs((double) reference[i]);
            if (m > 1e-6) max_rel = std::max(max_rel, d / m);
        }
        printf("\ncheck        max abs %.3g, max rel %.3g vs the warm-up pass%s\n",
               max_abs, max_rel, max_abs == 0.0 ? " (bit-identical)" : "");
        if (max_abs != 0.0) {
            fprintf(stderr, "bench: results are schedule-dependent\n");
            return 1;
        }
    }

    // What the fast path costs in accuracy. The dequantize-and-dot
    // reference is exact f32 weights against exact f32 activations; the
    // kernel path quantizes the activation to vec_dot_type. That is a
    // real numerical difference, and a decode path whose argmax depends
    // on it will change tokens -- so it gets measured, not assumed.
    {
        std::vector<float> ref_out((size_t) n_experts * (size_t) ne01);
        std::vector<float> row((size_t) ne00);
        const struct ggml_type_traits * traits = ggml_get_type_traits(wtype);
        const size_t row_bytes = ggml_row_size(wtype, ne00);
        for (int e = 0; e < n_experts; e++) {
            const uint8_t * ew = weights_base + (size_t) e * expert_stride;
            for (int64_t o = 0; o < ne01; o++) {
                if (wtype == GGML_TYPE_F32) {
                    memcpy(row.data(), ew + (size_t) o * row_bytes, (size_t) ne00 * sizeof(float));
                } else {
                    traits->to_float(ew + (size_t) o * row_bytes, row.data(), ne00);
                }
                double acc = 0.0;
                for (int64_t j = 0; j < ne00; j++) acc += (double) row[j] * (double) act[(size_t) j];
                ref_out[(size_t) e * (size_t) ne01 + (size_t) o] = (float) acc;
            }
        }
        double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0, ref_sq = 0.0;
        int64_t nonzero_ref = 0;
        for (size_t i = 0; i < out.size(); i++) {
            const double a = (double) out[i], b = (double) ref_out[i];
            const double d = std::fabs(a - b);
            max_abs = std::max(max_abs, d);
            if (std::fabs(b) > 1e-6) { max_rel = std::max(max_rel, d / std::fabs(b)); nonzero_ref++; }
            sum_sq += d * d; ref_sq += b * b;
        }
        double ref_min = 0.0, ref_max = 0.0, ref_absmean = 0.0;
        for (size_t i = 0; i < ref_out.size(); i++) {
            ref_min = std::min(ref_min, (double) ref_out[i]);
            ref_max = std::max(ref_max, (double) ref_out[i]);
            ref_absmean += std::fabs((double) ref_out[i]);
        }
        ref_absmean /= (double) ref_out.size();
        printf("\naccuracy     vs dequantize-and-dot over the same weights\n");
        printf("  reference  range [%.4g, %.4g], mean |v| %.4g, %" PRId64 "/%zu above 1e-6\n",
               ref_min, ref_max, ref_absmean, nonzero_ref, out.size());
        printf("  max abs    %.4g      max rel %.4g\n", max_abs, max_rel);
        printf("  rel RMS    %.4g\n", ref_sq > 0 ? std::sqrt(sum_sq / ref_sq) : 0.0);
        if (nonzero_ref == 0) {
            fprintf(stderr, "bench: reference is all zero -- the fixture says nothing\n");
            return 1;
        }
    }
    return 0;
}
