// Resolver for the MoE expert-weight cache's versioned backend-registry
// procs (see ggml/include/ggml-backend.h for the typedefs and lookup
// names, and ggml/src/ggml-sycl/ggml-sycl.cpp for the implementation).
//
// The server used to write directly to `extern volatile` globals declared
// weak in ggml-sycl.cpp. That only works when the SYCL backend is statically
// linked into the same executable: a backend loaded via GGML_BACKEND_DL=ON
// lives in a separately dlopen'd .so, and a weak extern in the main
// executable does not resolve to a symbol that only exists in a plugin
// loaded at runtime. Going through ggml_backend_reg_get_proc_address
// instead works identically for static and dynamic backends, because the
// backend registry itself is populated the same way in both cases.
//
// A build with no cache-capable backend registered (CPU-only, or a SYCL
// build without GGML_MOE_EXPERT_CACHE compiled in) simply has no reg that
// resolves these names -- every function pointer below stays null, and
// every caller here already treats null as "feature not available."
#ifndef MOE_CACHE_IFACE_H
#define MOE_CACHE_IFACE_H

#include "ggml-backend.h"

struct moe_cache_procs {
    ggml_backend_moe_cache_configure_t  configure       = nullptr;
    ggml_backend_moe_cache_reset_all_t  reset_all       = nullptr;
    ggml_backend_moe_cache_set_phase_t  set_phase       = nullptr;
    ggml_backend_moe_cache_stats_json_t stats_json      = nullptr;
    ggml_backend_moe_hybrid_set_mode_t  hybrid_set_mode = nullptr;

    bool available() const { return configure != nullptr; }
};

// Resolved once (backend regs don't change after ggml_backend_load_all())
// and cached for the process lifetime.
inline const moe_cache_procs & moe_cache_get_procs() {
    static const moe_cache_procs procs = [] {
        moe_cache_procs p;
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            auto * configure = (ggml_backend_moe_cache_configure_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cache_configure_v1");
            if (!configure) {
                continue;
            }
            p.configure       = configure;
            p.reset_all       = (ggml_backend_moe_cache_reset_all_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cache_reset_all_v1");
            p.set_phase       = (ggml_backend_moe_cache_set_phase_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cache_set_phase_v1");
            p.stats_json      = (ggml_backend_moe_cache_stats_json_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cache_stats_json_v1");
            p.hybrid_set_mode = (ggml_backend_moe_hybrid_set_mode_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_hybrid_set_mode_v1");
            break; // first backend reg that implements the cache wins
        }
        return p;
    }();
    return procs;
}

#endif // MOE_CACHE_IFACE_H
