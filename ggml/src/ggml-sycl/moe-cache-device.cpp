// MoE Expert Cache — device operations.
//
// The only translation unit in the cache that includes the SYCL headers.
// moe-cache.cpp holds the policy (admission, eviction, phase, reset) and
// calls through these three functions, so that policy compiles as plain C++
// and tests/test-moe-cache.cpp can build and run on a
// machine with no GPU.
//
// Failures are reported by return value rather than by exception: the policy
// side has no SYCL headers and so cannot catch sycl::exception.

#include "moe-cache.hpp"
#include "common.hpp"

#include <cstdio>

void * moe_cache_device_alloc(size_t bytes, void * queue) {
    if (!queue || bytes == 0) {
        return nullptr;
    }
    queue_ptr q = static_cast<queue_ptr>(queue);
    void * ptr = nullptr;
    try {
        ptr = ggml_sycl_malloc_device(bytes, *q);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "moe_cache: pool allocation of %zu bytes failed: %s\n",
                bytes, e.what());
        return nullptr;
    }
    // USM allocators can return null on OOM without throwing, so the
    // try/catch above is not sufficient on its own.
    return ptr;
}

void moe_cache_device_free(void * ptr, void * queue) {
    if (!ptr || !queue) {
        return;
    }
    queue_ptr q = static_cast<queue_ptr>(queue);
    try {
        ggml_sycl_free_device(ptr, *q);
    } catch (...) {
        // Destructor path: a throw here would terminate.
    }
}

bool moe_cache_device_copy(void * dst, const void * src, size_t bytes, void * queue) {
    if (!dst || !src || !queue) {
        return false;
    }
    queue_ptr q = static_cast<queue_ptr>(queue);
    try {
        // Async copy: submit without blocking.  The copy is ordered on the
        // same queue as subsequent compute, so the GPU serializes it before
        // any kernel that touches this slot.  Never .wait() in the
        // steady-state loop -- that would stall the pipeline.
        auto ev = q->memcpy(dst, src, bytes);
        (void)ev;  // event available if a future consumer needs explicit sync
    } catch (const sycl::exception & e) {
        fprintf(stderr, "moe_cache: promote_projection failed: %s\n", e.what());
        return false;
    }
    return true;
}
