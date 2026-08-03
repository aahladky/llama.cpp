// MoE Expert Cache — device operations.
//
// The only translation unit in the cache that includes the SYCL headers.
// moe-cache.cpp holds the policy (admission, eviction, phase, reset) and
// calls through these functions, so that policy compiles as plain C++
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

// ---- async admission fills ---------------------------------------------
//
// A second in-order queue on the same device+context carries the fill
// copies, so no fill ever stands in front of compute on the main queue.
// Ordering against compute is one-directional and explicit: each fill
// depends on a barrier event captured on the compute queue at submission
// time, which orders it after every kernel or D2D hit copy that might
// still read the slot's previous contents. Nothing on the compute queue
// ever waits on a fill -- a slot only becomes servable after its event
// reports complete, checked host-side at step end.

void * moe_cache_device_create_transfer_queue(void * compute_queue) {
    if (!compute_queue) {
        return nullptr;
    }
    queue_ptr cq = static_cast<queue_ptr>(compute_queue);
    try {
        sycl::queue * tq = new sycl::queue(cq->get_context(), cq->get_device(),
                                           sycl::property::queue::in_order{});
        return tq;
    } catch (const sycl::exception & e) {
        fprintf(stderr, "moe_cache: transfer queue creation failed: %s\n", e.what());
        return nullptr;
    }
}

void moe_cache_device_destroy_transfer_queue(void * transfer_queue) {
    if (!transfer_queue) {
        return;
    }
    sycl::queue * q = static_cast<sycl::queue *>(transfer_queue);
    try {
        // Drain in-flight fills so nothing writes into a pool the caller
        // is about to free.
        q->wait();
    } catch (...) {
        // Destructor path: a throw here would terminate.
    }
    delete q;
}

void * moe_cache_device_copy_async_after(void * dst, const void * src, size_t bytes,
                                         void * transfer_queue, void * compute_queue) {
    if (!dst || !src || bytes == 0 || !transfer_queue || !compute_queue) {
        return nullptr;
    }
    sycl::queue * tq = static_cast<sycl::queue *>(transfer_queue);
    queue_ptr     cq = static_cast<queue_ptr>(compute_queue);
    try {
        sycl::event barrier = cq->ext_oneapi_submit_barrier();
        sycl::event ev = tq->submit([&](sycl::handler & h) {
            h.depends_on(barrier);
            h.memcpy(dst, src, bytes);
        });
        return new sycl::event(std::move(ev));
    } catch (const sycl::exception & e) {
        fprintf(stderr, "moe_cache: async fill submission failed: %s\n", e.what());
        return nullptr;
    }
}

bool moe_cache_device_event_complete(void * ev) {
    if (!ev) {
        return true;
    }
    sycl::event * e = static_cast<sycl::event *>(ev);
    try {
        return e->get_info<sycl::info::event::command_execution_status>()
               == sycl::info::event_command_status::complete;
    } catch (...) {
        // If the status query itself fails, treat the fill as done rather
        // than pinning the slot in pending forever; the seq/occupancy
        // checks at retirement keep this safe.
        return true;
    }
}

void moe_cache_device_event_free(void * ev) {
    delete static_cast<sycl::event *>(ev);
}
