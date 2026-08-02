// MoE execution fingerprint — a per-op record of WHERE and HOW each routed
// MUL_MAT_ID executed.
//
// This exists to answer one question: when two identical runs of the same
// binary on the same config produce different tokens, do their ops execute
// the same way?  If two runs' fingerprints differ, the nondeterminism is in
// the execution path (which expert was served from a slot, which was staged,
// which ran on the CPU tier, what geometry was learned, when) and not in the
// arithmetic.  If the fingerprints match and the logits still differ, the
// arithmetic is where to look next.
//
// Entirely off unless GGML_MOE_FINGERPRINT names a file.  Every entry point
// begins with an inlined check of one cached flag, so a production build with
// the variable unset pays a predictable-branch per call and nothing else.
//
// Record types, one per line, space separated:
//   V  <version> <pid>                                      -- header
//   G  <step> <dev> <gate> <up> <down> <slots> <slot_bytes> <pool_id>
//                                                           -- geometry finalized
//   S  <step> <dev> <layer> <proj> <expert> <disp> <base_id> <host_id>
//                                                           -- one staging decision
//   O  <step> <dev> <layer> <proj> <ne00> <ne01> <ne11> <ne12> <path>
//      <plan_n> <cpu_rows> <gpu_rows> <reorder> <src0_id> <ids_hash>
//                                                           -- one routed op
//   D  <step> <dev> <layer> <proj> <hash>                   -- tensor hash,
//      only under GGML_MOE_FINGERPRINT_DST; proj >= 0 is the op's output,
//      -1-p is projection p's staged weights over the used experts, -10-p is
//      its activations, -20-p is the used-expert count
//   E  <step> <dev> <hits> <misses> <evict> <promo> <fallback> <served>
//      <skips> <slots_used>                                  -- step end
//   X  <step> <text>                                         -- free-form event
//
// Pointers are never printed raw: each distinct pointer is mapped to the
// ordinal of its first appearance in this run.  Two runs that touch the same
// objects in the same order therefore produce byte-identical fingerprints,
// while an allocator that hands back different addresses does not by itself
// manufacture a diff.  Raw addresses are available separately via
// GGML_MOE_FINGERPRINT_RAW=1 for the cases where the address itself is the
// question.

#ifndef GGML_SYCL_MOE_FINGERPRINT_HPP
#define GGML_SYCL_MOE_FINGERPRINT_HPP

#include <cstddef>
#include <cstdint>

// Staging dispositions, in the order moe_cache_hook_copy can reach them.
#define MOE_FP_DISP_NOT_ROUTED   'n'  // name did not parse as a routed expert
#define MOE_FP_DISP_LEARNING     'l'  // learning pass: observed, staged plainly
#define MOE_FP_DISP_HIT          'h'  // served from a cache slot (D2D)
#define MOE_FP_DISP_PROMOTE      'p'  // miss, promoted into a slot, then D2D
#define MOE_FP_DISP_HYBRID_SKIP  'k'  // miss, declined, staging skipped for CPU tier
#define MOE_FP_DISP_FALLBACK     'f'  // miss, declined, scheduler stages from host

// True when GGML_MOE_FINGERPRINT is set to a writable path.  Cached; safe to
// call on the hot path.
bool moe_fp_enabled();

// One staging decision for one (layer, projection, expert).
void moe_fp_stage(int dev, int layer, int proj, int32_t expert, char disp,
                  const void * cpy_base, const void * host_src);

// Geometry finalized (the learning window closed).  Emitted on the edge, so
// the step number is when it happened.
void moe_fp_geom(int dev, size_t gate, size_t up, size_t down,
                 int slots, size_t slot_bytes, const void * pool);

// One routed MUL_MAT_ID execution.  `path` is a short literal naming the
// branch taken ("mmvq_fused", "rowwise", "batched").
//
// `ids_hash` is the decisive field.  It is a hash of the routing tensor --
// WHICH experts this op was told to use.  The router derives it from the
// hidden state, so the first op whose ids_hash differs between two runs is
// the first place a numeric difference became visible, and every op before
// it agreed.  Unlike the staging records it does not need a cache to exist,
// which is what makes a no-cache run comparable to a cache run.
void moe_fp_op(int dev, int layer, int proj,
               int64_t ne00, int64_t ne01, int64_t ne11, int64_t ne12,
               const char * path, int plan_n, int cpu_rows, int gpu_rows,
               int reorder, const void * src0_data, uint64_t ids_hash);

// FNV-1a over a byte range.  Exposed so callers hash on the thread that
// already holds the data, and only when the fingerprint is on.
uint64_t moe_fp_hash(const void * data, size_t bytes);

// True when GGML_MOE_FINGERPRINT_DST=1: additionally hash each routed op's
// OUTPUT.  This forces a device sync per op and is far too slow for a
// battery -- it exists for single-request localization, where knowing the
// first op whose output differs is worth the cost.
bool moe_fp_dst_enabled();

// One routed op's output hash (see moe_fp_dst_enabled).
void moe_fp_dst(int dev, int layer, int proj, uint64_t dst_hash);

// End of one graph compute, for one device: the cache's cumulative totals at
// that moment.  Called once per device that has a cache.
void moe_fp_step_end(int dev, uint64_t hits, uint64_t misses, uint64_t evictions,
                     uint64_t promotions, uint64_t fallbacks, uint64_t served,
                     uint64_t staging_skips, int slots_used);

// Close the step.  Called once per completed graph, after every device's
// moe_fp_step_end, so one graph is one step no matter how many devices have
// caches.
void moe_fp_step_advance();

// Free-form event, for anything that happens once (reset, phase change).
void moe_fp_event(const char * text);

// Current step index (graph computes completed).  Cheap; for callers that
// want to correlate their own logging.
uint64_t moe_fp_step();

#endif // GGML_SYCL_MOE_FINGERPRINT_HPP
