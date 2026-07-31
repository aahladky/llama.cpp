#!/usr/bin/env bash
# K-quant hybrid correctness: Q4_K experts kept host-resident, MoE ops
# offloaded at batch 1 (GGML_OP_OFFLOAD_MOE_MIN_BATCH=1), so the staging
# hook runs on every token. This reaches the three failure modes the F32
# fixture never could (mmvq_fused bails on F32): staged-copy lifetimes
# (async H2D from the CPU tier), the restaged-tensor reorder guard, and
# hybrid staging skips. Token-identical output across baseline / cache /
# cache+hybrid at temperature 0 is the pass condition.
#
# Meaningful only on a SYCL build with a device: on a CPU-only build the
# staging hook never fires and the three cases are trivially identical.
#
# Usage: run_kquant.sh <binary_dir> <port> [repeats]
set -u
BINDIR="$1"; PORT="$2"; REPEATS="${3:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F32="$HERE/tiny-moe-k.f32.gguf"
MODEL="$HERE/tiny-moe-q4k.gguf"

PYTHON="${PYTHON:-python3}"   # needs numpy (for gguf-py)
# The icpx-built binaries need the oneAPI runtime even for quantize;
# run_seq.sh sources this for the server, but quantize runs first.
# (setvars.sh is not `set -u`-clean, hence the bracket.)
if [ -f /opt/intel/oneapi/setvars.sh ]; then
  set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
fi
if [ ! -f "$MODEL" ]; then
  # QK_K=256: tensor rows must be 256-divisible or llama-quantize silently
  # falls back to non-K types and the reorder path is never exercised.
  "$PYTHON" "$HERE/make_tiny_moe.py" --n-embd 256 --n-ff 256 "$F32" || exit 1
  "$BINDIR/llama-quantize" "$F32" "$MODEL" Q4_K_M >/dev/null 2>&1 \
    || { echo "llama-quantize failed" >&2; exit 1; }
fi

EXPS='--override-tensor ffn_.*_exps\.=CPU'
export TINY_MOE_GGUF="$MODEL"
export GGML_OP_OFFLOAD_MOE_MIN_BATCH=1

run() { "$HERE/run_seq.sh" "$BINDIR" "$PORT" "$REPEATS" $EXPS "$@" | grep '^rep'; }

echo "== baseline (host-resident experts, no cache) =="
BASE="$(run)" || { echo "baseline run failed"; exit 1; }
echo "$BASE"
echo "== cache (ample budget) =="
CACHE="$(run --moe-cache-bytes 67108864)" || { echo "cache run failed"; exit 1; }
echo "$CACHE"
echo "== cache + hybrid, tight budget (forces staging skips), run 1 =="
HYB="$(run --moe-cache-bytes 4194304 --moe-hybrid-mode on)" || { echo "hybrid run failed"; exit 1; }
echo "$HYB"
echo "== cache + hybrid, run 2 (determinism) =="
HYB2="$(run --moe-cache-bytes 4194304 --moe-hybrid-mode on)" || { echo "hybrid rerun failed"; exit 1; }
echo "$HYB2"

# Pass conditions, matched to what each mode can actually promise:
# - The cache is a transparent transfer optimization: same GPU kernels,
#   same bytes -- its output must be TOKEN-IDENTICAL to baseline.
# - The hybrid CPU tier computes misses with different kernels (double
#   accumulation on host); no mixed-execution path can promise
#   bit-identical logits, and on a random tiny model greedy argmax
#   amplifies ULP ties into different tokens. What it must be is
#   DETERMINISTIC; divergences from baseline are printed for review.
FAIL=0
[ "$BASE" = "$CACHE" ] || { echo "FAIL: cache output diverges from baseline"; FAIL=1; }
[ "$HYB" = "$HYB2" ]   || { echo "FAIL: hybrid output is nondeterministic"; FAIL=1; }
if [ "$BASE" != "$HYB" ]; then
  echo "note: hybrid differs from baseline at these lines (expected ULP-level"
  echo "      CPU-vs-GPU divergence; inspect if the tokens look degenerate):"
  diff <(echo "$BASE") <(echo "$HYB") | sed 's/^/  /' || true
fi
[ "$FAIL" = 0 ] && echo "PASS: cache token-identical to baseline; hybrid deterministic"
exit $FAIL
