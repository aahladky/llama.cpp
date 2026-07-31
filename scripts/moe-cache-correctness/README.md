# MoE expert-cache correctness fixtures

Ad hoc scripts for deterministic token-identity checks of the MoE cache
against a stock-upstream oracle. Not wired into any CI/test runner --
these are reproduction tools that exercise real devices and a real
server, complementing the host-only policy tests in
`tests/test-moe-cache.cpp` / `tests/test-moe-hybrid.cpp`. (Lives under
`scripts/`, not `tests/`, because `tests/.gitignore` blanket-ignores
everything in that directory.)

## Usage

```bash
# 1. Generate the tiny deterministic MoE model fixture (~76 MiB, gitignored,
#    fully reproducible: fixed seed, real llama-3 BPE vocab copied from
#    models/ggml-vocab-llama-bpe.gguf, everything else synthetic and tiny).
python3 make_tiny_moe.py                 # writes ./tiny-moe.gguf

# 2. Single completion against a given build, with optional server args.
#    IMPORTANT: pass -ot "exps=CPU" to force expert weights host-resident --
#    otherwise ggml offloads them fully to VRAM and the cache hook (which
#    only intercepts host->device expert copies) never engages at all.
./run_case.sh <bin-dir> <port> <num-requests> <reset-before-last:0|1> [server args...]

# 3. Sequence of 4 distinct long-form prompts (>32 tokens each, so the
#    expert-copy staging path in ggml-backend.cpp actually triggers --
#    short prompts/decode-only batches sit below the op-offload batch
#    threshold and never reach the hook), repeated N times.
./run_seq.sh <bin-dir> <port> <repeats> [server args...]
```

Compare the printed `tokens:` arrays against a stock-upstream oracle build
(no cache code at all) run the same way. Any drift is a real bug; the
model's actual generated text is gibberish (untrained random weights) and
is not itself meaningful.

## Known quirks worth knowing before re-running this

- A `moe_cache_hits_total`/`misses_total` metric staying flat across
  identical repeated requests is not a bug -- it lines up with llama.cpp's
  own graph-reuse optimization (see "graphs reused" in server logs) skipping
  redundant restaging when nothing changed. Use *distinct* prompts (as
  `run_seq.sh` does) to see real hit/miss/eviction activity.
- Per-slot size is `gate_bytes + up_bytes + down_bytes` together (one slot
  holds all three projections of one expert), not one projection at a time
  -- `--moe-cache-bytes` needs to be at least that large for the cache to
  initialize at all (silently no-ops below that, logged as
  `moe_cache: init failed on device 0`).
