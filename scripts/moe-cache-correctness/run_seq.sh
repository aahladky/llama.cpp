#!/usr/bin/env bash
# Usage: run_seq.sh <binary_dir> <port> <repeats> [extra llama-server args...]
BINDIR="$1"; PORT="$2"; REPEATS="$3"; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${TINY_MOE_GGUF:-$HERE/tiny-moe.gguf}"
if [ ! -f "$MODEL" ]; then
  echo "Model fixture not found at $MODEL -- run: python3 $HERE/make_tiny_moe.py $MODEL" >&2
  exit 1
fi
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
rm -f /tmp/case-server-$PORT.log
LD_LIBRARY_PATH="$BINDIR:$LD_LIBRARY_PATH" "$BINDIR/llama-server" \
  -m "$MODEL" \
  -ngl 99 --port "$PORT" --no-webui --metrics --parallel 1 "$@" > /tmp/case-server-$PORT.log 2>&1 &
SPID=$!
for i in $(seq 1 30); do
  sleep 1
  if grep -q "model loaded" /tmp/case-server-$PORT.log 2>/dev/null; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER DIED"; cat /tmp/case-server-$PORT.log; exit 1; fi
done
P1="The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. "
P2="A journey of a thousand miles begins with a single step. A journey of a thousand miles begins with a single step. A journey of a thousand miles begins with a single step. A journey of a thousand miles begins with a single step. A journey of a thousand miles begins with a single step. "
P3="To be or not to be, that is the question. To be or not to be, that is the question. To be or not to be, that is the question. To be or not to be, that is the question. To be or not to be, that is the question. To be or not to be, that is the question. "
P4="Roses are red, violets are blue, sugar is sweet. Roses are red, violets are blue, sugar is sweet. Roses are red, violets are blue, sugar is sweet. Roses are red, violets are blue, sugar is sweet. Roses are red, violets are blue, sugar is sweet. "
for rep in $(seq 1 $REPEATS); do
  for p in "$P1" "$P2" "$P3" "$P4"; do
    RESP=$(curl -s http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" \
      -d "{\"prompt\": \"$p\", \"n_predict\": 8, \"temperature\": 0, \"seed\": 42, \"id_slot\": 0, \"return_tokens\": true}")
    echo -n "rep$rep [${p:0:20}]: "
    echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tokens', d))"
  done
  echo "--- metrics after rep$rep ---"
  curl -s http://127.0.0.1:$PORT/metrics | grep -E "moe_cache_(hits|misses|evictions|promotions|slots_used)_total|moe_cache_slots_used"
done
kill -9 $SPID 2>/dev/null
wait $SPID 2>/dev/null
