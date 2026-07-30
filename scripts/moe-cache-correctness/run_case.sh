#!/usr/bin/env bash
# Usage: run_case.sh <binary_dir> <port> <num_requests> <reset_before_last:0|1> [extra llama-server args...]
BINDIR="$1"; PORT="$2"; NREQ="${3:-1}"; RESET_LAST="${4:-0}"; shift 4
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
  -ngl 99 --port "$PORT" --no-webui "$@" > /tmp/case-server-$PORT.log 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 30); do
  sleep 1
  if grep -q "model loaded" /tmp/case-server-$PORT.log 2>/dev/null; then READY=1; break; fi
  if ! kill -0 $SPID 2>/dev/null; then break; fi
done
if [ "$READY" != "1" ]; then
  echo "SERVER FAILED TO START"
  tail -30 /tmp/case-server-$PORT.log
  kill -9 $SPID 2>/dev/null
  exit 1
fi
for r in $(seq 1 $NREQ); do
  if [ "$r" = "$NREQ" ] && [ "$RESET_LAST" = "1" ]; then
    curl -s -X POST "http://127.0.0.1:$PORT/cache/reset" > /dev/null
  fi
  BIGPROMPT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 6)")
  RESP=$(curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$BIGPROMPT\", \"n_predict\": 8, \"temperature\": 0, \"seed\": 42, \"return_tokens\": true}")
  echo "request $r tokens:"
  echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(' ', d.get('tokens', d))" 2>&1
done
echo "--- metrics (moe_cache) ---"
curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -i moe_cache
kill -9 $SPID 2>/dev/null
wait $SPID 2>/dev/null
