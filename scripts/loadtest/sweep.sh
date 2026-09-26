#!/usr/bin/env bash
# Concurrency sweep for ONE model: ensure weights -> serve -> run cells -> teardown.
# Usage: sweep.sh <hf_id> <tag> [levels_csv]   e.g. sweep.sh Qwen/Qwen3.5-9B qwen35-9b "1,8,32,96,192,384"
set -u

MODEL="$1"
TAG="$2"
LEVELS="${3:-1,8,32,96,192,384}"

LT=/scratch/medrl/loadtest
POOL=$LT/prompts/pool.json
RESULTS=$LT/results/$TAG.jsonl
LOG=$LT/logs/$TAG.serve.log
MASTER_LOG=$LT/logs/master.log
PORT=$(( 21000 + RANDOM % 8000 ))

mkdir -p "$LT/results" "$LT/logs"
: > "$RESULTS"

export HF_HOME=/scratch/medrl/.cache/huggingface
export HF_HUB_CACHE=/scratch/medrl/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/medrl/.cache/datasets
# Gated repos (medgemma family) need the token for download AND for serving load.
export HF_TOKEN="${HF_TOKEN:-***REDACTED***}"
source /root/medrl/.venv/bin/activate

THINK_FLAG=""
case "$MODEL" in
  Qwen/*|*II-Medical-8B*|*MedReason*) THINK_FLAG="--thinking-off" ;;
esac

log() { echo "[$(date +%H:%M:%S)] [$TAG] $*" >> "$MASTER_LOG"; }

cleanup() {
  pkill -f "vllm serve $MODEL " 2>/dev/null || true
  sleep 3
  pkill -9 -f "vllm serve $MODEL " 2>/dev/null || true
  for i in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$USED" -lt 2000 ] && break
    sleep 2
  done
}

# ---- prefetch weights (so serve-start failures are visible separately) ----
log "prefetch $MODEL"
if command -v hf >/dev/null 2>&1; then HFD="hf download"; else HFD="huggingface-cli download"; fi
if ! $HFD "$MODEL" >> "$LOG.download" 2>&1; then
  echo "{\"tag\":\"$TAG\",\"model\":\"$MODEL\",\"cell\":\"download\",\"fatal\":true}" >> "$RESULTS"
  log "DOWNLOAD FAILED $MODEL"
  exit 1
fi

# ---- launch server ----
log "serving $MODEL on :$PORT"
# Derive context length from the model config; floor 4096 (longest pool prompts
# ~2.7k tok + decode), cap 16384 (EuroLLM-9B only allows 4096).
MAXLEN=$(python3 - "$MODEL" <<'PYEOF'
import sys
from transformers import AutoConfig
try:
    c = AutoConfig.from_pretrained(sys.argv[1])
    m = getattr(c, "max_position_embeddings", None)
    if m is None:
        m = getattr(c, "model_max_length", None)
    if m is None:
        print(16384)
    else:
        print(max(4096, min(int(m), 16384)))
except Exception:
    print(8192)
PYEOF
)
log "max_model_len=$MAXLEN"
nohup vllm serve "$MODEL" \
  --port "$PORT" --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len "$MAXLEN" --max-num-seqs 400 \
  --language-model-only > "$LOG" 2>&1 &
SERVER_PID=$!

UP=""
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then UP=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 5
done

if [ -z "$UP" ]; then
  echo "{\"tag\":\"$TAG\",\"model\":\"$MODEL\",\"cell\":\"server_start\",\"fatal\":true}" >> "$RESULTS"
  log "SERVER FAILED TO START $MODEL (see $LOG)"
  cleanup
  exit 1
fi
log "server up after ~$((i*5))s"

run_cell() {
  local level="$1"; shift
  local extra=("$@")
  local nreq
  nreq=$(python3 -c "print(min(3*$level+8, 384))")
  python3 /root/medrl/scripts/loadtest/client.py \
    --url "http://127.0.0.1:$PORT" --pool "$POOL" \
    --level "$level" --requests "$nreq" --tag "$TAG" \
    --out-append "$RESULTS" "${extra[@]}" >> "$LT/logs/$TAG.client.log" 2>&1
  # A failed cell (client error) still produced its row(s); only server death
  # aborts the remaining cells.
  if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "SERVER DIED during level $level"
    return 2
  fi
  return 0
}

SERVER_DOWN=0
for LVL in ${LEVELS//,/ }; do
  if [ "$SERVER_DOWN" = "1" ]; then break; fi
  log "cell C=$LVL chat ($TAG)"
  run_cell "$LVL" $THINK_FLAG || SERVER_DOWN=1
  log "cell C=$LVL decode ($TAG)"
  run_cell "$LVL" --decode-only || SERVER_DOWN=1
done

# long-input probe at C=192 (both modes)
if [ "$SERVER_DOWN" != "1" ]; then
  log "long-input probes ($TAG)"
  python3 /root/medrl/scripts/loadtest/client.py \
    --url "http://127.0.0.1:$PORT" --pool "$POOL" \
    --level 192 --requests 384 --tag "$TAG-long" --long-only \
    --out-append "$RESULTS" $THINK_FLAG >> "$LT/logs/$TAG.client.log" 2>&1 || true
  python3 /root/medrl/scripts/loadtest/client.py \
    --url "http://127.0.0.1:$PORT" --pool "$POOL" \
    --level 192 --requests 384 --tag "$TAG-long" --long-only --decode-only \
    --out-append "$RESULTS" >> "$LT/logs/$TAG.client.log" 2>&1 || true
fi

# annotate model identity into results file
python3 - "$TAG" "$MODEL" "$RESULTS" <<'EOF'
import json, sys
tag, model, path = sys.argv[1:4]
rows = []
with open(path) as f:
    for line in f:
        r = json.loads(line)
        r.setdefault("model", model)
        r["tag"] = tag
        rows.append(r)
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
EOF

cleanup
log "done $TAG ($(wc -l < "$RESULTS") cells)"
