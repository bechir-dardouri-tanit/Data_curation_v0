#!/usr/bin/env bash
# B7 pilot GPU sequence: embed -> bench-vectors -> semantic decontam -> judge -> difficulty -> coverage.
# One model served at a time; every stage seals its manifest; EXIT marker at the end.
set -u
source /root/medrl/.venv/bin/activate
export HF_HOME=/scratch/medrl/.cache/huggingface HF_HUB_CACHE=/scratch/medrl/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/medrl/.cache/datasets
export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN first}"
LOG=/scratch/medrl/curation/logs/pilot-gpu.log
RUN=pilot-b7
mkdir -p /scratch/medrl/curation/logs

wait_health() { # port, name
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "[serve] $2 up" >> "$LOG"; return 0; }
    sleep 5
  done
  echo "[serve] $2 FAILED to come up" >> "$LOG"; return 1
}

# ---- embedding server ----
echo "[serve] starting embedder (Qwen3-Embedding-4B, pooling)" >> "$LOG"
nohup vllm serve Qwen/Qwen3-Embedding-4B --port 8101 --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --runner pooling --max-model-len 512 --max-num-seqs 400 \
  > /scratch/medrl/curation/logs/pilot-embed-serve.log 2>&1 &
wait_health 8101 embedder || { echo "EXIT:embed-serve-failed" >> "$LOG"; exit 1; }

python - << 'PYEOF' >> "$LOG" 2>&1
import time
from medrl.curation.stages import embed
from medrl.curation.store import save_manifest, seal_manifest

t0 = time.monotonic()
m = seal_manifest(embed.run_embed("pilot-b7", base_url="http://127.0.0.1:8101"))
save_manifest(m)
print(f"[05_embed] in={m.rows_in} out={m.rows_out} wall={m.wall_s}s", flush=True)
print("  notes:", {k: v for k, v in m.notes.items() if isinstance(v, (int, float, str))}, flush=True)
print("PHASE-EMBED DONE", round(time.monotonic() - t0, 1), "s", flush=True)
PYEOF
[ $? -ne 0 ] && { echo "EXIT:embed-failed" >> "$LOG"; exit 1; }

# benchmark vectors (same embed server)
python /root/medrl/scripts/curation/embed_benchmarks.py \
  --base-url http://127.0.0.1:8101 \
  --out /scratch/medrl/curation/$RUN/bench_vectors >> "$LOG" 2>&1 \
  || { echo "EXIT:benchvec-failed" >> "$LOG"; exit 1; }
echo "[bench_vectors] done" >> "$LOG"

python - << 'PYEOF' >> "$LOG" 2>&1
from medrl.curation.stages import decontam_sem
from medrl.curation.store import save_manifest, seal_manifest
RUN = "pilot-b7"
m = seal_manifest(decontam_sem.run_decontam_sem(
    RUN,
    bench_vectors_path=f"/scratch/medrl/curation/{RUN}/bench_vectors.json",
))
save_manifest(m)
print(f"[06_decontam_sem] in={m.rows_in} out={m.rows_out} wall={m.wall_s}s", flush=True)
print("  rates:", {k: round(v.get("_all", 0), 5) for k, v in m.flag_rates.items()}, flush=True)
print("  notes:", {k: v for k, v in m.notes.items() if isinstance(v, (int, float))}, flush=True)
print("PHASE-SEMANTIC DONE", flush=True)
PYEOF
[ $? -ne 0 ] && { echo "EXIT:semantic-failed" >> "$LOG"; exit 1; }

pkill -f "vllm serve Qwen/Qwen3-Embedding-4B" 2>/dev/null; sleep 8

# ---- judge server ----
echo "[serve] starting judge (Qwen3.5-4B, thinking off)" >> "$LOG"
nohup vllm serve Qwen/Qwen3.5-4B --port 8102 --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 400 \
  --reasoning-parser qwen3 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' > /scratch/medrl/curation/logs/pilot-judge-serve.log 2>&1 &
wait_health 8102 judge || { echo "EXIT:judge-serve-failed" >> "$LOG"; exit 1; }

python - << 'PYEOF' >> "$LOG" 2>&1
from medrl.curation.stages import judge
from medrl.curation.store import save_manifest, seal_manifest
RUN = "pilot-b7"
m = seal_manifest(judge.stage_entry(RUN))
save_manifest(m)
print(f"[11_judge] in={m.rows_in} out={m.rows_out} wall={m.wall_s}s", flush=True)
print("  notes:", {k: v for k, v in m.notes.items() if isinstance(v, (int, float))}, flush=True)
print("PHASE-JUDGE DONE", flush=True)
PYEOF
[ $? -ne 0 ] && { echo "EXIT:judge-failed" >> "$LOG"; exit 1; }

pkill -f "vllm serve Qwen/Qwen3.5-4B" 2>/dev/null; sleep 8

# ---- 9B difficulty server ----
echo "[serve] starting 9B (thinking on, 32k ctx)" >> "$LOG"
nohup vllm serve Qwen/Qwen3.5-9B --port 8103 --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 400 \
  --reasoning-parser qwen3 --language-model-only > /scratch/medrl/curation/logs/pilot-9b-serve.log 2>&1 &
wait_health 8103 9b || { echo "EXIT:9b-serve-failed" >> "$LOG"; exit 1; }

python - << 'PYEOF' >> "$LOG" 2>&1
from pathlib import Path
from medrl.curation.serving import Gateway, ServerHandle
from medrl.curation.stages import coverage, difficulty
from medrl.curation.store import save_manifest, seal_manifest
RUN = "pilot-b7"
gw = Gateway(ServerHandle(model="Qwen/Qwen3.5-9B", port=8103), max_concurrency=192)
m12 = seal_manifest(difficulty.stage_entry(
    RUN,
    gateway=gw,
    generation_out=Path(f"/scratch/medrl/curation/{RUN}/12_difficulty/generations.jsonl"),
))
save_manifest(m12)
print(f"[12_difficulty] in={m12.rows_in} out={m12.rows_out} wall={m12.wall_s}s", flush=True)
print("  rates:", {k: round(v.get("_all", 0), 5) for k, v in m12.flag_rates.items()}, flush=True)
print("  notes:", {k: v for k, v in m12.notes.items() if isinstance(v, (int, float, str))}, flush=True)
m13 = seal_manifest(coverage.stage_entry(RUN, input_stage="12_difficulty"))
save_manifest(m13)
print(f"[13_coverage] out={m13.rows_out} wall={m13.wall_s}s", flush=True)
print("PHASE-DIFFICULTY DONE", flush=True)
PYEOF
[ $? -ne 0 ] && { echo "EXIT:difficulty-failed" >> "$LOG"; exit 1; }

pkill -f "vllm serve Qwen/Qwen3.5-9B" 2>/dev/null
echo "PILOT-GPU-CHAIN DONE" >> "$LOG"
