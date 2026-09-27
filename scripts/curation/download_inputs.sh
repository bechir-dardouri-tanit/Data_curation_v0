#!/usr/bin/env bash
# B0 downloads: the 11 pool datasets + 3 serving models into the scratch cache.
# Sequential within category; resumable (hf download skips complete files).
set -u
export HF_HOME=/scratch/medrl/.cache/huggingface
export HF_HUB_CACHE=/scratch/medrl/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/medrl/.cache/datasets
export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN first}"
LOG=/scratch/medrl/curation/logs/downloads.log
mkdir -p /scratch/medrl/curation/logs
source /root/medrl/.venv/bin/activate

DATASETS=(
  Intelligent-Internet/II-Medical-Reasoning-SFT
  hongzhouyu/FineMed-SFT
  lavita/ChatDoctor-HealthCareMagic-100k
  GeneralReasoning/GeneralThought-430K
  FreedomIntelligence/Medical-R1-Distill-Data
  UCSC-VLAA/m23k-tokenized
  UCSC-VLAA/MedReason
  FreedomIntelligence/medical-o1-reasoning-SFT
  hongzhouyu/FineMed-DPO
  Intelligent-Internet/II-Medical-RL
  Intelligent-Internet/ChatDoctor-RL
)
MODELS=(Qwen/Qwen3-Embedding-4B Qwen/Qwen3.5-9B)

echo "[$(date +%H:%M:%S)] downloads start" >> "$LOG"
for d in "${DATASETS[@]}"; do
  echo "[$(date +%H:%M:%S)] dataset $d" >> "$LOG"
  hf download "$d" --repo-type dataset >> "$LOG" 2>&1 || echo "[$(date +%H:%M:%S)] FAIL dataset $d" >> "$LOG"
done
for m in "${MODELS[@]}"; do
  echo "[$(date +%H:%M:%S)] model $m" >> "$LOG"
  hf download "$m" >> "$LOG" 2>&1 || echo "[$(date +%H:%M:%S)] FAIL model $m" >> "$LOG"
done
echo "[$(date +%H:%M:%S)] downloads done" >> "$LOG"
