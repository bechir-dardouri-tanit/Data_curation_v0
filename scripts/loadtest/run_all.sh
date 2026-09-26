#!/usr/bin/env bash
# Master driver: run the concurrency sweep over every checkpoint, smallest first.
# Resumable: models with a completed results file are skipped.
set -u

LT=/scratch/medrl/loadtest
MASTER_LOG=$LT/logs/master.log
LEVELS="${LEVELS:-1,8,32,96,192,384}"

# tag|model   (ordered by parameter count ascending)
MODELS=(
  "luth-0.8b|kurakurai/Luth-2-0.8B"
  "luth-2b|kurakurai/Luth-2-2B"
  "medgemma-4b|google/medgemma-4b-it"
  "medgemma-1.5-4b|google/medgemma-1.5-4b-it"
  "qwen35-4b|Qwen/Qwen3.5-4B"
  "biomistral-7b|BioMistral/BioMistral-7B"
  "huatuogpt-o1-7b|FreedomIntelligence/HuatuoGPT-o1-7B"
  "ii-medical-8b|Intelligent-Internet/II-Medical-8B"
  "ii-medical-8b-1706|Intelligent-Internet/II-Medical-8B-1706"
  "medreason-8b|UCSC-VLAA/MedReason-8B"
  "apertus-8b|EPFLiGHT/Apertus-8B-MeditronFO"
  "qwen35-9b|Qwen/Qwen3.5-9B"
  "eurollm-9b|EPFLiGHT/EuroLLM-9B-MeditronFO"
  "medgemma-27b|google/medgemma-27b-text-it"
  "qwen38-27b|Qwen/Qwen3.8-27B"
)

echo "[$(date +%H:%M:%S)] MASTER START (levels=$LEVELS)" >> "$MASTER_LOG"

for entry in "${MODELS[@]}"; do
  TAG="${entry%%|*}"; MODEL="${entry##*|}"
  RES="$LT/results/$TAG.jsonl"
  if [ -s "$RES" ] && ! grep -q '"fatal":true' "$RES" && [ "$(wc -l < "$RES")" -ge 13 ]; then
    echo "[$(date +%H:%M:%S)] SKIP $TAG (already complete)" >> "$MASTER_LOG"
    continue
  fi
  echo "[$(date +%H:%M:%S)] BEGIN $TAG ($MODEL)" >> "$MASTER_LOG"
  bash /root/medrl/scripts/loadtest/sweep.sh "$MODEL" "$TAG" "$LEVELS" \
    >> "$LT/logs/$TAG.sweep.out" 2>&1
  echo "[$(date +%H:%M:%S)] END $TAG (rc=$?)" >> "$MASTER_LOG"
done

python3 /root/medrl/scripts/loadtest/aggregate.py || true
echo "[$(date +%H:%M:%S)] MASTER DONE" >> "$MASTER_LOG"
