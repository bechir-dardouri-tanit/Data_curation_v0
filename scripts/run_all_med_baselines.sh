#!/bin/bash
# Run all medical reasoning model baselines sequentially
# Usage: ./run_all_med_baselines.sh

set -e

MODELS=(
    "baseline_4b"
    "baseline_medreason"
    "baseline_huatuo_o1"
    "baseline_medgemma_4b"
    "baseline_ii_medical_8b"
    "baseline_9b_think_off"
    "baseline_biomistral"
    # "baseline_medgemma_27b"  # Skip 27B due to memory constraints
)

LOG_DIR="/tmp/medrl_baseline_logs"
mkdir -p "$LOG_DIR"

echo "Starting medical reasoning model baseline evaluations..."
echo "Logs will be saved to: $LOG_DIR"
echo ""

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Starting: $model"
    echo "========================================"
    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="$LOG_DIR/${model}_${timestamp}.log"

    if uv run medrl eval run -c "$model" > "$log_file" 2>&1; then
        echo "✅ $model completed successfully"
        echo "Results will be in: experiments/eval/"
    else
        echo "❌ $model failed (see $log_file)"
        tail -50 "$log_file"
    fi

    # Wait a bit between runs for GPU cleanup
    sleep 10
    echo ""
done

echo "========================================"
echo "All baseline evaluations completed!"
echo "========================================"
