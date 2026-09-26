#!/bin/bash
# Sequential baseline launcher - waits for each eval to complete before starting next
# Prevents GPU memory conflicts by running evaluations one at a time

set -e

cd /root/medrl
source .venv/bin/activate

BASELINES=(
    "baseline_biomistral"
    "baseline_medreason"
    "baseline_huatuo_o1"
    "baseline_medgemma_4b"
    "baseline_ii_medical_8b"
    "baseline_medgemma_27b"
)

LOG_DIR="/tmp/eval_logs"
mkdir -p "$LOG_DIR"

echo "Starting sequential baseline launcher at $(date)"
echo "Remaining baselines: ${#BASELINES[@]}"

for i in "${!BASELINES[@]}"; do
    baseline="${BASELINES[$i]}"
    log_file="$LOG_DIR/${baseline}.log"

    echo "[$((i+1))/${#BASELINES[@]}] Launching $baseline at $(date)"

    # Launch evaluation in background
    nohup python -m medrl.cli.main eval run --config "$baseline" > "$log_file" 2>&1 &
    eval_pid=$!

    echo "Started $baseline (PID: $eval_pid)"

    # Wait for completion
    while kill -0 $eval_pid 2>/dev/null; do
        sleep 60
        # Check for completion indicators in log
        if grep -q "finished_at.*null" "$log_file" 2>/dev/null; then
            echo "  ✓ $baseline still running..."
        elif grep -q '"status": "completed"' "$log_file" 2>/dev/null; then
            echo "  ✓ $baseline completed successfully!"
            break
        elif grep -q '"status": "failed"' "$log_file" 2>/dev/null; then
            echo "  ✗ $baseline failed - check $log_file"
            break
        fi
    done

    # Wait a bit for GPU cleanup
    sleep 30
    echo "  Completed $baseline at $(date)"
done

echo "All baselines launched at $(date)"
echo "Check logs in $LOG_DIR for details"