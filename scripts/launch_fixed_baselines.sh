#!/bin/bash
# Re-launch remaining baselines with fixed configurations

set -e

cd /root/medrl
source .venv/bin/activate

echo "Starting fixed baseline launcher at $(date)"

# Remaining baselines with their original issues now fixed
REMAINING_BASELINES=(
    "baseline_9b_think_off"
    "baseline_medreason"
    "baseline_huatuo_o1"
    "baseline_medgemma_4b"
    "baseline_ii_medical_8b"
    "baseline_medgemma_27b"
)

LOG_DIR="/tmp/eval_logs_fixed"
mkdir -p "$LOG_DIR"

for i in "${!REMAINING_BASELINES[@]}"; do
    baseline="${REMAINING_BASELINES[$i]}"
    log_file="$LOG_DIR/${baseline}.log"

    echo "[$((i+1))/${#REMAINING_BASELINES[@]}] Launching $baseline at $(date)"

    # Launch evaluation
    nohup python -m medrl.cli.main eval run --config "$baseline" > "$log_file" 2>&1 &
    eval_pid=$!

    echo "  Started $baseline (PID: $eval_pid)"

    # Wait for completion
    while kill -0 $eval_pid 2>/dev/null; do
        sleep 120  # Check every 2 minutes

        # Check for completion indicators
        if grep -q '"status": "completed"' "$log_file" 2>/dev/null; then
            echo "  ✓ $baseline completed successfully!"
            break
        elif grep -q '"status": "failed"' "$log_file" 2>/dev/null; then
            echo "  ✗ $baseline failed - check $log_file"
            break
        fi

        echo "    Still running... ($(date))"
    done

    # Wait for GPU cleanup
    echo "  Waiting for GPU cleanup..."
    sleep 60
done

echo "All fixed baselines completed at $(date)"
echo "Check logs in $LOG_DIR for details"