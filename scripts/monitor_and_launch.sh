#!/bin/bash
# Improved sequential baseline launcher with proper monitoring

set -e

cd /root/medrl
source .venv/bin/activate

echo "Starting improved sequential launcher at $(date)"

# Function to check if evaluation is running
is_eval_running() {
    ps aux | grep "python.*medrl.*eval" | grep -v grep | grep -q "baseline_$1"
    return $?
}

# Function to check evaluation completion status
get_eval_status() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        if grep -q '"status": "completed"' "$log_file" 2>/dev/null; then
            echo "completed"
        elif grep -q '"status": "failed"' "$log_file" 2>/dev/null; then
            echo "failed"
        elif grep -q '"status": "running"' "$log_file" 2>/dev/null; then
            echo "running"
        else
            echo "unknown"
        fi
    else
        echo "no_log"
    fi
}

# Function to find the most recent eval directory for a baseline
find_eval_dir() {
    local baseline="$1"
    local latest_dir=$(ls -td /scratch/medrl/runs/eval-* 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        # Check if this directory matches our baseline
        if grep -q "\"hf_id\":.*${baseline//-/_}" "$latest_dir/manifest.json" 2>/dev/null || \
           grep -q "$baseline" "$latest_dir/manifest.json" 2>/dev/null; then
            echo "$latest_dir"
            return 0
        fi
    fi
    echo ""
}

# Wait for current 9B think OFF to complete
echo "Waiting for current 9B think OFF evaluation to complete..."
while is_eval_running "9b_think_off"; do
    sleep 60
    echo "  Still running at $(date)"
done

echo "✓ 9B think OFF completed at $(date)"

# Now launch remaining baselines sequentially
REMAINING_BASELINES=(
    "baseline_biomistral"
    "baseline_medreason"
    "baseline_huatuo_o1"
    "baseline_medgemma_4b"
    "baseline_ii_medical_8b"
    "baseline_medgemma_27b"
)

for i in "${!REMAINING_BASELINES[@]}"; do
    baseline="${REMAINING_BASELINES[$i]}"
    log_file="/tmp/eval_logs/${baseline}.log"

    echo "[$((i+1))/${#REMAINING_BASELINES[@]}] Launching $baseline at $(date)"

    # Launch evaluation
    nohup python -m medrl.cli.main eval run --config "$baseline" > "$log_file" 2>&1 &
    eval_pid=$!

    echo "  Started $baseline (PID: $eval_pid)"

    # Monitor progress
    while kill -0 $eval_pid 2>/dev/null; do
        sleep 120  # Check every 2 minutes

        # Check for success/failure indicators
        if grep -q '"status": "completed"' "$log_file" 2>/dev/null; then
            echo "  ✓ $baseline completed successfully!"
            break
        elif grep -q '"status": "failed"' "$log_file" 2>/dev/null; then
            echo "  ✗ $baseline failed - check $log_file"
            break
        fi

        # Show progress indicator
        echo "    Still running... ($(date))"
    done

    # Wait for GPU cleanup
    echo "  Waiting for GPU cleanup..."
    sleep 60
done

echo "All remaining baselines completed at $(date)"