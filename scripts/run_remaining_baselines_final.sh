#!/bin/bash
# Final sequential baseline launcher - runs all remaining baselines with monitoring
# All configs have been fixed with proper extraction fail rates and syntax

set -e

cd /root/medrl
source .venv/bin/activate

# Fix disk space issue for II-Medical-8B
export HF_HOME=/scratch/medrl/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/medrl/.cache/datasets

mkdir -p /tmp/eval_logs
mkdir -p /scratch/medrl/.cache/huggingface
mkdir -p /scratch/medrl/.cache/datasets

echo "=================================================="
echo "FINAL BASELINE SEQUENTIAL LAUNCHER"
echo "Starting at $(date)"
echo "=================================================="
echo "HF_HOME: $HF_HOME"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "Disk space check:"
df -h / | tail -1
df -h /scratch | tail -1
echo "=================================================="

# Define remaining baselines in priority order
BASELINES=(
    "baseline_9b_think_off"
    "baseline_ii_medical_8b"
    "baseline_medreason"
    "baseline_medgemma_4b"
    "baseline_medgemma_27b"
)

TOTAL=${#BASELINES[@]}

for i in "${!BASELINES[@]}"; do
    baseline="${BASELINES[$i]}"
    num=$((i+1))
    log_file="/tmp/eval_logs/${baseline}.log"
    
    echo ""
    echo "=================================================="
    echo "[$num/$TOTAL] Launching: $baseline"
    echo "Time: $(date)"
    echo "Log: $log_file"
    echo "=================================================="
    
    # Check GPU status before launching
    echo "GPU status before launch:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
    
    # Launch evaluation in background
    nohup python -m medrl.cli.main eval run --config "$baseline" > "$log_file" 2>&1 &
    eval_pid=$!
    
    echo "Started $baseline (PID: $eval_pid)"
    echo "Waiting for completion..."
    
    # Monitor progress with status checks
    start_time=$(date +%s)
    check_count=0
    
    while kill -0 $eval_pid 2>/dev/null; do
        sleep 120  # Check every 2 minutes
        check_count=$((check_count + 1))
        
        # Show progress every 10 minutes
        if [ $((check_count % 5)) -eq 0 ]; then
            elapsed=$((($(date +%s) - start_time) / 60))
            echo "  [$(date +%H:%M)] Still running (elapsed: ${elapsed}min, PID: $eval_pid alive)"
            
            # Show GPU utilization briefly
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
            echo "  GPU utilization: ${gpu_util}%"
        fi
        
        # Check for completion in log file
        if [ -f "$log_file" ]; then
            if grep -q '"status": "completed"' "$log_file" 2>/dev/null; then
                echo "  ✓ COMPLETED: $baseline finished successfully!"
                break
            elif grep -q '"status": "failed"' "$log_file" 2>/dev/null; then
                echo "  ✗ FAILED: $baseline failed - check $log_file"
                # Try to get error details
                error_line=$(grep -A5 '"error":' "$log_file" | head -6)
                echo "  Error details: $error_line"
                break
            fi
        fi
    done
    
    # Final status check
    if ! kill -0 $eval_pid 2>/dev/null; then
        wait $eval_pid
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "  ✓ Process exited cleanly"
        else
            echo "  ⚠ Process exited with code: $exit_code"
        fi
    fi
    
    # GPU cleanup buffer
    echo "  Waiting for GPU cleanup (60 seconds)..."
    sleep 60
    
    echo "  Completed $baseline at $(date)"
    echo "=================================================="
done

echo ""
echo "=================================================="
echo "ALL BASELINES COMPLETED!"
echo "Finished at $(date)"
echo "=================================================="
echo "Results can be found in /scratch/medrl/runs/eval-*/"
echo "Logs available in /tmp/eval_logs/"
echo "=================================================="
