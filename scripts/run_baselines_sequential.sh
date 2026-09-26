#!/bin/bash
# Sequential launcher for baseline evaluations
# Waits for each job to complete successfully before starting the next

set -e

MEDRL_DIR="/root/medrl"
PYTHON="${MEDRL_DIR}/.venv/bin/python"
LOG_DIR="/tmp"
MODELS=(
    "baseline_4b"
    "baseline_9b_think_on"
    "baseline_9b_think_off"
    "baseline_biomistral"
    "baseline_huatuo_o1"
    "baseline_ii_medical_8b"
    "baseline_medgemma_27b"
    "baseline_medgemma_4b"
    "baseline_medreason"
)

echo "Starting sequential baseline evaluations..."
echo "Logs will be written to ${LOG_DIR}/eval_<name>.log"

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Launching: $model"
    echo "=========================================="
    
    cd "$MEDRL_DIR"
    
    # Launch evaluation in background with logging
    nohup $PYTHON -m medrl.cli.main eval run --config "$model" > "${LOG_DIR}/eval_${model}.log" 2>&1 &
    PID=$!
    
    echo "Started with PID: $PID"
    echo "Log file: ${LOG_DIR}/eval_${model}.log"
    
    # Wait for startup - check for successful vllm launch
    echo "Waiting for startup..."
    sleep 10
    
    # Check if process is still running
    if ! kill -0 $PID 2>/dev/null; then
        echo "ERROR: Process exited early. Check log:"
        tail -50 "${LOG_DIR}/eval_${model}.log"
        exit 1
    fi
    
    # Check for successful startup indicators
    if grep -q "vllm serve" "${LOG_DIR}/eval_${model}.log" 2>/dev/null; then
        echo "✓ vllm startup detected"
    else
        echo "Waiting for vllm startup..."
        sleep 10
    fi
    
    # Wait for completion
    echo "Running evaluation (monitor with: tail -f ${LOG_DIR}/eval_${model}.log)"
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $model completed successfully"
    else
        echo "✗ $model failed with exit code $EXIT_CODE"
        echo "Check log: ${LOG_DIR}/eval_${model}.log"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All baseline evaluations completed!"
echo "=========================================="
