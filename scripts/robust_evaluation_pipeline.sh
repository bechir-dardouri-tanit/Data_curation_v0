#!/bin/bash
# Robust Evaluation Pipeline - Production Ready
# Handles failures gracefully with automatic retry and comprehensive error categorization

set -e

# Environment setup
source /root/medrl/scripts/setup_eval_env.sh
cd /root/medrl
source .venv/bin/activate

# Configuration
BASELINE_CONFIGS=(
    "baseline_9b_think_off"
    "baseline_ii_medical_8b" 
    "baseline_medreason"
    "baseline_medgemma_4b"
    "baseline_medgemma_27b"
)

LOG_DIR="/tmp/eval_pipeline_logs"
STATE_FILE="/tmp/eval_pipeline_state.json"
REPORT_DIR="/tmp/eval_pipeline_reports"

# Create directories
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Resource thresholds
MIN_DISK_GB=50
MIN_GPU_MEM_GB=70
MAX_RETRIES=3

echo "=== ROBUST EVALUATION PIPELINE ==="
echo "Started: $(date)"
echo "Disk space: $(df -h / | tail -1 | awk '{print $4}')"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)MB"

# Pre-flight resource checks
check_resources() {
    local disk_free=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
    local gpu_mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    local gpu_mem_gb=$((gpu_mem_free / 1024))
    
    echo "Resource Check:"
    echo "  Disk free: ${disk_free}GB (required: ${MIN_DISK_GB}GB)"
    echo "  GPU memory free: ${gpu_mem_gb}GB (required: ${MIN_GPU_MEM_GB}GB)"
    
    if [ "$disk_free" -lt "$MIN_DISK_GB" ]; then
        echo "ERROR: Insufficient disk space"
        return 1
    fi
    
    if [ "$gpu_mem_gb" -lt "$MIN_GPU_MEM_GB" ]; then
        echo "ERROR: Insufficient GPU memory"
        return 1
    fi
    
    echo "✓ Resource check passed"
    return 0
}

# GPU cleanup function
cleanup_gpu() {
    echo "GPU cleanup: Killing orphaned vLLM processes..."
    pkill -f "vllm serve" || true
    sleep 5
    echo "✓ GPU cleanup completed"
}

# Error categorization
categorize_error() {
    local log_file="$1"
    if grep -q "CUDA out of memory\|GPU memory" "$log_file"; then
        echo "GPU_OOM"
    elif grep -q "extraction fail rate" "$log_file"; then
        echo "EXTRACTION_FAILURE"
    elif grep -q "TemplateError\|Conversation roles" "$log_file"; then
        echo "TEMPLATE_ERROR"
    elif grep -q "Engine core initialization" "$log_file"; then
        echo "SERVER_STARTUP_FAILURE"
    elif grep -q "401\|403\|404\|Repository" "$log_file"; then
        echo "NETWORK_ERROR"
    elif grep -q "judge.*unparseable\|JSON" "$log_file"; then
        echo "JUDGE_ERROR"
    else
        echo "UNKNOWN_ERROR"
    fi
}

# Run single baseline with retry
run_baseline_with_retry() {
    local baseline="$1"
    local log_file="$LOG_DIR/${baseline}.log"
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        echo "[$((retry_count + 1))/$MAX_RETRIES] Running $baseline..."
        
        # GPU cleanup before run
        cleanup_gpu
        
        # Run evaluation
        timeout 7200 python -m medrl.cli.main eval run --config "$baseline" > "$log_file" 2>&1
        local exit_code=$?
        
        if [ $exit_code -eq 0 ] && grep -q '"status": "completed"' "$log_file"; then
            echo "✓ $baseline completed successfully"
            return 0
        fi
        
        # Analyze failure
        local error_type=$(categorize_error "$log_file")
        echo "✗ $baseline failed: $error_type"
        
        # Skip retry for template errors (unlikely to succeed)
        if [ "$error_type" = "TEMPLATE_ERROR" ]; then
            echo "Template error - skipping retry"
            return 1
        fi
        
        # Extended delay for GPU/extraction errors
        if [ "$error_type" = "GPU_OOM" ] || [ "$error_type" = "EXTRACTION_FAILURE" ]; then
            echo "Waiting 300s before retry..."
            sleep 300
        else
            echo "Waiting 60s before retry..."
            sleep 60
        fi
        
        retry_count=$((retry_count + 1))
    done
    
    echo "✗ $baseline failed after $MAX_RETRIES attempts"
    return 1
}

# Main execution
main() {
    # Resource check
    if ! check_resources; then
        echo "ERROR: Resource check failed"
        exit 1
    fi
    
    # Process baselines
    local success_count=0
    local failure_count=0
    
    for baseline in "${BASELINE_CONFIGS[@]}"; do
        echo ""
        echo "=== Processing: $baseline ==="
        
        if run_baseline_with_retry "$baseline"; then
            success_count=$((success_count + 1))
        else
            failure_count=$((failure_count + 1))
        fi
        
        # GPU cleanup between runs
        cleanup_gpu
        sleep 30
    done
    
    # Generate report
    echo ""
    echo "=== PIPELINE SUMMARY ==="
    echo "Completed: $(date)"
    echo "Successful: $success_count"
    echo "Failed: $failure_count"
    
    if [ $failure_count -eq 0 ]; then
        echo "Status: ALL SUCCESS"
        exit 0
    elif [ $success_count -gt 0 ]; then
        echo "Status: PARTIAL SUCCESS"
        exit 2
    else
        echo "Status: ALL FAILED"
        exit 1
    fi
}

main "$@"
