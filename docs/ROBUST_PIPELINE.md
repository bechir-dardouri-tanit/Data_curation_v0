# Robust Evaluation Pipeline

Production-ready evaluation pipeline for MedRL with comprehensive error handling, resource management, and monitoring capabilities.

## Overview

The robust evaluation pipeline (`scripts/robust_evaluation_pipeline.sh`) provides bulletproof execution of baseline evaluations with:

- **Failure isolation**: Individual baseline failures don't stop the pipeline
- **Automatic retry**: Intelligent retries with adjusted parameters for different error types
- **Resource management**: Pre-flight checks and GPU cleanup between runs
- **Progress monitoring**: Real-time health checks and progress tracking
- **State persistence**: Resume capability from failures
- **Comprehensive reporting**: HTML and text reports with detailed error categorization

## Features

### 1. Failure Handling

The pipeline classifies errors into specific categories for targeted handling:

- **EXTRACTION_FAILURE**: MCQA answer extraction failures (config may need adjustment)
- **GPU_OOM**: GPU out-of-memory errors (longer cleanup delay)
- **TEMPLATE_ERROR**: Chat template issues (unlikely to succeed on retry)
- **SERVER_STARTUP_FAILURE**: vLLM server startup problems
- **NETWORK_ERROR**: Connection/refused/timeout issues
- **JUDGE_ERROR**: Judge-specific grading problems
- **UNKNOWN_ERROR**: Unclassified failures

### 2. Resource Management

#### Pre-flight Checks

- **Disk space**: Verifies minimum free space (configurable, default 50GB)
- **GPU memory**: Checks per-GPU free memory (configurable, default 70GB)
- **Model accessibility**: Validates Hugging Face model availability
- **Port availability**: Checks for conflicts on vLLM ports (8000-8002, 5000-5001)

#### GPU Cleanup

- Kills orphaned vLLM processes
- Attempts GPU reset (when permissions allow)
- Runs between all evaluations and retries

### 3. Monitoring and Recovery

#### Progress Tracking

- Background monitoring process
- Periodic status logging every 5 minutes
- Detection of consecutive completion failures
- Status indicators: starting → generating → grading → grading_items

#### State Persistence

- JSON state file with pipeline progress
- Per-baseline state tracking
- Resume capability from any failure point

### 4. Error Recovery

Automatic retry strategy:

1. **First retry**: Standard 60s delay + GPU cleanup
2. **GPU_OOM/EXTRACTION_FAILURE**: Extended 300s delay for GPU recovery
3. **Template errors**: Skipped (unlikely to succeed)
4. **Max retries**: 3 attempts by default (configurable)

## Usage

### Basic Commands

```bash
# Run a single baseline
scripts/robust_evaluation_pipeline.sh --single baseline_4b

# Run multiple baselines in sequence
scripts/robust_evaluation_pipeline.sh --pipeline baseline_4b baseline_9b_think_on baseline_biomistral

# Resume from previous failures
scripts/robust_evaluation_pipeline.sh --resume

# Run pre-flight checks only
scripts/robust_evaluation_pipeline.sh --check-only baseline_4b

# GPU cleanup only
scripts/robust_evaluation_pipeline.sh --cleanup-only

# Show help
scripts/robust_evaluation_pipeline.sh --help
```

### Wrapper Script

Run all baselines with the convenience wrapper:

```bash
scripts/run_baselines_robust.sh
```

### Testing

Validate the pipeline installation:

```bash
scripts/test_pipeline.sh
```

## Configuration

Environment variables (with defaults):

```bash
MEDRL_DIR=/root/medrl              # MedRL installation directory
LOG_DIR=/tmp/eval_pipeline_logs    # Log file directory
STATE_DIR=/tmp/eval_pipeline_state # State persistence directory
REPORT_DIR=/tmp/eval_pipeline_reports # Report output directory
```

Resource thresholds (modify in script):

```bash
MIN_DISK_GB=50                    # Minimum free disk space (GB)
MIN_GPU_MEMORY_GB=70000           # Minimum free GPU memory (MiB)
MAX_STARTUP_TIME_S=600            # Maximum vLLM startup time (seconds)
HEALTH_CHECK_INTERVAL_S=30         # Progress check interval (seconds)
MAX_RETRIES=3                     # Maximum retry attempts
RETRY_DELAY_S=60                  # Standard retry delay (seconds)
EXTRACTION_FAIL_RETRY_DELAY_S=300 # Extended delay for GPU/extraction errors
```

## Output

### Log Files

Located in `LOG_DIR` (default: `/tmp/eval_pipeline_logs/`):

- `pipeline.log`: Master pipeline log
- `<baseline>.log`: Per-baseline evaluation output

### State Files

Located in `STATE_DIR` (default: `/tmp/eval_pipeline_state/`):

- `pipeline_state.json`: Overall pipeline state
- `pipeline.pid`: Running process ID

### Reports

Located in `REPORT_DIR` (default: `/tmp/eval_pipeline_reports/`):

- `final_report_*.txt`: Human-readable text report
- `final_report_*.html`: Detailed HTML report (if `jq` available)

## Monitoring

### Check Pipeline Status

```bash
# Current state
cat /tmp/eval_pipeline_state/pipeline_state.json | jq '.'

# Active processes
ps aux | grep robust_evaluation_pipeline

# GPU utilization
nvidia-smi

# Recent logs
tail -f /tmp/eval_pipeline_logs/pipeline.log
```

### Baseline-Specific Monitoring

```bash
# Follow specific baseline log
tail -f /tmp/eval_pipeline_logs/baseline_4b.log

# Check for errors
grep -i error /tmp/eval_pipeline_logs/baseline_4b.log

# Check progress markers
grep -E "vllm serve|healthy|grading|completed" /tmp/eval_pipeline_logs/baseline_4b.log
```

## Error Troubleshooting

### GPU Out of Memory

**Symptoms**: Exit code with GPU_OOM classification

**Solutions**:
1. Check GPU memory: `nvidia-smi`
2. Increase `MIN_GPU_MEMORY_GB` threshold
3. Run cleanup: `scripts/robust_evaluation_pipeline.sh --cleanup-only`
4. Reduce batch size in cluster config
5. Retry with extended delay

### Extraction Failures

**Symptoms**: Exit code with EXTRACTION_FAILURE classification

**Solutions**:
1. Check `max_extraction_fail_rate` in baseline config
2. Review model's answer contract compliance
3. Consider adjusting sampling parameters
4. May indicate model capability issue

### Server Startup Failure

**Symptoms**: vLLM server fails to start

**Solutions**:
1. Check baseline config syntax
2. Verify model accessibility
3. Check port availability
4. Review server logs: `tail /scratch/medrl/runs/eval-*/serve-policy.log`

## Production Checklist

Before running the pipeline in production:

- [ ] Verify GPU availability (`nvidia-smi`)
- [ ] Check disk space (50GB+ free)
- [ ] Confirm model accessibility
- [ ] Test with single baseline first
- [ ] Set up monitoring/alerting for long-running jobs
- [ ] Configure log rotation for `LOG_DIR`
- [ ] Test resume capability with `--resume`
- [ ] Review and adjust resource thresholds
- [ ] Set up report archival

## Architecture

```
robust_evaluation_pipeline.sh
├── Configuration (environment variables, thresholds)
├── Logging (timestamped, leveled output)
├── State Management (JSON persistence)
├── Error Classification (pattern-based)
├── Resource Checks (disk, GPU, model, ports)
├── GPU Cleanup (orphaned processes, reset)
├── Progress Monitoring (background process)
├── Evaluation Execution (with retry logic)
└── Report Generation (text + HTML)
```

## Exit Codes

- **0**: All baselines completed successfully
- **1**: All baselines failed or critical error
- **2**: Pipeline completed with some failures (partial success)

## Example Workflow

```bash
# 1. Test pipeline installation
scripts/test_pipeline.sh

# 2. Run pre-flight checks for a baseline
scripts/robust_evaluation_pipeline.sh --check-only baseline_4b

# 3. Run single baseline to validate setup
scripts/robust_evaluation_pipeline.sh --single baseline_4b

# 4. Monitor progress in another terminal
tail -f /tmp/eval_pipeline_logs/pipeline.log

# 5. If failure occurs, resume automatically
scripts/robust_evaluation_pipeline.sh --resume

# 6. Review final report
cat /tmp/eval_pipeline_reports/final_report_*.txt
```

## Contributing

When modifying the pipeline:

1. Maintain backward compatibility with environment variables
2. Add tests to `scripts/test_pipeline.sh`
3. Update this documentation
4. Test error scenarios intentionally
5. Validate resume capability

## License

Part of the MedRL project.
