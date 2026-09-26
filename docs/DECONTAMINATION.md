# Contamination Detection for Phase 0

## Overview

The contamination detection system scans evaluation benchmarks for potential data leakage with training data using n-gram overlap analysis. This is a pre-flight check that runs before any GPU work begins.

## Usage

### Basic Configuration

Set the training data paths via environment variable:

```bash
export MEDRL_DECONTAM_TRAINING_PATHS="/path/to/train1.jsonl,/path/to/train2.jsonl"
```

### Running with Decontamination Check

The decontamination check runs automatically during `run_eval()` when training paths are configured:

```python
from medrl.eval.runner import run_eval
from medrl.core.config import EvalConfig

config = EvalConfig(...)
results = run_eval(config)  # Will run decontamination check if MEDRL_DECONTAM_TRAINING_PATHS is set
```

### Report Location

After a run, detailed contamination reports are saved to:
- `<run_dir>/decontamination_report.json` - Detailed per-item statistics
- Console output includes a human-readable summary

## Configuration

The system can be configured via `DecontaminationConfig`:

```python
from medrl.eval.decontam import DecontaminationConfig

config = DecontaminationConfig(
    n_gram_size=13,              # Size of n-grams for overlap detection
    critical_threshold=0.5,     # >50% overlap = critical
    high_threshold=0.2,         # 20-50% overlap = high
    medium_threshold=0.05,       # 5-20% overlap = medium
    low_threshold=0.01,          # 1-5% overlap = low
    fail_on_critical=True,       # Fail run on critical contamination
    training_data_paths=["/path/to/data"],
)
```

## Severity Levels

- **CRITICAL**: >50% n-gram overlap OR >200 character exact match
- **HIGH**: 20-50% overlap OR >100 character exact match
- **MEDIUM**: 5-20% overlap OR >50 character exact match
- **LOW**: 1-5% overlap
- **NONE**: <1% overlap

## Negative Controls

For contaminated items, the system can generate negative control items (perturbed versions) to test whether the model truly knows the material or is memorizing.

## Implementation Details

- Uses word-level 13-grams for texts with sufficient words
- Falls back to character-level n-grams for short texts
- Training data is loaded from JSONL, JSON, or directories
- Supports various data formats (messages, conversations, etc.)
