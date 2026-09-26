"""Supervised Fine-Tuning (SFT) infrastructure.

This module provides production-ready SFT training with:
- FSDP2 distributed training
- Sequence packing (16-32k context)
- Assistant-only loss masking
- Thinking mode support
- Gradient clipping
- W&B integration
- Checkpoint management

Typical usage:
    >>> from medrl.train.sft import load_sft_config, run_sft
    >>> config = load_sft_config("configs/sft/base_qwen.yaml")
    >>> result = run_sft(config)
"""

from medrl.train.sft.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointState,
)
from medrl.train.sft.config import (
    SequencePackingMode,
    SFTCheckpointConfig,
    SFTConfig,
    SFTDataConfig,
    SFTLossConfig,
    SFTOptimizerConfig,
    SFTSchedulerConfig,
    load_sft_config,
)
from medrl.train.sft.data import (
    PackedSequence,
    SFTCollator,
    SFTDataset,
    SFTExample,
    create_sft_dataloader,
    estimate_tokens_per_example,
    load_json,
    load_jsonl,
    load_parquet,
)
from medrl.train.sft.runner import (
    EvalMetrics,
    SFTRunner,
    StreamingDataIterator,
    TrainingMetrics,
    run_sft,
)

__all__ = [
    # Checkpoint
    "CheckpointManager",
    "CheckpointMetadata",
    "CheckpointState",
    "EvalMetrics",
    "PackedSequence",
    "SFTCheckpointConfig",
    "SFTCollator",
    # Config
    "SFTConfig",
    "SFTDataConfig",
    # Data
    "SFTDataset",
    "SFTExample",
    "SFTLossConfig",
    "SFTOptimizerConfig",
    # Runner
    "SFTRunner",
    "SFTSchedulerConfig",
    "SequencePackingMode",
    "StreamingDataIterator",
    "TrainingMetrics",
    "create_sft_dataloader",
    "estimate_tokens_per_example",
    "load_json",
    "load_jsonl",
    "load_parquet",
    "load_sft_config",
    "run_sft",
]
