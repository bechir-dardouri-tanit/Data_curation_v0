"""Preference optimization (DPO/SimPO) training infrastructure.

This module provides a complete implementation of Direct Preference Optimization
(DPO) and Simple Preference Optimization (SimPO) for training language models
from preference pairs.

Key components:
    - PrefOptConfig: Configuration for preference optimization
    - PrefTrainConfig: Complete training configuration
    - PrefLoss: DPO/SimPO loss implementations
    - PrefDataset: Dataset for loading preference pairs
    - run_pref_train: Main training loop

Example usage:
    >>> from medrl.train.pref import (
    ...     PrefOptConfig,
    ...     PrefTrainConfig,
    ...     run_pref_train,
    ...     dpo_preset,
    ...     simpo_preset,
    ... )
    >>> from medrl.core.config import ModelConfig, ClusterConfig
    >>>
    >>> # Configure training
    >>> config = PrefTrainConfig(
    ...     model=ModelConfig(hf_id="Qwen/Qwen2.5-0.5B"),
    ...     pref=dpo_preset(
    ...         beta=0.1,
    ...         train_path="data/prefs.jsonl",
    ...         learning_rate=5e-7,
    ...     ),
    ... )
    >>> cluster = ClusterConfig(name="local", num_gpus=2)
    >>>
    >>> # Run training
    >>> state = run_pref_train(config, cluster)

References:
    DPO: https://arxiv.org/abs/2305.18290
    SimPO: https://arxiv.org/abs/2405.14734
"""

from __future__ import annotations

# Configuration
from medrl.train.pref.config import (
    LengthNorm,
    PairSampling,
    PrefOptConfig,
    PrefOptMethod,
    PrefTrainConfig,
    dpo_preset,
    rpo_preset,
    simpo_preset,
)

# Data loading
from medrl.train.pref.data import (
    PrefBatch,
    PrefCollator,
    PrefDataset,
    PrefSampler,
    create_dataloader,
)

# Loss functions
from medrl.train.pref.loss import (
    PrefLoss,
    PrefLossOutput,
    compute_grad_norm,
    compute_log_probs,
    dpo_loss,
    sft_loss,
    simpo_loss,
)

# Training runner
from medrl.train.pref.runner import (
    BetaTuner,
    CheckpointManager,
    PrefTrackingSink,
    TrainMetrics,
    TrainState,
    run_pref_train,
)

__all__ = [
    "BetaTuner",
    "CheckpointManager",
    "LengthNorm",
    "PairSampling",
    "PrefBatch",
    "PrefCollator",
    # Data
    "PrefDataset",
    # Loss
    "PrefLoss",
    "PrefLossOutput",
    # Configuration
    "PrefOptConfig",
    "PrefOptMethod",
    "PrefSampler",
    "PrefTrackingSink",
    "PrefTrainConfig",
    "TrainMetrics",
    "TrainState",
    "compute_grad_norm",
    "compute_log_probs",
    "create_dataloader",
    "dpo_loss",
    "dpo_preset",
    "rpo_preset",
    # Training
    "run_pref_train",
    "sft_loss",
    "simpo_loss",
    "simpo_preset",
]
