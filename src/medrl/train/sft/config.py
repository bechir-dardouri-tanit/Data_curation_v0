"""SFT configuration schemas.

Configuration for supervised fine-tuning, building on the core config patterns.
Supports FSDP2 training, sequence packing, and Axolotl-style config loading.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from medrl.core.config import (
    ClusterConfig,
    Frozen,
    ModelConfig,
    ThinkingConfig,
    TrackingConfig,
)


class SFTOptimizerType(StrEnum):
    """Optimizer type for SFT training."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"


class SchedulerType(StrEnum):
    """Learning rate scheduler type."""

    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    POLYNOMIAL = "polynomial"


class SequencePackingMode(StrEnum):
    """How to pack sequences into the context window."""

    NONE = "none"  # No packing, one example per batch
    CONCAT = "concat"  # Concatenate examples with separator tokens
    SMART = "smart"  # Optimize packing to minimize waste


class SFTOptimizerConfig(Frozen):
    """Optimizer configuration for SFT training."""

    type: SFTOptimizerType = SFTOptimizerType.ADAMW
    lr: Annotated[float, Field(gt=0, le=10)] = 5e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: Annotated[float, Field(gt=0)] = 1e-8
    weight_decay: Annotated[float, Field(ge=0)] = 0.01

    # Adam-specific
    adam_beta1: float | None = None
    adam_beta2: float | None = None

    @model_validator(mode="after")
    def _use_adam_betas(self) -> Self:
        # Normalize adam_beta1/beta2 to betas
        if self.adam_beta1 is not None or self.adam_beta2 is not None:
            if self.type not in (SFTOptimizerType.ADAMW, SFTOptimizerType.ADAM):
                raise ValueError("adam_beta1/beta2 only valid for adam/adamw")
            b1 = self.adam_beta1 or self.betas[0]
            b2 = self.adam_beta2 or self.betas[1]
            object.__setattr__(self, "betas", (b1, b2))
        return self


class SFTSchedulerConfig(Frozen):
    """Learning rate scheduler configuration."""

    type: SchedulerType = SchedulerType.COSINE
    warmup_ratio: Annotated[float, Field(ge=0, le=1)] = 0.03
    warmup_steps: int | None = None  # If set, overrides warmup_ratio
    min_lr_ratio: Annotated[float, Field(ge=0, le=1)] = 0.1
    num_cycles: Annotated[float, Field(gt=0)] = 0.5  # For cosine

    @property
    def actual_warmup_steps(self) -> int | None:
        """Return the warmup steps, preferring explicit over ratio-based."""
        return self.warmup_steps


class SFTDataConfig(Frozen):
    """Data loading configuration."""

    train_path: str  # Path to training data (JSONL, Parquet, or directory)
    validation_path: str | None = None  # Path to validation data
    test_path: str | None = None  # Path to test data

    # Axolotl-style data spec
    dataset_type: Literal["json", "jsonl", "parquet", "hub"] = "jsonl"
    conversation_format: Literal["sharegpt", "openai", "custom"] = "openai"
    data_files: tuple[str, ...] = ()  # Additional data files to include

    # Sequence handling
    max_seq_len: Annotated[int, Field(ge=512)] = 8192
    packing: SequencePackingMode = SequencePackingMode.SMART
    packing_separator: str = ""  # Separator token between packed sequences

    # Data quality filters
    min_response_length: int | None = None
    max_response_length: Annotated[int, Field(ge=1)] = 32768
    filter_empty: bool = True
    truncation: Literal["error", "truncate", "skip"] = "truncate"

    # Augmentation
    shuffle: bool = True
    seed: int = 42

    # Multiprocessing
    num_workers: Annotated[int, Field(ge=0)] = 4
    prefetch_factor: Annotated[int, Field(ge=1)] = 2

    @property
    def max_context_len(self) -> int:
        """Alias for compatibility with thinking-aware context length."""
        return self.max_seq_len


class SFTLossConfig(Frozen):
    """Loss computation configuration."""

    # Assistant-only masking
    mask_user_tokens: bool = True  # Only compute loss on assistant tokens
    mask_system_tokens: bool = True  # Also mask system tokens
    mask_response_separator: bool = False  # Mask separator tokens in packed data

    # Thinking mode handling
    include_thinking_in_loss: bool = True  # Include <think > content in loss
    separate_answer_loss: bool = False  # Track thinking vs answer loss separately

    # Loss scaling
    label_smoothing: Annotated[float, Field(ge=0, le=1)] = 0.0
    ignore_index: int = -100  # PyTorch convention for ignored tokens


class SFTCheckpointConfig(Frozen):
    """Checkpoint management configuration."""

    save_dir: str = "checkpoints"
    save_total_limit: Annotated[int, Field(ge=0)] = 3  # 0 = unlimited
    save_steps: int | None = None  # Save every N steps
    save_epochs: int | None = None  # Save every N epochs
    save_strategy: Literal["steps", "epochs", "best"] = "best"

    # Validation
    eval_steps: int | None = None
    eval_epochs: int | None = None
    metric_for_best_model: str = "loss"  # Metric to monitor for best model
    greater_is_better: bool = False

    # Resumption
    resume_from_checkpoint: str | None = None
    auto_resume: bool = True  # Auto-resume from latest checkpoint in save_dir

    # Best model tracking
    load_best_model_at_end: bool = True

    @property
    def should_save_steps(self) -> bool:
        return self.save_strategy == "steps" and self.save_steps is not None

    @property
    def should_save_epochs(self) -> bool:
        return self.save_strategy == "epochs" and self.save_epochs is not None


class SFTConfig(Frozen):
    """Complete SFT training configuration.

    This config extends the base patterns with SFT-specific settings while
    maintaining the topology-invariant design (global batch sizes, cluster presets).
    """

    # Model and cluster
    model: ModelConfig
    cluster: ClusterConfig

    # Data
    data: SFTDataConfig

    # Training hyperparameters (global, topology-invariant)
    max_steps: int | None = None  # Maximum training steps (None = epochs only)
    num_epochs: Annotated[int, Field(ge=1)] = 1
    global_batch_size: Annotated[int, Field(ge=1)] = 512  # Number of examples

    # Optimization
    optimizer: SFTOptimizerConfig = SFTOptimizerConfig()
    scheduler: SFTSchedulerConfig = SFTSchedulerConfig()
    max_grad_norm: Annotated[float, Field(ge=0)] = 1.0  # Gradient clipping

    # Loss
    loss: SFTLossConfig = SFTLossConfig()

    # Thinking mode (optional for SFT)
    thinking: ThinkingConfig | None = None  # None = disabled

    # Checkpointing
    checkpoint: SFTCheckpointConfig = SFTCheckpointConfig()

    # Tracking
    tracking: TrackingConfig = TrackingConfig()

    # Output
    output_dir: str | None = None  # Defaults to runs_dir()/sft/{run_id}
    run_name: str | None = None  # Auto-generated if None
    seed: int = 42

    # Advanced
    gradient_checkpointing: bool = True  # Override cluster setting if True
    fsdp_sharding_strategy: Literal["full_shard", "shard_grad_op", "no_shard"] = "full_shard"
    fsdp_offload_params: bool = False  # Override cluster setting
    bf16: bool = True  # Use bfloat16 mixed precision

    @model_validator(mode="after")
    def _validate_steps_vs_epochs(self) -> Self:
        if self.max_steps is None and self.num_epochs is None:
            raise ValueError("At least one of max_steps or num_epochs must be set")
        return self

    @model_validator(mode="after")
    def _validate_checkpoint_save(self) -> Self:
        ckpt = self.checkpoint
        # Only validate steps/epochs strategies when save_steps/save_epochs are explicitly None
        # "best" strategy doesn't require them
        if ckpt.save_strategy == "steps" and ckpt.save_steps is None:
            raise ValueError("save_strategy=steps requires save_steps to be set")
        if ckpt.save_strategy == "epochs" and ckpt.save_epochs is None:
            raise ValueError("save_strategy=epochs requires save_epochs to be set")
        return self

    @property
    def effective_activation_checkpointing(self) -> bool:
        """Whether to use activation checkpointing (explicit wins over cluster)."""
        return self.gradient_checkpointing

    @property
    def effective_param_offload(self) -> bool:
        """Whether to offload parameters (explicit wins over cluster)."""
        return self.fsdp_offload_params or self.cluster.param_offload


def load_sft_config(path: str | Path) -> SFTConfig:
    """Load an SFT config from a YAML file.

    Supports Axolotl-style config format with automatic field mapping.
    Expands cluster and model presets from configs/cluster/ and configs/model/.

    Args:
        path: Path to YAML config file or preset name.

    Returns:
        Validated SFTConfig instance.
    """
    import yaml

    from medrl.core.config import validate_dict
    from medrl.core.paths import repo_root

    def resolve(spec: str | Path) -> Path:
        candidate = Path(spec)
        if candidate.suffix in {".yaml", ".yml"}:
            return candidate if candidate.is_absolute() else repo_root() / candidate
        # Try as preset
        preset = repo_root() / "configs" / "sft" / f"{candidate}.yaml"
        if preset.is_file():
            return preset
        raise FileNotFoundError(f"No SFT config found at {spec}")

    path = resolve(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Expand cluster preset
    if "cluster" in raw:
        cluster_spec = raw["cluster"]
        if isinstance(cluster_spec, str):
            cluster_path = repo_root() / "configs" / "cluster" / f"{cluster_spec}.yaml"
            if cluster_path.is_file():
                raw["cluster"] = yaml.safe_load(cluster_path.read_text())
            else:
                raise FileNotFoundError(f"Cluster preset not found: {cluster_spec}")

    # Expand model preset
    if "model" in raw:
        model_spec = raw["model"]
        if isinstance(model_spec, str):
            model_path = repo_root() / "configs" / "model" / f"{model_spec}.yaml"
            if model_path.is_file():
                raw["model"] = yaml.safe_load(model_path.read_text())
            else:
                # Try as direct HF id
                raw["model"] = {"hf_id": model_spec}

    # Axolotl field mapping (common alternative field names)
    axolotl_map = {
        "learning_rate": ("optimizer", "lr"),
        "optimizer": ("optimizer", "type"),
        "weight_decay": ("optimizer", "weight_decay"),
        "warmup_ratio": ("scheduler", "warmup_ratio"),
        "warmup_steps": ("scheduler", "warmup_steps"),
        "lr_scheduler": ("scheduler", "type"),
        "gradient_accumulation_steps": ("cluster", "grad_accum_steps"),
        "micro_batch_size": ("cluster", "micro_batch_size"),
        "train_batch_size": ("global_batch_size",),
        "eval_batch_size": ("cluster", "micro_batch_size"),  # For validation
        "save_total_limit": ("checkpoint", "save_total_limit"),
        "save_steps": ("checkpoint", "save_steps"),
        "eval_steps": ("checkpoint", "eval_steps"),
        "max_seq_length": ("data", "max_seq_len"),
        "dataset": ("data", "train_path"),
        "val_set": ("data", "validation_path"),
        "bf16": ("bf16",),
        "gradient_checkpointing": ("gradient_checkpointing",),
    }

    # Apply Axolotl field mapping
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        if key in axolotl_map:
            field_path = axolotl_map[key]
            if len(field_path) == 1:
                mapped[field_path[0]] = value
            else:
                # Nested mapping
                parent = mapped.setdefault(field_path[0], {})
                if isinstance(parent, dict):
                    parent[field_path[1]] = value
        else:
            mapped[key] = value

    return validate_dict(SFTConfig, mapped)  # type: ignore[no-any-return]
