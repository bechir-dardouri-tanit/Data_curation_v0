"""Preference optimization configuration schemas.

Defines the configuration for DPO (Direct Preference Optimization) and SimPO
(Simple Preference Optimization) training, following the same patterns as the
core configuration schemas.

Both DPO and SimPO are preference optimization methods that train models on
pairs of responses (chosen vs rejected) without requiring a separate reward
model. SimPO is a simplified variant that uses the response length as the
normalization factor.

Reference:
    DPO: https://arxiv.org/abs/2305.18290
    SimPO: https://arxiv.org/abs/2405.14734
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import Field, model_validator

from medrl.core.config import Frozen, ModelConfig, TrackingConfig
from medrl.core.hashing import hash_obj

# --------------------------------------------------------------------------------------
# Algorithm selection
# --------------------------------------------------------------------------------------


class PrefOptMethod(StrEnum):
    """Preference optimization algorithm."""

    DPO = "dpo"  # Direct Preference Optimization
    SimPO = "simpo"  # Simple Preference Optimization


class LengthNorm(StrEnum):
    """How to normalize logits by length."""

    NONE = "none"  # No length normalization
    MEAN = "mean"  # Divide by sequence length (DPO default)
    SIMPO = "simpo"  # SimPO normalization: length penalty term


class PairSampling(StrEnum):
    """Strategy for sampling preference pairs during training."""

    RANDOM = "random"  # Uniform random sampling
    BALANCED = "balanced"  # Ensure equal chosen/rejected per batch
    HARD_NEGATIVES = "hard_negatives"  # Prioritize hard negatives
    STRATIFIED = "stratified"  # Sample proportionally across sources


# --------------------------------------------------------------------------------------
# Preference optimization config
# --------------------------------------------------------------------------------------


class PrefOptConfig(Frozen):
    """Configuration for preference optimization training.

    Supports both DPO and SimPO with configurable hyperparameters.

    Attributes:
        method: The preference optimization algorithm to use.
        beta: Temperature parameter controlling the strength of the preference
            penalty (lower = stronger preference). Typical values: 0.05-0.1 for DPO,
            0.1-0.2 for SimPO.
        length_norm: How to normalize by sequence length.
        sft_loss_weight: Weight for the SFT (supervised fine-tuning) loss component.
            When > 0, adds a cross-entropy loss on the chosen responses (RPO-style).
        pair_sampling: Strategy for sampling preference pairs.
        max_length: Maximum sequence length for training.
        max_prompt_length: Maximum prompt length (truncates if exceeded).
        max_completion_length: Maximum completion length for chosen/rejected.
    """

    # Algorithm selection
    method: PrefOptMethod = PrefOptMethod.DPO
    beta: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1
    length_norm: LengthNorm = LengthNorm.MEAN

    # SFT loss component (RPO-style anchor)
    sft_loss_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    # Sampling strategy
    pair_sampling: PairSampling = PairSampling.RANDOM
    shuffle_buffer_size: Annotated[int, Field(ge=1)] = 10_000

    # Length constraints
    max_length: Annotated[int, Field(ge=128)] = 4096
    max_prompt_length: Annotated[int, Field(ge=1)] = 1024
    max_completion_length: Annotated[int, Field(ge=1)] = 2048

    # Data
    train_path: str = Field(description="Path to training preferences JSONL")
    val_path: str | None = Field(
        default=None, description="Path to validation preferences JSONL"
    )
    num_examples: int | None = Field(
        default=None, description="Limit training examples (for debugging)"
    )

    # Beta tuning
    beta_tune: bool = False
    beta_tune_steps: Annotated[int, Field(ge=1)] = 100
    beta_tune_range: tuple[float, float] = (0.05, 0.2)

    # Training
    learning_rate: Annotated[float, Field(gt=0.0)] = 5e-7
    warmup_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    epochs: Annotated[int, Field(ge=1)] = 1

    # Logging
    log_interval: Annotated[int, Field(ge=1)] = 10
    save_interval: Annotated[int, Field(ge=1)] = 100
    eval_interval: Annotated[int, Field(ge=1)] = 100

    @model_validator(mode="after")
    def _validate_method_defaults(self) -> Self:
        """Set sensible defaults based on the method."""
        if self.method == PrefOptMethod.SimPO and self.length_norm == LengthNorm.MEAN:
            # SimPO benefits from its specific length normalization
            self.object_setattr("length_norm", LengthNorm.SIMPO)
        return self

    @model_validator(mode="after")
    def _validate_length_constraints(self) -> Self:
        """Ensure length constraints are internally consistent."""
        if self.max_prompt_length + self.max_completion_length > self.max_length:
            raise ValueError(
                f"max_prompt_length ({self.max_prompt_length}) + "
                f"max_completion_length ({self.max_completion_length}) cannot exceed "
                f"max_length ({self.max_length})"
            )
        return self

    @model_validator(mode="after")
    def _validate_beta_tuning(self) -> Self:
        """Validate beta tuning configuration."""
        if self.beta_tune:
            low, high = self.beta_tune_range
            if not (0.0 < low < high <= 1.0):
                raise ValueError(
                    f"beta_tune_range must be (0, 1) with low < high, got {self.beta_tune_range}"
                )
        return self

    @property
    def effective_beta(self) -> float:
        """The beta value to use (after tuning if enabled)."""
        return self.beta

    def object_setattr(self, key: str, value: object) -> None:
        """Helper to set attributes on frozen model during validation."""
        object.__setattr__(self, key, value)


class PrefTrainConfig(Frozen):
    """Complete configuration for a preference optimization training run.

    Combines model, cluster, optimization, and tracking configurations.
    """

    model: ModelConfig
    pref: PrefOptConfig
    tracking: TrackingConfig = TrackingConfig()

    # Optional checkpointing
    output_dir: str = "outputs/pref"
    checkpoint_dir: str | None = None
    resume_from_checkpoint: bool = False

    # Optional experiment metadata
    run_name: str | None = None
    seed: int = 42

    @property
    def fingerprint(self) -> str:
        """Stable content hash of this configuration."""
        return hash_obj(self.model_dump(mode="json"))


# --------------------------------------------------------------------------------------
# Model-specific presets
# --------------------------------------------------------------------------------------


def dpo_preset(**overrides: Any) -> PrefOptConfig:
    """Create a DPO configuration with sensible defaults.

    DPO typically uses:
    - beta around 0.05-0.1
    - Mean length normalization
    - No SFT loss (pure preference)

    Args:
        **overrides: Any fields to override from defaults.

    Returns:
        A DPO configuration.
    """
    # Set default train_path if not provided
    if "train_path" not in overrides:
        overrides = {"train_path": "data/prefs/train.jsonl", **overrides}
    return PrefOptConfig(
        method=PrefOptMethod.DPO,
        beta=0.1,
        length_norm=LengthNorm.MEAN,
        sft_loss_weight=0.0,
        **overrides,
    )


def simpo_preset(**overrides: Any) -> PrefOptConfig:
    """Create a SimPO configuration with sensible defaults.

    SimPO typically uses:
    - beta around 0.1-0.2 (higher than DPO)
    - SimPO-specific length normalization
    - Optional SFT loss for stability

    Args:
        **overrides: Any fields to override from defaults.

    Returns:
        A SimPO configuration.
    """
    # Set default train_path if not provided
    if "train_path" not in overrides:
        overrides = {"train_path": "data/prefs/train.jsonl", **overrides}
    return PrefOptConfig(
        method=PrefOptMethod.SimPO,
        beta=0.15,
        length_norm=LengthNorm.SIMPO,
        sft_loss_weight=0.0,
        **overrides,
    )


def rpo_preset(**overrides: Any) -> PrefOptConfig:
    """Create an RPO (Reward Policy Optimization) configuration.

    RPO is DPO with an additional SFT loss component for stability.

    Args:
        **overrides: Any fields to override from defaults.

    Returns:
        An RPO configuration.
    """
    # Set default train_path if not provided
    if "train_path" not in overrides:
        overrides = {"train_path": "data/prefs/train.jsonl", **overrides}
    return PrefOptConfig(
        method=PrefOptMethod.DPO,
        beta=0.1,
        length_norm=LengthNorm.MEAN,
        sft_loss_weight=0.2,  # Key RPO difference
        **overrides,
    )


def load_pref_config(path: str | Path) -> PrefTrainConfig:
    """Load a preference training config from a YAML file.

    Args:
        path: Path to YAML config file or preset name.

    Returns:
        Validated PrefTrainConfig instance.
    """
    import yaml

    from medrl.core.config import validate_dict
    from medrl.core.paths import repo_root

    def resolve(spec: str | Path) -> Path:
        candidate = Path(spec)
        if candidate.suffix in {".yaml", ".yml"}:
            return candidate if candidate.is_absolute() else repo_root() / candidate
        # Try as preset
        preset = repo_root() / "configs" / "pref" / f"{candidate}.yaml"
        if preset.is_file():
            return preset
        raise FileNotFoundError(f"No preference config found at {spec}")

    path = resolve(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return validate_dict(PrefTrainConfig, raw)  # type: ignore[no-any-return]
