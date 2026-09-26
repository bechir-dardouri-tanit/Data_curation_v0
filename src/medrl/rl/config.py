"""RL configuration schemas.

GSPO (Group-wise Sequence Preference Optimization) and DAPO (Direct Advantage Policy Optimization)
training configurations. All configs are frozen and hashable, following the pattern from
:mod:`medrl.core.config`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from medrl.core.config import Frozen, ModelConfig, SamplingConfig, ThinkingConfig

# --------------------------------------------------------------------------------------
# RL algorithms
# --------------------------------------------------------------------------------------


class RLAlgorithm(StrEnum):
    """RL algorithm to use for training."""

    GSPO = "gspo"
    """Group-wise Sequence Preference Optimization (sequence-level, preference-based)."""

    DAPO = "dapo"
    """Direct Advantage Policy Optimization (token-level, value-based fallback)."""


# --------------------------------------------------------------------------------------
# Reward configuration
# --------------------------------------------------------------------------------------


class RewardConfig(Frozen):
    """Reward function configuration for RL training.

    Rewards are computed from the same verifiers used in evaluation, ensuring that the
    training target matches the reported metric.
    """

    verifier_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    """Weight for correctness rewards (MCQA, numeric, rubric)."""

    format_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    """Weight for format rule compliance."""

    length_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    """Weight for length shaping rewards."""

    safety_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    """Weight for safety/deferral rewards."""

    # Length shaping parameters
    min_length: int = 50
    """Minimum response length for full reward (shorter responses are penalized)."""

    max_length: int = 8192
    """Maximum response length for full reward (longer responses are penalized)."""

    # Safety/deferral parameters
    deferral_phrases: tuple[str, ...] = (
        "consult",
        "seek medical advice",
        "not a substitute",
        "healthcare professional",
    )
    """Phrases that indicate appropriate deferral/uncertainty."""

    harm_keywords: tuple[str, ...] = ()
    """Keywords that indicate potentially harmful advice (reduces reward)."""

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> RewardConfig:
        total = self.verifier_weight + self.format_weight + self.length_weight + self.safety_weight
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Reward weights must sum to ~1.0, got {total:.3f} "
                f"(verifier={self.verifier_weight}, format={self.format_weight}, "
                f"length={self.length_weight}, safety={self.safety_weight})"
            )
        return self


# --------------------------------------------------------------------------------------
# Sampling configuration for rollouts
# --------------------------------------------------------------------------------------


class RolloutSamplingConfig(Frozen):
    """Sampling configuration for RL rollouts.

    Extends the base SamplingConfig with RL-specific options like dynamic
    sampling (dropping all-correct/all-wrong groups) and temperature scheduling.
    """

    base: SamplingConfig = SamplingConfig()

    # Dynamic sampling
    drop_all_correct: bool = True
    """Drop groups where all responses are correct (no learning signal)."""

    drop_all_wrong: bool = True
    """Drop groups where all responses are incorrect (no learning signal)."""

    min_correct_per_group: int = 1
    """Minimum number of correct responses required in a group."""

    # Temperature scheduling
    temperature_schedule: Literal["constant", "decay", "warmup"] = "constant"
    """How to adjust temperature during training."""

    initial_temperature: float = 0.8
    """Starting temperature for decay/warmup schedules."""

    final_temperature: float = 0.5
    """Target temperature for decay/warmup schedules."""

    temperature_warmup_steps: int = 100
    """Steps for warmup schedule."""

    temperature_decay_steps: int = 1000
    """Steps for decay schedule."""

    # Group-based sampling
    group_size: int = 8
    """Number of responses per prompt for group-wise training."""

    @model_validator(mode="after")
    def _validate_temperature_schedule(self) -> RolloutSamplingConfig:
        if self.temperature_schedule == "decay" and self.final_temperature >= self.initial_temperature:
            raise ValueError(
                "temperature_schedule=decay requires final_temperature < initial_temperature"
            )
        if self.temperature_schedule == "warmup" and self.final_temperature <= self.initial_temperature:
            raise ValueError(
                "temperature_schedule=warmup requires final_temperature > initial_temperature"
            )
        return self


# --------------------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------------------


class RLTrainingConfig(Frozen):
    """Main RL training configuration.

    Defines the training loop, rollout generation, and policy update parameters.
    Follows the same pattern as EvalConfig for consistency.
    """

    model: ModelConfig
    """The policy model to train."""

    algorithm: RLAlgorithm = RLAlgorithm.GSPO
    """RL algorithm to use."""

    # Rollout generation
    rollout_sampling: RolloutSamplingConfig = RolloutSamplingConfig()
    """Sampling parameters for rollout generation."""

    thinking: ThinkingConfig = ThinkingConfig()
    """Thinking/reasoning budget for rollouts."""

    n_rollouts: Annotated[int, Field(ge=1)] = 1024
    """Number of prompts per rollout batch."""

    rollout_buffer_size: Annotated[int, Field(ge=1)] = 10_000
    """Maximum number of rollouts to buffer before training."""

    # Training parameters
    learning_rate: Annotated[float, Field(gt=0.0)] = 1e-5
    """Learning rate for policy updates."""

    max_epochs: Annotated[int, Field(ge=1)] = 10
    """Maximum training epochs."""

    batch_size: Annotated[int, Field(ge=1)] = 64
    """Training batch size."""

    gradient_accumulation_steps: Annotated[int, Field(ge=1)] = 1
    """Gradient accumulation steps."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    # Reward configuration
    reward: RewardConfig = RewardConfig()
    """Reward function configuration."""

    # KL penalty (for DAPO-style algorithms)
    kl_penalty_coeff: float = 0.1
    """KL divergence penalty coefficient."""

    kl_target: float = 0.05
    """Target KL divergence for adaptive penalty."""

    # Advantage estimation (for DAPO)
    gamma: float = 0.99
    """Discount factor for advantage estimation."""

    gae_lambda: float = 0.95
    """GAE lambda for advantage estimation."""

    # Checkpointing
    checkpoint_dir: str | None = None
    """Directory to save checkpoints."""

    checkpoint_interval: int = 100
    """Save a checkpoint every N steps."""

    # Monitoring
    log_interval: int = 10
    """Log metrics every N steps."""

    eval_interval: int = 500
    """Run evaluation every N steps."""

    # Failure handling
    max_consecutive_failures: int = 10
    """Stop after this many consecutive rollout failures."""

    rollout_timeout_s: float = 600.0
    """Timeout for a single rollout batch."""

    resume_from_checkpoint: str | None = None
    """Path to checkpoint to resume from."""

    @model_validator(mode="after")
    def _validate_algorithm_requirements(self) -> RLTrainingConfig:
        if self.algorithm is RLAlgorithm.DAPO and self.kl_penalty_coeff <= 0:
            raise ValueError("DAPO requires kl_penalty_coeff > 0")
        return self


# --------------------------------------------------------------------------------------
# Colocated rollout configuration
# --------------------------------------------------------------------------------------


class ColocatedRolloutConfig(Frozen):
    """Configuration for colocated rollout serving (vLLM shares GPUs with training).

    The rollout engine and trainer share the same GPUs, time-sliced according to
    the rollout placement strategy (from ClusterConfig).
    """

    enabled: bool = False
    """Enable colocated rollout (default: sequential rollout)."""

    rollout_port: int | None = None
    """Port for the colocated vLLM server."""

    rollout_max_model_len: int = 40960
    """Maximum context length for rollouts."""

    rollout_gpu_memory_utilization: float = 0.5
    """GPU memory fraction for rollout engine (remainder for training)."""

    rollout_sleep_between_phases: bool = True
    """Sleep the rollout engine between training phases to free memory."""

    @model_validator(mode="after")
    def _validate_memory_fraction(self) -> ColocatedRolloutConfig:
        if not (0.1 <= self.rollout_gpu_memory_utilization <= 0.9):
            raise ValueError(
                f"rollout_gpu_memory_utilization must be in [0.1, 0.9], "
                f"got {self.rollout_gpu_memory_utilization}"
            )
        return self
