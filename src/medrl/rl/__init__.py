"""Reinforcement Learning infrastructure for medical LLM post-training.

This module implements GSPO (Group-wise Sequence Preference Optimization) and
DAPO (Direct Advantage Policy Optimization) for training medical reasoning models.

Main components:
- :mod:`medrl.rl.config`: Configuration schemas for RL training
- :mod:`medrl.rl.rewards`: Reward function implementations
- :mod:`medrl.rl.samplers`: Rollout sampling strategies
- :mod:`medrl.rl.runner`: Main RL training loop

Example usage:
    >>> from medrl.rl.config import RLTrainingConfig, RLAlgorithm
    >>> from medrl.rl.runner import RLRunner
    >>> config = RLTrainingConfig(model=..., algorithm=RLAlgorithm.GSPO)
    >>> runner = RLRunner(config, cluster_config, run_dir)
    >>> result = runner.run()
"""

from __future__ import annotations

from medrl.rl.config import (
    ColocatedRolloutConfig,
    RewardConfig,
    RLAlgorithm,
    RLTrainingConfig,
    RolloutSamplingConfig,
)
from medrl.rl.rewards import (
    RewardResult,
    compute_format_reward,
    compute_group_rewards,
    compute_length_reward,
    compute_rubric_reward,
    compute_safety_reward,
    compute_total_reward,
    compute_verifier_reward,
)
from medrl.rl.runner import RLResult, RLRunner, RolloutError, RolloutRecord, TrainingError
from medrl.rl.samplers import (
    RolloutSampler,
    SampleResult,
    TemperatureScheduler,
    create_preference_pairs,
    filter_groups,
    sample_group,
    sample_group_stratified,
)

__all__ = [
    "ColocatedRolloutConfig",
    "RLAlgorithm",
    "RLResult",
    "RLRunner",
    "RLTrainingConfig",
    "RewardConfig",
    "RewardResult",
    "RolloutError",
    "RolloutRecord",
    "RolloutSampler",
    "RolloutSamplingConfig",
    "SampleResult",
    "TemperatureScheduler",
    "TrainingError",
    "compute_format_reward",
    "compute_group_rewards",
    "compute_length_reward",
    "compute_rubric_reward",
    "compute_safety_reward",
    "compute_total_reward",
    "compute_verifier_reward",
    "create_preference_pairs",
    "filter_groups",
    "sample_group",
    "sample_group_stratified",
]
