"""Rollout sampling strategies for RL training.

Implements dynamic sampling, group-based sampling, and temperature scheduling
for generating training rollouts.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medrl.rl.config import RolloutSamplingConfig


@dataclass(frozen=True)
class SampleResult:
    """Result of sampling a group of responses.

    Contains the sampled indices and any metadata about why certain samples
    were included or excluded.
    """

    indices: list[int]
    """Indices of selected samples."""

    dropped_all_correct: bool = False
    """Whether group was dropped for all samples being correct."""

    dropped_all_wrong: bool = False
    """Whether group was dropped for all samples being wrong."""

    n_correct: int = 0
    """Number of correct samples in the group."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for logging."""


@dataclass
class TemperatureScheduler:
    """Temperature scheduler for rollout sampling.

    Supports constant, decay, and warmup schedules.
    """

    schedule: str = "constant"
    """Schedule type: constant, decay, or warmup."""

    initial: float = 0.8
    """Initial temperature."""

    final: float = 0.5
    """Target temperature."""

    warmup_steps: int = 100
    """Steps for warmup schedule."""

    decay_steps: int = 1000
    """Steps for decay schedule."""

    _step: int = field(default=0, init=False)

    def step(self) -> float:
        """Get temperature for current step and advance.

        Returns:
            Temperature for this step.
        """
        if self.schedule == "constant":
            temp = self.initial
        elif self.schedule == "decay":
            progress = min(1.0, self._step / self.decay_steps)
            temp = self.initial - (self.initial - self.final) * progress
        elif self.schedule == "warmup":
            progress = min(1.0, self._step / self.warmup_steps)
            temp = self.initial + (self.final - self.initial) * progress
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        self._step += 1
        return temp

    @property
    def current_step(self) -> int:
        """Current step number."""
        return self._step

    def reset(self) -> None:
        """Reset the scheduler to step 0."""
        self._step = 0


def sample_group(
    rewards: Sequence[float],
    config: RolloutSamplingConfig,
    seed: int | None = None,
) -> SampleResult:
    """Sample a group from rewards with dynamic filtering.

    Implements the "drop all-correct/all-wrong" strategy: groups where all
    responses are correct or all are wrong provide no learning signal and are
    dropped.

    Args:
        rewards: Sequence of reward values for the group.
        config: Sampling configuration.
        seed: Optional random seed.

    Returns:
        SampleResult with selected indices and metadata.
    """
    if seed is not None:
        random.seed(seed)

    n_correct = sum(r > 1.0 for r in rewards)
    n_wrong = len(rewards) - n_correct

    # Check if group should be dropped
    if config.drop_all_correct and n_correct == len(rewards):
        return SampleResult(
            indices=[],
            dropped_all_correct=True,
            n_correct=n_correct,
            metadata={"reason": "all_correct"},
        )

    if config.drop_all_wrong and n_wrong == len(rewards):
        return SampleResult(
            indices=[],
            dropped_all_wrong=True,
            n_correct=n_correct,
            metadata={"reason": "all_wrong"},
        )

    # Check minimum correct threshold
    if n_correct < config.min_correct_per_group:
        return SampleResult(
            indices=[],
            n_correct=n_correct,
            metadata={"reason": "below_min_correct", "min_required": config.min_correct_per_group},
        )

    # All samples pass - return all indices
    return SampleResult(
        indices=list(range(len(rewards))),
        n_correct=n_correct,
        metadata={"kept": "all"},
    )


def sample_group_stratified(
    rewards: Sequence[float],
    n_samples: int,
    seed: int | None = None,
) -> SampleResult:
    """Stratified sampling to ensure diversity in selected samples.

    Samples roughly equal numbers of high-reward and low-reward responses
    to create meaningful preference pairs.

    Args:
        rewards: Sequence of reward values.
        n_samples: Number of samples to draw.
        seed: Optional random seed.

    Returns:
        SampleResult with selected indices.
    """
    if seed is not None:
        random.seed(seed)

    n = len(rewards)
    if n_samples >= n:
        return SampleResult(
            indices=list(range(n)),
            n_correct=sum(r > 1.0 for r in rewards),
            metadata={"kept": "all"},
        )

    # Split into high and low reward halves
    median_reward = sorted(rewards)[n // 2]
    high_indices = [i for i, r in enumerate(rewards) if r >= median_reward]
    low_indices = [i for i, r in enumerate(rewards) if r < median_reward]

    # Sample from each half
    n_high = min(n_samples // 2 + (n_samples % 2), len(high_indices))
    n_low = n_samples - n_high

    selected = []
    selected.extend(random.sample(high_indices, min(n_high, len(high_indices))))
    selected.extend(random.sample(low_indices, min(n_low, len(low_indices))))

    random.shuffle(selected)

    return SampleResult(
        indices=selected,
        n_correct=sum(r > 1.0 for r in rewards),
        metadata={
            "stratified": True,
            "n_high": len([i for i in selected if rewards[i] >= median_reward]),
            "n_low": len([i for i in selected if rewards[i] < median_reward]),
        },
    )


def filter_groups(
    all_rewards: Sequence[Sequence[float]],
    config: RolloutSamplingConfig,
    seed: int | None = None,
) -> list[SampleResult]:
    """Filter multiple groups of rewards.

    Args:
        all_rewards: Sequence of reward groups (each is a sequence of rewards).
        config: Sampling configuration.
        seed: Optional random seed.

    Returns:
        List of SampleResult, one per group.
    """
    if seed is not None:
        random.seed(seed)

    results = []
    for _i, rewards in enumerate(all_rewards):
        result = sample_group(rewards, config)
        results.append(result)

    return results


def create_preference_pairs(
    rewards: Sequence[float],
    indices: Sequence[int],
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """Create preference (chosen, rejected) pairs from rewards.

    Pairs are created such that the higher-reward response is "chosen" and
    the lower-reward response is "rejected". Ties are broken randomly.

    Args:
        rewards: Full sequence of reward values.
        indices: Indices of samples to create pairs from.
        seed: Optional random seed.

    Returns:
        List of (chosen_index, rejected_index) tuples.
    """
    if seed is not None:
        random.seed(seed)

    if len(indices) < 2:
        return []

    # Sort by reward (descending)
    sorted_indices = sorted(indices, key=lambda i: rewards[i], reverse=True)

    # Create pairs: best with worst, second-best with second-worst, etc.
    pairs: list[tuple[int, int]] = []
    left, right = 0, len(sorted_indices) - 1
    while left < right:
        pairs.append((sorted_indices[left], sorted_indices[right]))
        left += 1
        right -= 1

    return pairs


@dataclass
class RolloutSampler:
    """Main rollout sampler combining group filtering and pair creation."""

    config: RolloutSamplingConfig
    """Sampling configuration."""

    temperature_scheduler: TemperatureScheduler | None = None
    """Optional temperature scheduler."""

    def sample_rollouts(
        self,
        groups: Sequence[Sequence[float]],
        seed: int | None = None,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """Sample rollouts from reward groups.

        Returns list of (group_idx, preference_pairs) for groups that passed filtering.

        Args:
            groups: Sequence of reward groups (each is a sequence of rewards).
            seed: Optional random seed.

        Returns:
            List of (group_idx, preference_pairs) for kept groups.
        """
        group_results = filter_groups(groups, self.config, seed)

        results: list[tuple[int, list[tuple[int, int]]]] = []
        for group_idx, result in enumerate(group_results):
            if not result.indices:
                continue  # Dropped group

            rewards = groups[group_idx]
            pairs = create_preference_pairs(rewards, result.indices)
            if pairs:
                results.append((group_idx, pairs))

        return results

    def get_temperature(self) -> float:
        """Get current temperature from scheduler.

        Returns:
            Temperature for this step, or config.base.temperature if no scheduler.
        """
        if self.temperature_scheduler is None:
            return self.config.base.temperature
        return self.temperature_scheduler.step()
