"""Reward function implementations for RL training.

Rewards are computed from the same verifiers used in evaluation, ensuring that the
training target matches the reported metric. This module provides:

- Verifier rewards (MCQA, numeric, rubric)
- Judge rewards (rubric-based)
- Format rewards
- Safety/deferral rewards
- Length shaping rewards

All reward functions return a float in [0, 1] for easy composition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from medrl.eval.extraction import extract_mcqa, scan_numbers
from medrl.eval.scorers.judge import CriterionVerdict
from medrl.eval.tasks.spec import VerifyStyle
from medrl.eval.verifiers import verify_letter, verify_number
from medrl.rl.config import RewardConfig


@dataclass(frozen=True)
class RewardResult:
    """Result of a reward computation.

    The total reward is a weighted sum of components; each component is reported
    for logging and diagnostics.
    """

    total: float
    """Total reward in [0, 1]."""

    verifier: float = 0.0
    """Correctness reward from verifiers."""

    format: float = 0.0
    """Format rule compliance reward."""

    length: float = 0.0
    """Length shaping reward."""

    safety: float = 0.0
    """Safety/deferral reward."""

    details: dict[str, Any] | None = None
    """Additional details for logging."""


def compute_verifier_reward(
    content: str,
    verify_style: VerifyStyle,
    verify_params: dict[str, Any],
) -> float:
    """Compute verifier reward based on correctness.

    Uses the same verification logic as evaluation grading, ensuring the reward
    matches the reported metric.

    Args:
        content: The model-generated completion.
        verify_style: The verification style (letter, number, rubric, format_rules).
        verify_params: Parameters for verification (gold values, tolerance, etc.).

    Returns:
        Reward in [0, 1] where 1.0 means correct.
    """
    if not content:
        return 0.0

    if verify_style is VerifyStyle.LETTER:
        gold = verify_params.get("gold_letter", "")
        letters = verify_params.get("letters", "ABCDE")
        result = extract_mcqa(content, letters)
        if result.value is None:
            return 0.0
        correct = verify_letter(result.value, gold, letters)
        return 1.0 if correct else 0.0

    if verify_style is VerifyStyle.NUMBER:
        gold = verify_params.get("gold_number")
        lower = verify_params.get("lower")
        upper = verify_params.get("upper")
        rtol = verify_params.get("rtol", 0.005)
        atol = verify_params.get("atol", 1e-8)

        numbers = scan_numbers(content)
        if not numbers:
            return 0.0
        pred = numbers[-1][1]  # Last number in the response

        if lower is not None and upper is not None:
            correct = lower <= pred <= upper
        elif gold is not None:
            correct = verify_number(pred, gold, rtol=rtol, atol=atol)
        else:
            correct = False
        return 1.0 if correct else 0.0

    if verify_style is VerifyStyle.RUBRIC:
        # Rubric rewards are computed separately with judge integration
        # Return 0 here to signal it needs judge computation
        return 0.0

    if verify_style is VerifyStyle.FORMAT_RULES:
        # Format rules are handled separately
        return 0.0

    return 0.0


def compute_rubric_reward(
    verdicts: Sequence[CriterionVerdict],
) -> float:
    """Compute rubric reward from judge verdicts.

    Args:
        verdicts: Judge verdicts for each criterion.

    Returns:
        Reward in [0, 1] based on fraction of met criteria.
    """
    if not verdicts:
        return 0.0

    denominator = sum(v.weight for v in verdicts if v.weight > 0)
    if denominator == 0:
        return 0.0

    numerator = sum(v.weight for v in verdicts if v.met)
    numerator = min(max(numerator, 0.0), denominator)
    return numerator / denominator


def compute_format_reward(
    content: str,
    required_rules: dict[str, str],
) -> float:
    """Compute format reward based on rule compliance.

    Args:
        content: The model-generated completion.
        required_rules: Mapping of rule name to parameters.

    Returns:
        Reward in [0, 1] based on fraction of satisfied rules.
    """
    if not content or not required_rules:
        return 1.0 if not required_rules else 0.0

    from medrl.eval.verifiers import FORMAT_RULES

    satisfied = 0
    for rule_name, params in required_rules.items():
        try:
            rule_func = FORMAT_RULES.get(rule_name)
            if rule_func(content, params):
                satisfied += 1
        except KeyError:
            # Unknown rule: fail conservatively
            continue
        except Exception:
            # Rule execution error: count as not satisfied
            continue

    return satisfied / len(required_rules) if required_rules else 1.0


def compute_length_reward(
    content: str,
    min_length: int = 50,
    max_length: int = 8192,
) -> float:
    """Compute length shaping reward.

    Penalizes responses that are too short (likely incomplete) or too long
    (likely verbose or rambling).

    Args:
        content: The model-generated completion.
        min_length: Minimum length for full reward.
        max_length: Maximum length for full reward.

    Returns:
        Reward in [0, 1] with linear penalty outside [min_length, max_length].
    """
    length = len(content)
    if length < min_length:
        # Linear penalty from 0 to min_length
        return max(0.0, length / min_length)
    if length > max_length:
        # Linear penalty from max_length upward
        excess = length - max_length
        return max(0.0, 1.0 - excess / max_length)
    return 1.0


def compute_safety_reward(
    content: str,
    deferral_phrases: tuple[str, ...] = (),
    harm_keywords: tuple[str, ...] = (),
) -> float:
    """Compute safety/deferral reward.

    Rewards appropriate deferral (phrases indicating uncertainty or referral)
    and penalizes potential harm keywords.

    Args:
        content: The model-generated completion.
        deferral_phrases: Phrases that indicate appropriate uncertainty/referral.
        harm_keywords: Keywords suggesting potentially harmful advice.

    Returns:
        Reward in [0, 1].
    """
    if not content:
        return 0.0

    content_lower = content.casefold()

    # Check for deferral (positive signal)
    has_deferral = any(phrase.casefold() in content_lower for phrase in deferral_phrases)

    # Check for harm keywords (negative signal)
    has_harm = any(keyword.casefold() in content_lower for keyword in harm_keywords)

    # If harm keywords are defined, presence reduces reward
    base_reward = (0.0 if has_harm else 0.5) if harm_keywords else 0.5

    # Deferral boosts reward
    if has_deferral:
        base_reward += 0.5

    return min(1.0, base_reward)


def compute_total_reward(
    content: str,
    verify_style: VerifyStyle,
    verify_params: dict[str, Any],
    rubric_verdicts: Sequence[CriterionVerdict] | None = None,
    format_rules: dict[str, str] | None = None,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Compute total reward from all components.

    This is the main entry point for reward computation. It combines verifier,
    format, length, and safety rewards according to the configured weights.

    Args:
        content: The model-generated completion.
        verify_style: The verification style.
        verify_params: Parameters for verification.
        rubric_verdicts: Optional judge verdicts for rubric items.
        format_rules: Optional format rule requirements.
        config: Reward configuration (uses defaults if None).

    Returns:
        RewardResult with total and component-wise rewards.
    """
    if config is None:
        config = RewardConfig()

    # Verifier reward
    if verify_style is VerifyStyle.RUBRIC and rubric_verdicts:
        verifier_reward = compute_rubric_reward(rubric_verdicts)
    else:
        verifier_reward = compute_verifier_reward(content, verify_style, verify_params)

    # Format reward
    format_reward = compute_format_reward(content, format_rules or {})

    # Length reward
    length_reward = compute_length_reward(content, config.min_length, config.max_length)

    # Safety reward
    safety_reward = compute_safety_reward(content, config.deferral_phrases, config.harm_keywords)

    # Weighted sum
    total = (
        verifier_reward * config.verifier_weight
        + format_reward * config.format_weight
        + length_reward * config.length_weight
        + safety_reward * config.safety_weight
    )

    return RewardResult(
        total=total,
        verifier=verifier_reward,
        format=format_reward,
        length=length_reward,
        safety=safety_reward,
        details={
            "verify_style": verify_style.value,
            "content_length": len(content),
        },
    )


def compute_group_rewards(
    contents: Sequence[str],
    verify_style: VerifyStyle,
    verify_params: dict[str, Any],
    rubric_verdicts: Sequence[Sequence[CriterionVerdict]] | None = None,
    format_rules: dict[str, str] | None = None,
    config: RewardConfig | None = None,
) -> list[RewardResult]:
    """Compute rewards for a group of responses to the same prompt.

    Used for group-wise algorithms like GSPO where the relative quality of
    responses in a group determines the preference signal.

    Args:
        contents: Sequence of model-generated completions.
        verify_style: The verification style.
        verify_params: Parameters for verification.
        rubric_verdicts: Optional verdicts for each completion.
        format_rules: Optional format rule requirements.
        config: Reward configuration.

    Returns:
        List of RewardResult, one per completion.
    """
    if config is None:
        config = RewardConfig()

    results: list[RewardResult] = []
    for i, content in enumerate(contents):
        verdicts = None
        if rubric_verdicts and i < len(rubric_verdicts):
            verdicts = rubric_verdicts[i]

        result = compute_total_reward(
            content=content,
            verify_style=verify_style,
            verify_params=verify_params,
            rubric_verdicts=verdicts,
            format_rules=format_rules,
            config=config,
        )
        results.append(result)

    return results
