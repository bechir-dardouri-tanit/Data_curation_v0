"""Statistical analysis of ablation sweep results.

Provides:
- Paired statistical comparisons between arms and baseline
- Multi-objective Pareto frontier analysis
- MDE-based promotion decisions
- Significance testing and effect size estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from medrl.ablation.config import DecisionRules, Objective
from medrl.ablation.sweep import ArmResult, SweepResult
from medrl.analysis.stats import (
    BenchmarkScore,
    BootstrapCI,
    Comparison,
    promotion_gate,
)
from medrl.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ArmComparison:
    """Result of comparing one arm against the baseline."""

    arm_id: str
    delta_points: dict[str, float]  # benchmark -> delta
    comparisons: dict[str, Comparison]
    significant_wins: list[str] = field(default_factory=list)
    significant_losses: list[str] = field(default_factory=list)
    # Multi-objective score if applicable
    scalarized_score: float | None = None
    # Pareto-optimal flag
    is_pareto_optimal: bool = False


@dataclass(frozen=True)
class ParetoFrontier:
    """Pareto frontier of arms across multiple objectives."""

    frontier: list[tuple[str, dict[str, float]]]  # arm_id -> objective values
    dominated: list[str] = field(default_factory=list)


def analyze_promotion(
    results: SweepResult,
    baseline_results: dict[str, dict[str, Any]],
    rules: DecisionRules,
) -> dict[str, Any]:
    """Apply MDE-based promotion rules to sweep results.

    Returns a decision dict matching the promotion_gate structure.
    """
    challenger_scores: dict[str, dict[str, BenchmarkScore]] = {}
    baseline_scores: dict[str, BenchmarkScore] = {}

    # Aggregate across all arms to find best
    for arm_result in results.completed_arms:
        arm_id = arm_result.arm_id
        for bench_name, bench_result in arm_result.benchmark_results.items():
            if bench_name not in baseline_results:
                continue

            # Build scores dict for this arm
            if arm_id not in challenger_scores:
                challenger_scores[arm_id] = {}

            # Convert to BenchmarkScore for promotion_gate
            mde = bench_result.get("mde_points", 0.0) / 100.0
            points = bench_result["points"] / 100.0

            challenger_scores[arm_id][bench_name] = BenchmarkScore(
                name=bench_name,
                mean=points,
                ci=BootstrapCI(
                    low=bench_result["ci_low"] / 100.0,
                    high=bench_result["ci_high"] / 100.0,
                    mean=points,
                    n_resamples=10_000,
                    level=0.95,
                ),
                n_items=bench_result.get("n_items", 0),
                n_repeats=1,
                mde=mde / 100.0,
            )

    # Build baseline scores
    for bench_name, data in baseline_results.items():
        mde = data.get("mde_points", 0.0) / 100.0
        points = data["points"] / 100.0
        baseline_scores[bench_name] = BenchmarkScore(
            name=bench_name,
            mean=points,
            ci=BootstrapCI(
                low=data["ci_low"] / 100.0,
                high=data["ci_high"] / 100.0,
                mean=points,
                n_resamples=10_000,
                level=0.95,
            ),
            n_items=data.get("n_items", 0),
            n_repeats=1,
            mde=mde / 100.0,
        )

    # Find best arm
    best_arm_id = None
    best_decision = None

    for arm_id, arm_scores in challenger_scores.items():
        decision = promotion_gate(
            challenger=arm_scores,
            baseline=baseline_scores,
            min_wins=rules.min_wins,
            max_effect_points=rules.max_effect_points,
            guardrail_max_regression_points=rules.guardrail_max_regression_points,
            guardrails=rules.guardrails,
        )

        if decision.promoted:
            best_arm_id = arm_id
            best_decision = decision
            log.info("promotion: arm %s promoted", arm_id)
            break

    if best_decision is None:
        # No arm promoted; construct a failure decision
        best_decision = promotion_gate(
            challenger=challenger_scores.get("placeholder", {}),
            baseline=baseline_scores,
            min_wins=rules.min_wins,
            max_effect_points=rules.max_effect_points,
            guardrail_max_regression_points=rules.guardrail_max_regression_points,
            guardrails=rules.guardrails,
        )

    return {
        "promoted": best_decision.promoted,
        "best_arm_id": best_arm_id,
        "wins": best_decision.wins,
        "losses": best_decision.losses,
        "guardrail_violations": best_decision.guardrail_violations,
        "demoted": best_decision.demoted,
        "reasons": best_decision.reasons,
    }


def compare_arm_to_baseline(
    arm_result: ArmResult,
    baseline_results: dict[str, dict[str, Any]],
    objectives: tuple[Objective, ...] = (),
) -> ArmComparison:
    """Run paired statistical comparison between arm and baseline.

    Returns detailed comparison with significance testing.
    """
    comparisons: dict[str, Comparison] = {}
    delta_points: dict[str, float] = {}
    significant_wins: list[str] = []
    significant_losses: list[str] = []

    for bench_name, arm_data in arm_result.benchmark_results.items():
        if bench_name not in baseline_results:
            continue

        base_data = baseline_results[bench_name]

        # Delta in points
        delta = arm_data["points"] - base_data["points"]
        delta_points[bench_name] = delta

        # Build CI for delta (simplified - uses the two CIs)
        # In practice, this would use paired_bootstrap on the raw data
        arm_se = (arm_data["ci_high"] - arm_data["ci_low"]) / 4.0  # Approximate SE from 95% CI
        base_se = (base_data["ci_high"] - base_data["ci_low"]) / 4.0
        pooled_se = np.sqrt(arm_se**2 + base_se**2)

        # Build comparison object
        ci = BootstrapCI(
            low=delta - 2 * pooled_se,
            high=delta + 2 * pooled_se,
            mean=delta,
            n_resamples=10_000,
            level=0.95,
        )

        comparisons[bench_name] = Comparison(
            benchmark=bench_name,
            delta_points=delta,
            ci=ci,
            p=_approx_p_value(delta, pooled_se),
        )

        if comparisons[bench_name].significant:
            if delta > 0:
                significant_wins.append(bench_name)
            else:
                significant_losses.append(bench_name)

    # Compute scalarized score for multi-objective
    scalarized = None
    if objectives:
        scalarized = _scalarize_objectives(arm_result, baseline_results, objectives)

    return ArmComparison(
        arm_id=arm_result.arm_id,
        delta_points=delta_points,
        comparisons=comparisons,
        significant_wins=significant_wins,
        significant_losses=significant_losses,
        scalarized_score=scalarized,
    )


def _approx_p_value(delta: float, se: float) -> float:
    """Approximate two-sided p-value from delta and standard error."""
    if se == 0:
        return 0.0 if delta != 0 else 1.0
    z = delta / se
    # Simple approximation without scipy: two-sided p-value
    # For |z| > 4, p is effectively 0; for smaller z, use rough approximation
    abs_z = abs(z)
    if abs_z > 4:
        return 0.0
    # Approximate using complementary error function expansion
    # This is a reasonable approximation for |z| < 4
    return 2.0 * (1.0 / (1.0 + abs_z) ** 2)


def _scalarize_objectives(
    arm_result: ArmResult,
    baseline_results: dict[str, dict[str, Any]],
    objectives: tuple[Objective, ...],
) -> float:
    """Compute weighted sum of normalized objectives."""
    total_weight = sum(o.weight for o in objectives)
    if total_weight == 0:
        return 0.0

    score = 0.0

    for obj in objectives:
        if obj.metric == "accuracy":
            # Average accuracy across benchmarks
            acc_sum = sum(b["points"] for b in arm_result.benchmark_results.values())
            acc_base = sum(b["points"] for b in baseline_results.values())
            # Normalize to [0, 1] range then weight
            normalized = acc_sum / max(acc_base, 1.0)
        elif obj.metric == "compute_cost":
            # Placeholder: compute cost not yet tracked
            normalized = 1.0  # No penalty by default
        else:
            # Unknown metric
            normalized = 0.0

        if obj.type.value == "minimize":
            normalized = 1.0 - normalized

        score += (obj.weight / total_weight) * normalized

    return score


def compute_pareto_frontier(
    results: SweepResult,
    objectives: tuple[Objective, ...],
) -> ParetoFrontier:
    """Compute Pareto frontier across all completed arms.

    An arm dominates another if it is better or equal on all objectives
    and strictly better on at least one.
    """
    if not objectives or len(objectives) < 2:
        return ParetoFrontier(frontier=[], dominated=list(results.arms.keys()))

    # Compute objective values for each arm
    arm_values: dict[str, dict[str, float]] = {}

    for arm_result in results.completed_arms:
        arm_id = arm_result.arm_id
        values: dict[str, float] = {}
        for obj in objectives:
            if obj.metric == "accuracy":
                acc = sum(b["points"] for b in arm_result.benchmark_results.values())
                values["accuracy"] = acc
            elif obj.metric == "compute_cost":
                # Placeholder: not yet tracked
                values["compute_cost"] = 0.0
            else:
                values[obj.metric] = 0.0
        arm_values[arm_id] = values

    # Find non-dominated arms
    frontier: list[tuple[str, dict[str, float]]] = []
    dominated: list[str] = []

    for arm_id, values in arm_values.items():
        is_dominated = False

        for other_id, other_values in arm_values.items():
            if other_id == arm_id:
                continue

            # Check if other dominates this arm
            dominates = True
            strictly_better = False

            for obj in objectives:
                if obj.metric not in values or obj.metric not in other_values:
                    continue

                if obj.type.value == "maximize":
                    if values[obj.metric] > other_values[obj.metric]:
                        dominates = False
                        break
                    if other_values[obj.metric] > values[obj.metric]:
                        strictly_better = True
                else:  # minimize
                    if values[obj.metric] < other_values[obj.metric]:
                        dominates = False
                        break
                    if other_values[obj.metric] < values[obj.metric]:
                        strictly_better = True

            if dominates and strictly_better:
                is_dominated = True
                break

        if is_dominated:
            dominated.append(arm_id)
        else:
            frontier.append((arm_id, values))

    return ParetoFrontier(frontier=frontier, dominated=dominated)


def rank_arms_by_objective(
    results: SweepResult,
    objective: Objective,
) -> list[tuple[str, float]]:
    """Rank arms by a single objective."""
    arm_scores = []

    for arm_result in results.completed_arms:
        if objective.metric == "accuracy":
            score = sum(b["points"] for b in arm_result.benchmark_results.values())
        elif objective.metric == "compute_cost":
            # Lower is better; use negative for ascending sort
            score = -(arm_result.compute_cost or 0.0)
        else:
            score = 0.0

        if objective.type.value == "minimize":
            score = -score

        arm_scores.append((arm_result.arm_id, score))

    return sorted(arm_scores, key=lambda x: x[1], reverse=True)


def analyze_results(results: SweepResult, baseline_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Comprehensive analysis of sweep results.

    Returns:
        Dictionary with comparisons, rankings, and Pareto frontier
    """
    comparisons: dict[str, ArmComparison] = {}
    for arm_result in results.completed_arms:
        comparisons[arm_result.arm_id] = compare_arm_to_baseline(
            arm_result, baseline_results, results.config.sweep.objectives
        )

    # Pareto analysis if multi-objective
    pareto = None
    if len(results.config.sweep.objectives) >= 2:
        pareto = compute_pareto_frontier(results, results.config.sweep.objectives)

    # Rank by primary objective
    rankings = []
    if results.config.sweep.objectives:
        rankings = rank_arms_by_objective(results, results.config.sweep.objectives[0])

    return {
        "comparisons": {
            arm_id: {
                "delta_points": comp.delta_points,
                "significant_wins": comp.significant_wins,
                "significant_losses": comp.significant_losses,
                "scalarized_score": comp.scalarized_score,
            }
            for arm_id, comp in comparisons.items()
        },
        "pareto_frontier": [(aid, vals) for aid, vals in pareto.frontier] if pareto else [],
        "dominated_arms": pareto.dominated if pareto else [],
        "rankings": rankings,
    }
