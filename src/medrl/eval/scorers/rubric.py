"""Rubric scoring with judge-reliability layers, and HealthBench aggregation.

HealthBench scores a conversation against weighted binary criteria; the reported number is
the *macro* mean over conversations of the per-conversation weighted met-fraction. The
grader is an LLM, and graders are order-sensitive and self-inconsistent -- a weaker grader
systematically inflates HealthBench-Hard -- so before any number is believed it goes
through the reliability layers here:

- **Majority vote** over ``n_consistency`` gradings (callers raise the judge temperature
  above zero so the votes are not identical).
- **Position swap**: grade once more with the criterion list reversed, and keep a criterion
  only if both orders agree. ``flip_count`` reports how often they did not -- the cheapest
  measurable proxy for "this judge is unreliable on this rubric", and the number that
  justifies paying for the strong judge tier.

This module is imported by the RL reward, not copied: the graded target during training is
the same object that is reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from medrl.eval.scorers.judge import Criterion, CriterionVerdict, Judge, RubricScore


def score_conversation(
    messages: list[dict[str, str]],
    criteria: Sequence[Criterion],
    judge: Judge,
) -> RubricScore:
    """Grade once and fold into a score. The reliability layers wrap this."""
    verdicts = judge.grade(messages, criteria)
    return _fold(verdicts)


def _fold(verdicts: Sequence[CriterionVerdict], *, flip_count: int = 0) -> RubricScore:
    total_weight = sum(v.weight for v in verdicts)
    met_weight = sum(v.weight for v in verdicts if v.met)
    fraction = met_weight / total_weight if total_weight > 0 else 0.0
    return RubricScore(
        met_fraction=fraction,
        verdicts=tuple(verdicts),
        n_met=sum(1 for v in verdicts if v.met),
        n_total=len(verdicts),
        weighted=any(v.weight != 1.0 for v in verdicts),
        flip_count=flip_count,
    )


def grade_with_robustness(
    messages: list[dict[str, str]],
    criteria: Sequence[Criterion],
    judge: Judge,
    n_consistency: int = 1,
    position_swap: bool = True,
) -> RubricScore:
    """Grade with majority vote and position swap.

    A criterion survives the swap only if both orderings agree it is met; disagreement
    resolves to *not met* because a criterion that cannot be stably judged has no business
    paying reward, and an inflated reward is the failure mode RL is best at exploiting.
    """
    if n_consistency < 1:
        raise ValueError(f"n_consistency must be >= 1, got {n_consistency}")

    # Pass 1..n: majority vote per criterion id.
    votes: dict[str, list[bool]] = {}
    for _ in range(n_consistency):
        for verdict in judge.grade(messages, criteria):
            votes.setdefault(verdict.id, []).append(verdict.met)
    majority = {cid: sum(v) * 2 > len(v) for cid, v in votes.items()}

    flipped: set[str] = set()
    if position_swap:
        for verdict in judge.grade(messages, list(criteria)[::-1]):
            if verdict.met != majority.get(verdict.id, False):
                flipped.add(verdict.id)
                majority[verdict.id] = False  # conservative: unstable criteria do not pay

    verdicts = tuple(
        CriterionVerdict(
            id=c.id,
            met=majority.get(c.id, False),
            weight=c.weight,
            rationale="flipped under position swap" if c.id in flipped else None,
        )
        for c in criteria
    )
    return _fold(verdicts, flip_count=len(flipped))


def aggregate(scores: Iterable[RubricScore]) -> float:
    """HealthBench's reported number: macro mean of per-conversation met-fractions, 0-100."""
    values = [s.met_fraction for s in scores]
    if not values:
        return 0.0
    return 100.0 * sum(values) / len(values)


def aggregate_micro(scores: Iterable[RubricScore]) -> float:
    """Diagnostics-only micro aggregation: total met weight over total weight.

    Macro is the reported number (it weights conversations, matching how HealthBench treats
    each patient conversation as the unit); micro is kept only to see whether long
    conversations with many criteria are dragging the macro mean."""
    total = met = 0.0
    for score in scores:
        for verdict in score.verdicts:
            total += verdict.weight
            met += verdict.weight if verdict.met else 0.0
    return 100.0 * met / total if total > 0 else 0.0


def calibration_report(
    verdicts_a: Sequence[CriterionVerdict],
    verdicts_b: Sequence[CriterionVerdict],
) -> dict[str, object]:
    """Agreement between two judges on the same conversations' criteria.

    Used to compare the fast judge tier against the strong tier on a labeled subsample;
    Cohen's kappa corrects agreement for chance, which raw agreement wildly overstates on
    skewed rubrics where most criteria are met.
    """
    by_id_a = {v.id: v.met for v in verdicts_a}
    by_id_b = {v.id: v.met for v in verdicts_b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    if not shared:
        raise ValueError("no shared criterion ids between judges")

    n = len(shared)
    agree = sum(by_id_a[c] == by_id_b[c] for c in shared)
    p_yes_a = sum(by_id_a[c] for c in shared) / n
    p_yes_b = sum(by_id_b[c] for c in shared) / n
    p_observed = agree / n
    p_expected = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1.0 else 1.0

    return {
        "n_criteria": n,
        "agreement": p_observed,
        "cohens_kappa": kappa,
        "disagreement_ids": [c for c in shared if by_id_a[c] != by_id_b[c]],
        "judge_a_positive_rate": p_yes_a,
        "judge_b_positive_rate": p_yes_b,
    }
