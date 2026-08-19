"""Statistics for decision-grade evaluation.

A benchmark score measured on a finite item set is an estimate, not a fact. This module
exists to make two failure modes mechanical instead of judgement calls:

1. Reading noise as signal. Every number we surface carries an interval, and every
   comparison a pairing, so "challenger beat baseline by 0.4pt" is only ever reported
   together with the uncertainty that decides whether it means anything.
2. Trusting a benchmark that cannot resolve the effect we care about. The minimum
   detectable effect (MDE) is the smallest true difference a benchmark's item sample can
   reliably surface. If it exceeds the effect size the promotion pipeline acts on, the
   benchmark is demoted to reporting-only by :func:`promotion_gate` -- a rule, not an
   opinion, so the same data always yields the same grade.

The unit of independence is the *item*. Repeats of the same item share that item's
difficulty, so scores are collapsed to per-item means first and only items are resampled;
bootstrapping raw repeats would understate variance and fake precision. For the same
reason comparisons resample item indices *jointly* for both models: item difficulty is the
dominant variance component, and pairing cancels it exactly, while an unpaired comparison
pays for it twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from medrl.core.logging import get_logger

# Resamples are computed in fixed-size chunks so peak memory is chunk*n_items rather than
# n_resamples*n_items: a default 10k-resample run over a 10k-item benchmark would otherwise
# materialise ~800 MB of indices for no statistical benefit. The chunk size is a constant,
# never a parameter, because it changes the RNG stream and therefore the results.
_CHUNK_RESAMPLES = 2_000


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap interval for a mean.

    ``low``/``high`` are quantiles of the resampled means at ``level``; ``mean`` is the
    observed point estimate, kept beside the interval so a CI can never be reported
    detached from the number it qualifies.
    """

    low: float
    high: float
    mean: float
    n_resamples: int
    level: float


@dataclass(frozen=True)
class BenchmarkScore:
    """A benchmark's score plus everything needed to judge whether it can gate a decision.

    ``mde`` is in raw score units (scores are 0-1); use :attr:`mde_points` when comparing
    against an effect size expressed in points.
    """

    name: str
    mean: float
    ci: BootstrapCI
    n_items: int
    n_repeats: int
    mde: float

    @property
    def points(self) -> float:
        """Score in points (0-100), the unit benchmark tables are read in."""
        return self.mean * 100.0

    @property
    def mde_points(self) -> float:
        """MDE in points; the threshold :func:`promotion_gate` compares against."""
        return self.mde * 100.0


@dataclass(frozen=True)
class Comparison:
    """Result of comparing two models on one benchmark.

    ``delta_points`` is challenger minus baseline in points, positive meaning the
    challenger improved. ``significant`` is always derived from the interval, never passed
    in, so it can never disagree with the CI it summarises.
    """

    benchmark: str
    delta_points: float
    ci: BootstrapCI
    p: float
    significant: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "significant", self.ci.low > 0.0 or self.ci.high < 0.0)


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of :func:`promotion_gate`; ``reasons`` audits every input to the verdict."""

    promoted: bool
    wins: list[str] = field(default_factory=list)
    losses: list[str] = field(default_factory=list)
    guardrail_violations: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _resampled_means(
    values: npt.NDArray[np.float64],
    n_resamples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Means of ``n_resamples`` bootstrap resamples of a flat sample, chunked for memory."""
    n = values.size
    out = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _CHUNK_RESAMPLES):
        stop = min(start + _CHUNK_RESAMPLES, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = values[idx].mean(axis=1)
    return out


def bootstrap_ci(
    values: npt.ArrayLike,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of a flat sample.

    The sample is treated as one observation per element; callers holding (item, repeat)
    scores must collapse to item means first -- :func:`score_benchmark` and
    :func:`paired_bootstrap` do that for you, and bootstrapping raw repeats would
    understate the interval because repeats on an item are not independent.

    Deterministic for a fixed ``seed``; identical seeds across machines and runs are what
    makes a decision auditable after the fact.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    arr = np.ravel(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not np.isfinite(arr).all():
        raise ValueError("sample contains non-finite values (NaN judge scores?)")

    means = _resampled_means(arr, n_resamples, np.random.default_rng(seed))
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return BootstrapCI(
        low=float(low),
        high=float(high),
        mean=float(arr.mean()),
        n_resamples=n_resamples,
        level=level,
    )


def score_benchmark(
    name: str,
    scores: npt.NDArray[np.float64],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> BenchmarkScore:
    """Score one benchmark from an ``(n_items, n_repeats)`` score matrix.

    Repeats are averaged per item first (they share item difficulty), then the CI
    bootstraps over *items*, the independent unit. The MDE is ``2 * SE`` with
    ``SE = std(item_means, ddof=1) / sqrt(n_items)`` -- the width of the approximate
    two-sided 95% interval, i.e. the smallest true effect this item sample can reliably
    distinguish from zero. Comparing ``mde_points`` against the effect size a decision
    acts on is what :func:`promotion_gate` does mechanically.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_items, n_repeats), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(
            f"need >= 2 items to estimate item-level variance (std with ddof=1), "
            f"got {arr.shape[0]}"
        )
    if arr.shape[1] < 1:
        raise ValueError("need >= 1 repeat per item")
    if not np.isfinite(arr).all():
        raise ValueError(f"benchmark {name!r} has non-finite scores (NaN judge scores?)")

    item_means = arr.mean(axis=1)
    ci = bootstrap_ci(item_means, n_resamples=n_resamples, level=level, seed=seed)
    se = float(item_means.std(ddof=1)) / np.sqrt(arr.shape[0])
    return BenchmarkScore(
        name=name,
        mean=float(item_means.mean()),
        ci=ci,
        n_items=int(arr.shape[0]),
        n_repeats=int(arr.shape[1]),
        mde=2.0 * se,
    )


def paired_bootstrap(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[BootstrapCI, float]:
    """Bootstrap CI and two-sided p-value for ``mean(a) - mean(b)`` on paired score matrices.

    ``a`` and ``b`` are ``(n_items, n_repeats)`` matrices over the *same* items. Per-item
    means are taken first, then item indices are resampled **jointly** -- the same indices
    for both models in every resample -- so item difficulty (usually the dominant variance
    component) cancels and the interval reflects the difference rather than the difficulty
    spread. Resampling the two models independently would pay that variance twice and can
    swamp a real effect.

    ``p_two_sided`` is twice the proportion of resampled differences on the opposite side
    of zero from the majority of the distribution, clipped to 1.0. Limitations, on purpose:
    the resampling distribution is centred on the observed difference rather than on the
    null, so this p-value is anti-conservative for small effects and small item counts; it
    degenerates to 0.0 when no resample crosses zero ("no evidence against" is not
    "proof"); and resampled differences of exactly zero (common with 0/1 scoring) are
    counted on both sides, which is what makes a null difference yield p=1.0 rather than a
    nonsensical 0.0. Use the CI for decisions; treat the p-value as a coarse screen.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired matrices must share shape, got {arr_a.shape} vs {arr_b.shape}")
    if arr_a.ndim != 2:
        raise ValueError(f"paired matrices must be 2-D (n_items, n_repeats), got {arr_a.shape}")
    if arr_a.shape[0] < 2:
        raise ValueError(f"need >= 2 items, got {arr_a.shape[0]}")
    if arr_a.shape[1] < 1:
        # A (n, 0) matrix would sail through the finite check (vacuously true) and come
        # out the far side as an all-NaN interval with p=0.0 -- "infinitely
        # significant" garbage. Refuse it up front, like score_benchmark does.
        raise ValueError(f"need >= 1 repeat per item, got shape {arr_a.shape}")
    if not (np.isfinite(arr_a).all() and np.isfinite(arr_b).all()):
        raise ValueError("paired matrices contain non-finite values (NaN judge scores?)")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    diff = arr_a.mean(axis=1) - arr_b.mean(axis=1)
    n_items = diff.size

    # Joint resampling: one index matrix drives both models, which is the whole point.
    resampled = np.empty(n_resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, _CHUNK_RESAMPLES):
        stop = min(start + _CHUNK_RESAMPLES, n_resamples)
        idx = rng.integers(0, n_items, size=(stop - start, n_items))
        resampled[start:stop] = diff[idx].mean(axis=1)

    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(resampled, [alpha, 1.0 - alpha])
    diff_ci = BootstrapCI(
        low=float(low),
        high=float(high),
        mean=float(diff.mean()),
        n_resamples=n_resamples,
        level=level,
    )

    frac_le = float(np.count_nonzero(resampled <= 0.0)) / n_resamples
    frac_ge = float(np.count_nonzero(resampled >= 0.0)) / n_resamples
    p_two_sided = min(1.0, 2.0 * min(frac_le, frac_ge))
    return diff_ci, p_two_sided


def promotion_gate(
    challenger: dict[str, BenchmarkScore],
    baseline: dict[str, BenchmarkScore],
    min_wins: int = 3,
    max_effect_points: float = 1.0,
    guardrail_max_regression_points: float = 0.0,
    *,
    guardrails: Sequence[str] = (),
) -> PromotionDecision:
    """Decide promotion from per-benchmark scores; the policy is the code, not the reviewer.

    Rules, in evaluation order:

    - Only benchmarks present in *both* dicts are compared; one-sided entries are ignored
      with a reason (there is no paired estimate to compare).
    - A swing **resolved beyond the benchmark's own MDE** is always classified, win or
      loss, even when the MDE exceeds ``max_effect_points``: a benchmark too coarse to
      arbitrate a 1pt effect still sees a 50pt collapse, and discarding that would
      promote models that tanked a noisy benchmark.
    - Otherwise a benchmark whose MDE exceeds ``max_effect_points`` lands in ``demoted``
      and is ignored -- the mechanical decision-grade-to-reporting demotion. (The noise
      floor is ``max(mde_challenger, mde_baseline)``: a baseline measured too coarsely
      cannot arbitrate a promotion either.)
    - Among the rest: a *win* is ``challenger.points - baseline.points`` strictly
      greater than that shared MDE; a *loss* is the mirror. Anything inside the MDE is
      noise and counts as neither.
    - Guardrail benchmarks (named in ``guardrails``) never generate wins -- their charter
      is to block -- and are exempt from the MDE demotion, but they violate if the
      challenger regresses by more than ``guardrail_max_regression_points`` points. The
      default of 0.0 means any regression at all violates: guardrails exist to catch
      catastrophic forgetting, and forgiving small regressions is how they get eroded.
    - Promoted iff ``len(wins) >= min_wins`` AND no losses AND no guardrail violations.

    Guardrail names missing from either dict raise :class:`ValueError`: a guardrail that
    silently cannot be checked is an open hole, not a pass.
    """
    log = get_logger(__name__)
    wins: list[str] = []
    losses: list[str] = []
    violations: list[str] = []
    demoted: list[str] = []
    reasons: list[str] = []

    for guard in guardrails:
        if guard not in challenger or guard not in baseline:
            raise ValueError(
                f"guardrail benchmark {guard!r} must be scored on both challenger and baseline"
            )
    guardrail_names = set(guardrails)

    for name in sorted(challenger.keys() | baseline.keys()):
        if name not in challenger or name not in baseline:
            side = "challenger" if name in challenger else "baseline"
            reasons.append(f"{name}: present only in {side}; ignored (no paired estimate)")
            continue
        ch, ba = challenger[name], baseline[name]
        delta_points = ch.points - ba.points

        if name in guardrail_names:
            regression = -delta_points
            if regression > guardrail_max_regression_points:
                violations.append(name)
                reasons.append(
                    f"{name}: guardrail violation, regressed {-delta_points:.2f}pt "
                    f"(allowed {guardrail_max_regression_points:.2f}pt)"
                )
            else:
                reasons.append(
                    f"{name}: guardrail held ({delta_points:+.2f}pt, "
                    f"allowed regression {guardrail_max_regression_points:.2f}pt)"
                )
            continue

        mde_points = max(ch.mde_points, ba.mde_points)
        if abs(delta_points) > mde_points:
            # Resolved beyond the benchmark's own noise floor. Checked *before* the
            # demotion arm: a benchmark too coarse to arbitrate a 1pt effect can still
            # see a 50pt collapse, and discarding that (the old rule) promoted models
            # that tanked a noisy benchmark. Symmetric on purpose -- if a swing this
            # size is trusted enough to block a promotion, it is trusted enough to
            # earn a win.
            if delta_points > 0:
                wins.append(name)
                reasons.append(
                    f"{name}: win (+{delta_points:.2f}pt > mde {mde_points:.2f}pt"
                    + (f", despite mde > max effect {max_effect_points:.2f}pt" if mde_points > max_effect_points else "")
                    + ")"
                )
            else:
                losses.append(name)
                reasons.append(
                    f"{name}: loss ({delta_points:.2f}pt beyond mde {mde_points:.2f}pt"
                    + (f", despite mde > max effect {max_effect_points:.2f}pt" if mde_points > max_effect_points else "")
                    + ")"
                )
        elif mde_points > max_effect_points:
            demoted.append(name)
            reasons.append(
                f"{name}: demoted to reporting (mde {mde_points:.2f}pt > "
                f"max effect {max_effect_points:.2f}pt); unresolved and too coarse to "
                f"arbitrate a {max_effect_points:.2f}pt effect"
            )
        else:
            reasons.append(
                f"{name}: no detectable change ({delta_points:+.2f}pt within mde "
                f"{mde_points:.2f}pt)"
            )

    promoted = len(wins) >= min_wins and not losses and not violations
    reasons.append(
        f"verdict: {'promoted' if promoted else 'not promoted'} "
        f"(wins {len(wins)}/{min_wins} required, losses {len(losses)}, "
        f"guardrail violations {len(violations)}, demoted {len(demoted)})"
    )

    log.info(
        "promotion gate: %s (%d/%d wins, %d losses, %d guardrail violations, %d demoted)",
        "promoted" if promoted else "rejected",
        len(wins),
        min_wins,
        len(losses),
        len(violations),
        len(demoted),
    )
    for reason in reasons:
        log.debug("promotion gate: %s", reason)

    return PromotionDecision(
        promoted=promoted,
        wins=wins,
        losses=losses,
        guardrail_violations=violations,
        demoted=demoted,
        reasons=reasons,
    )
