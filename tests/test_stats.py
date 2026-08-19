"""Tests for :mod:`medrl.analysis.stats`.

Statistical assertions are seeded and use generous margins: the point is to pin the
*mechanisms* (CI shrinks with data, pairing cancels item difficulty, MDE demotion is
mechanical, gate rules fire exactly), not to re-derive sampling theory tightly enough to
flake on a different BLAS.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from medrl.analysis.stats import (
    BenchmarkScore,
    BootstrapCI,
    Comparison,
    bootstrap_ci,
    paired_bootstrap,
    promotion_gate,
    score_benchmark,
)


def _score(name: str, mean: float, *, mde: float = 0.004) -> BenchmarkScore:
    """Hand-built score for gate tests that exercise rules, not resampling."""
    return BenchmarkScore(
        name=name,
        mean=mean,
        ci=BootstrapCI(mean - 0.01, mean + 0.01, mean, 2_000, 0.95),
        n_items=400,
        n_repeats=8,
        mde=mde,
    )


def _gate_dicts(
    chal_means: dict[str, float], base_means: dict[str, float]
) -> tuple[dict[str, BenchmarkScore], dict[str, BenchmarkScore]]:
    return (
        {n: _score(n, m) for n, m in chal_means.items()},
        {n: _score(n, m) for n, m in base_means.items()},
    )


# --------------------------------------------------------------------------------------
# bootstrap_ci
# --------------------------------------------------------------------------------------


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.5, 0.15, size=200)
    ci = bootstrap_ci(values, n_resamples=4_000, level=0.9, seed=0)
    assert ci.low < ci.mean < ci.high
    assert ci.mean == pytest.approx(float(values.mean()))
    assert ci.n_resamples == 4_000
    assert ci.level == 0.9


def test_bootstrap_ci_constant_sample_is_degenerate() -> None:
    ci = bootstrap_ci(np.full(30, 0.7), n_resamples=1_000, seed=0)
    assert ci.low == ci.high == ci.mean == pytest.approx(0.7)


def test_bootstrap_ci_width_shrinks_with_sample_size() -> None:
    values = np.random.default_rng(123).normal(0.5, 0.15, size=600)
    small = bootstrap_ci(values[:60], n_resamples=4_000, seed=0)
    large = bootstrap_ci(values, n_resamples=4_000, seed=0)
    # 10x the data should buy roughly sqrt(10)~3x a narrower interval; demand half that.
    assert (large.high - large.low) * 1.5 < (small.high - small.low)


def test_bootstrap_ci_higher_level_is_wider() -> None:
    values = np.random.default_rng(4).normal(0.4, 0.2, size=150)
    lax = bootstrap_ci(values, n_resamples=4_000, level=0.80, seed=0)
    strict = bootstrap_ci(values, n_resamples=4_000, level=0.99, seed=0)
    assert strict.low < lax.low
    assert strict.high > lax.high


def test_bootstrap_ci_seed_determinism() -> None:
    values = np.random.default_rng(9).normal(0.5, 0.1, size=100)
    first = bootstrap_ci(values, n_resamples=2_000, seed=7)
    second = bootstrap_ci(values, n_resamples=2_000, seed=7)
    other = bootstrap_ci(values, n_resamples=2_000, seed=8)
    assert first == second
    assert first != other


def test_bootstrap_ci_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([], n_resamples=10)
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci([0.1, 0.2], level=1.5)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci([0.1, 0.2], n_resamples=0)
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ci([0.1, np.nan, 0.3])


# --------------------------------------------------------------------------------------
# score_benchmark
# --------------------------------------------------------------------------------------


def test_score_benchmark_matches_item_level_statistics() -> None:
    scores = np.random.default_rng(21).normal(0.6, 0.2, size=(120, 4))
    result = score_benchmark("mmlu", scores, n_resamples=2_000, seed=3)
    item_means = scores.mean(axis=1)
    se = float(item_means.std(ddof=1)) / np.sqrt(120)
    assert result.name == "mmlu"
    assert result.mean == pytest.approx(float(item_means.mean()))
    assert result.mde == pytest.approx(2.0 * se)
    assert result.n_items == 120
    assert result.n_repeats == 4
    assert result.points == pytest.approx(result.mean * 100.0)
    assert result.mde_points == pytest.approx(result.mde * 100.0)
    assert result.ci.low <= result.mean <= result.ci.high
    assert result.ci.mean == pytest.approx(result.mean)


def test_score_benchmark_mde_shrinks_with_items() -> None:
    scores = np.random.default_rng(31).normal(0.55, 0.25, size=(400, 2))
    few = score_benchmark("b", scores[:50], n_resamples=1_000, seed=0)
    many = score_benchmark("b", scores, n_resamples=1_000, seed=0)
    # 8x items -> SE (hence MDE) shrinks ~sqrt(8); demand a factor of 2.
    assert many.mde * 2.0 < few.mde


def test_score_benchmark_seed_determinism() -> None:
    scores = np.random.default_rng(41).normal(0.6, 0.1, size=(60, 3))
    assert score_benchmark("b", scores, n_resamples=1_000, seed=5) == score_benchmark(
        "b", scores, n_resamples=1_000, seed=5
    )


def test_score_benchmark_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        score_benchmark("b", np.zeros(10))
    with pytest.raises(ValueError, match=">= 2 items"):
        score_benchmark("b", np.zeros((1, 4)))
    with pytest.raises(ValueError, match="non-finite"):
        score_benchmark("b", np.array([[0.1, np.nan], [0.2, 0.3]]))


# --------------------------------------------------------------------------------------
# paired_bootstrap
# --------------------------------------------------------------------------------------


def test_paired_identical_matrices_give_null_difference() -> None:
    a = np.random.default_rng(51).normal(0.6, 0.2, size=(80, 3))
    ci, p = paired_bootstrap(a, a, n_resamples=2_000, seed=0)
    assert ci.low <= 0.0 <= ci.high
    assert ci.mean == pytest.approx(0.0)
    # All-zero resampled differences count on both sides of zero -> degenerate p is 1.
    assert p == 1.0


def test_paired_bootstrap_detects_real_effect() -> None:
    rng = np.random.default_rng(53)
    base = rng.normal(0.60, 0.05, size=(100, 4))
    chal = base + 0.05
    ci, p = paired_bootstrap(chal, base, n_resamples=4_000, seed=0)
    assert ci.low > 0.0
    assert ci.mean == pytest.approx(0.05, abs=1e-6)
    assert p < 0.05


def test_paired_bootstrap_rejects_zero_repeats() -> None:
    # Regression: (n, 0) matrices used to return an all-NaN CI with p=0.0 -- the finite
    # check passes vacuously on an empty repeat axis.
    with pytest.raises(ValueError, match=">= 1 repeat"):
        paired_bootstrap(np.zeros((4, 0)), np.zeros((4, 0)))


def test_paired_bootstrap_cancels_item_difficulty_variance() -> None:
    # Shared item difficulty dominates; only pairing removes it. An independent-resample
    # (unpaired) comparison on the same data pays that variance twice and misses the effect.
    rng = np.random.default_rng(11)
    n_items, n_repeats = 80, 4
    difficulty = rng.normal(0.0, 0.15, size=(n_items, 1))
    a = difficulty + rng.normal(0.50, 0.02, size=(n_items, n_repeats))
    b = difficulty + rng.normal(0.52, 0.02, size=(n_items, n_repeats))

    # Convention: paired_bootstrap(x, y) returns mean(x) - mean(y); b is the "challenger"
    # here, so (b, a) yields the improvement and a resolved effect is a strictly positive CI.
    paired_ci, _ = paired_bootstrap(b, a, n_resamples=4_000, seed=0)

    item_a, item_b = a.mean(axis=1), b.mean(axis=1)
    rng_u = np.random.default_rng(0)
    ia = rng_u.integers(0, n_items, size=(4_000, n_items))
    ib = rng_u.integers(0, n_items, size=(4_000, n_items))
    unpaired = item_a[ia].mean(axis=1) - item_b[ib].mean(axis=1)
    unpaired_low, unpaired_high = np.quantile(unpaired, [0.025, 0.975])

    paired_width = paired_ci.high - paired_ci.low
    unpaired_width = float(unpaired_high - unpaired_low)
    assert paired_width < unpaired_width / 3.0
    assert paired_ci.low > 0.0  # paired: the +0.02 effect is resolved
    assert float(unpaired_low) < 0.0 < float(unpaired_high)  # unpaired: same effect drowned


def test_paired_bootstrap_seed_determinism() -> None:
    rng = np.random.default_rng(57)
    a = rng.normal(0.6, 0.1, size=(50, 2))
    b = rng.normal(0.62, 0.1, size=(50, 2))
    first = paired_bootstrap(a, b, n_resamples=1_000, seed=9)
    second = paired_bootstrap(a, b, n_resamples=1_000, seed=9)
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_paired_bootstrap_rejects_mismatched_inputs() -> None:
    a = np.zeros((10, 2))
    with pytest.raises(ValueError, match="share shape"):
        paired_bootstrap(a, np.zeros((11, 2)))
    with pytest.raises(ValueError, match="2-D"):
        paired_bootstrap(np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError, match=">= 2 items"):
        paired_bootstrap(np.zeros((1, 2)), np.zeros((1, 2)))


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


def test_comparison_significant_is_derived_from_interval() -> None:
    positive = Comparison("mmlu", 1.5, BootstrapCI(0.5, 2.5, 1.5, 1_000, 0.95), 0.03)
    negative = Comparison("math", -1.5, BootstrapCI(-2.5, -0.5, -1.5, 1_000, 0.95), 0.03)
    spanning = Comparison("safety", 1.0, BootstrapCI(-0.5, 2.5, 1.0, 1_000, 0.95), 0.4)
    assert positive.significant
    assert negative.significant
    assert not spanning.significant
    # It is a real field (shows up in reprs/dumps), just not settable.
    assert dataclasses.asdict(positive)["significant"] is True


# --------------------------------------------------------------------------------------
# promotion_gate
# --------------------------------------------------------------------------------------


def test_promotion_gate_accepts_clear_win() -> None:
    chal, base = _gate_dicts(
        {"a": 0.62, "b": 0.64, "c": 0.59, "guard": 0.91},
        {"a": 0.58, "b": 0.60, "c": 0.55, "guard": 0.90},
    )
    decision = promotion_gate(chal, base, min_wins=3, guardrails=["guard"])
    assert decision.promoted
    assert decision.wins == ["a", "b", "c"]
    assert decision.losses == []
    assert decision.guardrail_violations == []
    assert decision.demoted == []
    # Guardrails may only block; an improving guardrail is not a win.
    assert "guard" not in decision.wins
    assert any(r.startswith("verdict: promoted") for r in decision.reasons)


def test_promotion_gate_rejects_too_few_wins() -> None:
    chal, base = _gate_dicts(
        {"a": 0.62, "b": 0.64, "c": 0.581},  # c is +0.1pt: inside its 0.4pt MDE
        {"a": 0.58, "b": 0.60, "c": 0.580},
    )
    decision = promotion_gate(chal, base, min_wins=3)
    assert not decision.promoted
    assert decision.wins == ["a", "b"]
    assert any("wins 2/3 required" in r for r in decision.reasons)
    assert any("no detectable change" in r for r in decision.reasons)


def test_promotion_gate_win_must_exceed_mde() -> None:
    chal, base = _gate_dicts({"a": 0.582}, {"a": 0.580})  # +0.2pt < 0.4pt MDE
    decision = promotion_gate(chal, base, min_wins=1)
    assert not decision.promoted
    assert decision.wins == []


def test_promotion_gate_rejects_on_loss() -> None:
    chal, base = _gate_dicts(
        {"a": 0.62, "b": 0.50, "c": 0.59},
        {"a": 0.58, "b": 0.60, "c": 0.55},
    )
    decision = promotion_gate(chal, base, min_wins=2)
    assert not decision.promoted
    assert decision.wins == ["a", "c"]
    assert decision.losses == ["b"]
    assert any("loss" in r for r in decision.reasons)


def test_promotion_gate_rejects_on_guardrail_regression() -> None:
    chal, base = _gate_dicts(
        {"a": 0.62, "b": 0.64, "c": 0.59, "tox": 0.898},
        {"a": 0.58, "b": 0.60, "c": 0.55, "tox": 0.900},
    )
    decision = promotion_gate(chal, base, min_wins=3, guardrails=["tox"])
    assert not decision.promoted
    assert decision.guardrail_violations == ["tox"]
    assert decision.wins == ["a", "b", "c"]  # the wins happened; the guardrail vetoed
    assert any("guardrail violation" in r for r in decision.reasons)


def test_promotion_gate_guardrail_regression_tolerance() -> None:
    chal, base = _gate_dicts(
        {"a": 0.62, "b": 0.64, "c": 0.59, "tox": 0.895},
        {"a": 0.58, "b": 0.60, "c": 0.55, "tox": 0.900},
    )
    # -0.5pt regression: violates at the default tolerance of 0.0, allowed at 1.0pt.
    strict = promotion_gate(chal, base, min_wins=3, guardrails=["tox"])
    lax = promotion_gate(
        chal, base, min_wins=3, guardrail_max_regression_points=1.0, guardrails=["tox"]
    )
    assert not strict.promoted
    assert strict.guardrail_violations == ["tox"]
    assert lax.promoted
    assert lax.guardrail_violations == []
    # Exactly zero regression is not "more than 0.0".
    flat_chal, flat_base = _gate_dicts(
        {"a": 0.62, "tox": 0.90}, {"a": 0.58, "tox": 0.90}
    )
    equal = promotion_gate(flat_chal, flat_base, min_wins=1, guardrails=["tox"])
    assert equal.promoted
    assert equal.guardrail_violations == []


def test_promotion_gate_resolves_swings_beyond_the_noise_floor() -> None:
    # 8 items of noisy binary scoring (mde ~38pt). A +50pt swing exceeds even that
    # floor, so it is classified as a win regardless of max_effect_points: a benchmark
    # too coarse to arbitrate a 1pt effect still sees a 50pt one. (The old rule demoted
    # it -- symmetric with the loss case below, which promoted models that tanked a
    # noisy benchmark.)
    base_scores = np.repeat(np.array([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]]), 2, axis=1)
    chal_scores = np.ones((8, 2))
    base = {"noisy": score_benchmark("noisy", base_scores, n_resamples=1_000, seed=0)}
    chal = {"noisy": score_benchmark("noisy", chal_scores, n_resamples=1_000, seed=0)}
    assert base["noisy"].mde_points > 1.0
    assert chal["noisy"].points > base["noisy"].points

    decision = promotion_gate(chal, base, min_wins=1, max_effect_points=1.0)
    assert decision.promoted
    assert decision.wins == ["noisy"]
    assert decision.demoted == []
    assert any("despite mde > max effect" in r for r in decision.reasons)


def test_promotion_gate_blocks_coarse_benchmark_collapses() -> None:
    # Regression (review finding): a resolved LOSS on an over-MDE benchmark used to be
    # demoted-and-ignored, so a model that tanked it could still promote.
    base_scores = np.repeat(np.array([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]]), 2, axis=1)
    chal_scores = np.zeros((8, 2))  # challenger breaks every item: -50pt
    base = {"noisy": score_benchmark("noisy", base_scores, n_resamples=1_000, seed=0)}
    chal = {"noisy": score_benchmark("noisy", chal_scores, n_resamples=1_000, seed=0)}

    decision = promotion_gate(chal, base, min_wins=1, max_effect_points=1.0)
    assert not decision.promoted
    assert decision.losses == ["noisy"]
    assert decision.demoted == []


def test_promotion_gate_demotes_unresolved_coarse_benchmark() -> None:
    # Delta within the noise floor AND mde > max_effect: nothing to arbitrate with.
    base_scores = np.repeat(np.array([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]]), 2, axis=1)
    chal_scores = np.repeat(np.array([[0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [1.0], [0.0]]), 2, axis=1)
    base = {"noisy": score_benchmark("noisy", base_scores, n_resamples=1_000, seed=0)}
    chal = {"noisy": score_benchmark("noisy", chal_scores, n_resamples=1_000, seed=0)}
    assert abs(chal["noisy"].points - base["noisy"].points) < base["noisy"].mde_points

    decision = promotion_gate(chal, base, min_wins=1, max_effect_points=1.0)
    assert not decision.promoted
    assert decision.demoted == ["noisy"]
    assert decision.wins == [] and decision.losses == []
    assert any("demoted to reporting" in r for r in decision.reasons)


def test_promotion_gate_ignores_one_sided_benchmarks() -> None:
    chal, base = _gate_dicts(
        {"a": 0.62, "b": 0.64, "c": 0.59, "ghost": 0.90},
        {"a": 0.58, "b": 0.60, "c": 0.55},
    )
    decision = promotion_gate(chal, base, min_wins=3)
    assert decision.promoted
    assert "ghost" not in decision.wins
    assert any("present only in challenger" in r for r in decision.reasons)


def test_promotion_gate_missing_guardrail_raises() -> None:
    chal, base = _gate_dicts({"a": 0.62}, {"a": 0.58})
    with pytest.raises(ValueError, match="guardrail benchmark"):
        promotion_gate(chal, base, guardrails=["tox"])
