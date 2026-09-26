"""Tests for the ablation sweep framework."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from medrl.ablation.config import (
    ABLATION_A1,
    ABLATION_A2,
    ABLATION_A3,
    ABLATION_A4,
    ABLATION_A5,
    ABLATION_A6,
    ABLATION_A7,
    AblationConfig,
    BayesianStrategy,
    DecisionRules,
    GridStrategy,
    Objective,
    ObjectiveType,
    Parameter,
    ParameterType,
    SweepConfig,
)
from medrl.ablation.sweep import (
    ArmResult,
    SweepArm,
    SweepResult,
    SweepStatus,
    SweepExecutor,
    _arm_id,
    _sweep_id,
)
from medrl.ablation.analysis import (
    analyze_promotion,
    compare_arm_to_baseline,
    compute_pareto_frontier,
)
from medrl.ablation.report import generate_markdown_report, generate_csv_export


def test_parameter_categorical() -> None:
    """Categorical parameter materializes choices correctly."""
    param = Parameter(
        name="mode",
        type=ParameterType.CATEGORICAL,
        values=["on", "off"],
    )
    assert param.choices == ["on", "off"]


def test_parameter_integer() -> None:
    """Integer parameter materializes range with step."""
    param = Parameter(
        name="budget",
        type=ParameterType.INTEGER,
        values=(0, 10),
        step=2,
    )
    assert param.choices == [0, 2, 4, 6, 8, 10]


def test_parameter_integer_default_step() -> None:
    """Integer parameter uses default step=1."""
    param = Parameter(
        name="budget",
        type=ParameterType.INTEGER,
        values=(5, 8),
    )
    assert param.choices == [5, 6, 7, 8]


def test_parameter_continuous() -> None:
    """Continuous parameter materializes min/max tuple."""
    param = Parameter(
        name="rate",
        type=ParameterType.CONTINUOUS,
        values=(0.0, 1.0),
    )
    assert param.choices == [0.0, 1.0]


def test_ablation_config_fingerprint() -> None:
    """Config fingerprint is stable for same inputs."""
    config1 = AblationConfig(
        name="test",
        baseline_id="eval-abc123",
        sweep=SweepConfig(
            parameters=(
                Parameter(name="p1", type=ParameterType.CATEGORICAL, values=["a", "b"]),
            ),
        ),
    )
    config2 = AblationConfig(
        name="test",
        baseline_id="eval-abc123",
        sweep=SweepConfig(
            parameters=(
                Parameter(name="p1", type=ParameterType.CATEGORICAL, values=["a", "b"]),
            ),
        ),
    )
    assert config1.fingerprint == config2.fingerprint

    # Different baseline gives different fingerprint
    config3 = config2.model_copy(update={"baseline_id": "eval-xyz789"})
    assert config2.fingerprint != config3.fingerprint


def test_decision_rules_validation() -> None:
    """Decision rules validate their fields."""
    rules = DecisionRules(
        min_wins=3,
        max_effect_points=1.0,
        guardrails=("bench1", "bench2"),
    )
    assert rules.min_wins == 3
    assert rules.guardrails == ("bench1", "bench2")

    # Invalid: negative wins
    with pytest.raises(ValidationError):
        DecisionRules(min_wins=-1)

    # Invalid: max effect < 0
    with pytest.raises(ValidationError):
        DecisionRules(max_effect_points=-0.5)


def test_objective_validation() -> None:
    """Objective validates weight range."""
    obj = Objective(
        metric="accuracy",
        type=ObjectiveType.MAXIMIZE,
        weight=0.8,
    )
    assert obj.weight == 0.8

    # Invalid: weight > 1
    with pytest.raises(ValidationError):
        Objective(metric="accuracy", weight=1.5)

    # Invalid: weight < 0
    with pytest.raises(ValidationError):
        Objective(metric="accuracy", weight=-0.1)


def test_grid_strategy_defaults() -> None:
    """Grid strategy has sensible defaults."""
    strategy = GridStrategy()
    assert strategy.continuous_samples == 5
    assert strategy.max_arms == 1000


def test_bayesian_strategy_defaults() -> None:
    """Bayesian strategy has sensible defaults."""
    strategy = BayesianStrategy()
    assert strategy.n_init == 10
    assert strategy.n_iter == 50
    assert strategy.acquisition == "ei"


def test_sweep_id_stable() -> None:
    """Sweep ID is deterministic from config."""
    config = AblationConfig(
        name="test_sweep",
        baseline_id="eval-abc123",
    )
    sweep_id1 = _sweep_id(config)
    sweep_id2 = _sweep_id(config)
    assert sweep_id1 == sweep_id2
    assert sweep_id1.startswith("ablation-")


def test_arm_id_stable() -> None:
    """Arm ID is deterministic from sweep and params."""
    sweep_id = "ablation-abc123"
    params = {"temperature": 0.5, "n_repeats": 8}
    arm_id1 = _arm_id(sweep_id, params)
    arm_id2 = _arm_id(sweep_id, params)
    assert arm_id1 == arm_id2

    # Different params give different ID
    params2 = {"temperature": 0.6, "n_repeats": 8}
    arm_id3 = _arm_id(sweep_id, params2)
    assert arm_id1 != arm_id3


def test_sweep_arm_fingerprint() -> None:
    """Arm fingerprint is deterministic."""
    arm = SweepArm(
        arm_id="test-arm",
        parameters={"mode": "on", "budget": 4096},
    )
    fp1 = arm.fingerprint
    fp2 = arm.fingerprint
    assert fp1 == fp2


def test_sweep_result_save_load(tmp_path) -> None:
    """Sweep result serialization round-trip."""
    result = SweepResult(
        sweep_id="test-sweep",
        config=AblationConfig(
            name="test",
            baseline_id="eval-abc",
        ),
        baseline_id="eval-abc",
        arms={
            "arm1": ArmResult(
                arm_id="arm1",
                run_id="eval-arm1",
                status=SweepStatus.COMPLETED,
                metrics={"accuracy": 85.0},
                benchmark_results={
                    "bench1": {"points": 85.0, "ci_low": 83.0, "ci_high": 87.0, "mde_points": 1.5}
                },
            )
        },
        started_at=datetime.now(UTC),
    )

    path = result.save(tmp_path / "results.json")
    assert path.exists()

    loaded = SweepResult.load(path)
    assert loaded.sweep_id == result.sweep_id
    assert len(loaded.arms) == len(result.arms)
    assert loaded.arms["arm1"].status == SweepStatus.COMPLETED


def test_ablation_presets_have_configs() -> None:
    """All A1-A7 presets are callable and return configs."""
    baseline = "eval-baseline-123"

    configs = [
        ABLATION_A1(baseline),
        ABLATION_A2(baseline),
        ABLATION_A3(baseline),
        ABLATION_A4(baseline),
        ABLATION_A5(baseline),
        ABLATION_A6(baseline),
        ABLATION_A7(baseline),
    ]

    for config in configs:
        assert isinstance(config, AblationConfig)
        assert config.baseline_id == baseline
        assert config.name.startswith("A")
        assert len(config.sweep.parameters) > 0


def test_analyze_promotion_basic() -> None:
    """Promotion analysis handles empty results."""
    result = SweepResult(
        sweep_id="test",
        config=AblationConfig(name="test", baseline_id="base"),
        baseline_id="base",
    )
    baseline = {"bench1": {"points": 80.0, "ci_low": 78.0, "ci_high": 82.0, "mde_points": 2.0}}

    decision = analyze_promotion(result, baseline, DecisionRules(min_wins=1))

    assert "promoted" in decision
    assert decision["promoted"] is False  # No arms completed


def test_compare_arm_to_baseline() -> None:
    """Comparison extracts delta and significance."""
    arm = ArmResult(
        arm_id="arm1",
        run_id="eval-arm1",
        status=SweepStatus.COMPLETED,
        benchmark_results={
            "bench1": {"points": 85.0, "ci_low": 83.0, "ci_high": 87.0},
        },
    )
    baseline = {
        "bench1": {"points": 80.0, "ci_low": 78.0, "ci_high": 82.0},
    }

    comparison = compare_arm_to_baseline(arm, baseline)

    assert comparison.arm_id == "arm1"
    assert "bench1" in comparison.delta_points
    assert comparison.delta_points["bench1"] == 5.0  # 85 - 80


def test_pareto_frontier_basic() -> None:
    """Pareto frontier handles empty results."""
    result = SweepResult(
        sweep_id="test",
        config=AblationConfig(
            name="test",
            baseline_id="base",
            sweep=SweepConfig(
                objectives=(
                    Objective(metric="accuracy", weight=0.5),
                    Objective(metric="compute_cost", type=ObjectiveType.MINIMIZE, weight=0.5),
                ),
            ),
        ),
        baseline_id="base",
    )

    frontier = compute_pareto_frontier(result, result.config.sweep.objectives)

    assert frontier.frontier == []
    assert frontier.dominated == []


def test_markdown_report_generation(tmp_path) -> None:
    """Markdown report generates without error."""
    result = SweepResult(
        sweep_id="test-sweep",
        config=AblationConfig(
            name="Test Sweep",
            baseline_id="eval-base",
            description="Test description",
            sweep=SweepConfig(
                parameters=(
                    Parameter(name="mode", type=ParameterType.CATEGORICAL, values=["on", "off"]),
                ),
            ),
        ),
        baseline_id="eval-base",
        arms={
            "arm1": ArmResult(
                arm_id="arm1",
                run_id="eval-arm1",
                status=SweepStatus.COMPLETED,
                benchmark_results={"bench1": {"points": 85.0, "ci_low": 83.0, "ci_high": 87.0}},
            )
        },
        promotion_decision={
            "promoted": True,
            "best_arm_id": "arm1",
            "wins": ["bench1"],
            "losses": [],
            "guardrail_violations": [],
            "demoted": [],
            "reasons": ["test reason"],
        },
    )
    baseline = {"bench1": {"points": 80.0, "ci_low": 78.0, "ci_high": 82.0}}

    output = tmp_path / "report.md"
    path = generate_markdown_report(result, baseline, output)

    assert path.exists()
    content = path.read_text()
    assert "Test Sweep" in content
    assert "Test description" in content
    assert "PROMOTED" in content


def test_csv_export(tmp_path) -> None:
    """CSV export generates with correct columns."""
    result = SweepResult(
        sweep_id="test-sweep",
        config=AblationConfig(
            name="Test",
            baseline_id="base",
            sweep=SweepConfig(
                objectives=(Objective(metric="accuracy"),),
            ),
        ),
        baseline_id="base",
        arms={
            "arm1": ArmResult(
                arm_id="arm1",
                run_id="eval-arm1",
                status=SweepStatus.COMPLETED,
                benchmark_results={
                    "bench1": {"points": 85.0, "ci_low": 83.0, "ci_high": 87.0},
                },
            )
        },
    )
    baseline = {"bench1": {"points": 80.0, "ci_low": 78.0, "ci_high": 82.0}}

    output = tmp_path / "results.csv"
    path = generate_csv_export(result, baseline, output)

    assert path.exists()
    content = path.read_text()
    assert "arm_id" in content
    assert "bench1_points" in content
    assert "bench1_delta" in content
