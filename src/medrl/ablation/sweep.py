"""Sweep orchestration and execution.

Handles:
- Generation of sweep arms from parameter spaces
- Execution of arms as eval runs with caching/resumption
- Progress tracking via manifests
- Integration with eval harness
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from medrl.ablation.config import (
    AblationConfig,
    BayesianStrategy,
    GridStrategy,
)
from medrl.core.config import EvalConfig
from medrl.core.hashing import hash_obj
from medrl.core.logging import get_logger
from medrl.core.manifest import RunManifest, Stage
from medrl.core.paths import experiments_dir, runs_dir
from medrl.eval.config_loader import load_eval_config
from medrl.eval.runner import run_eval

log = get_logger(__name__)


class SweepStatus(StrEnum):
    """State of a sweep arm."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Cached from previous run


@dataclass(frozen=True)
class SweepArm:
    """One configuration to execute in the sweep."""

    arm_id: str
    parameters: dict[str, Any]
    status: SweepStatus = SweepStatus.PENDING
    run_id: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def fingerprint(self) -> str:
        """Content hash of this arm's parameter assignment."""
        return hash_obj(self.parameters)


@dataclass(frozen=True)
class ArmResult:
    """Results from executing one sweep arm."""

    arm_id: str
    run_id: str
    status: SweepStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    benchmark_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    compute_cost: float | None = None
    error: str | None = None


@dataclass
class SweepResult:
    """Aggregated results from the entire sweep."""

    sweep_id: str
    config: AblationConfig
    baseline_id: str
    arms: dict[str, ArmResult] = field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    promotion_decision: dict[str, Any] | None = None

    @property
    def completed_arms(self) -> list[ArmResult]:
        return [r for r in self.arms.values() if r.status == SweepStatus.COMPLETED]

    @property
    def failed_arms(self) -> list[ArmResult]:
        return [r for r in self.arms.values() if r.status == SweepStatus.FAILED]

    def save(self, path: Path | None = None) -> Path:
        """Save sweep results to disk."""
        target = path or experiments_dir() / "ablation" / self.sweep_id
        target.mkdir(parents=True, exist_ok=True)
        results_file = target / "results.json"

        output = {
            "sweep_id": self.sweep_id,
            "config": self.config.model_dump(mode="json"),
            "baseline_id": self.baseline_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "arms": {
                arm_id: {
                    "run_id": result.run_id,
                    "status": result.status.value,
                    "metrics": result.metrics,
                    "benchmark_results": result.benchmark_results,
                    "compute_cost": result.compute_cost,
                    "error": result.error,
                }
                for arm_id, result in self.arms.items()
            },
            "promotion_decision": self.promotion_decision,
        }
        results_file.write_text(json.dumps(output, indent=2))
        return results_file

    @classmethod
    def load(cls, path: Path) -> SweepResult:
        """Load sweep results from disk."""
        data = json.loads(path.read_text())
        config = AblationConfig.model_validate(data["config"])
        result = cls(
            sweep_id=data["sweep_id"],
            config=config,
            baseline_id=data["baseline_id"],
        )
        if data.get("started_at"):
            result.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("finished_at"):
            result.finished_at = datetime.fromisoformat(data["finished_at"])

        for arm_id, arm_data in data["arms"].items():
            result.arms[arm_id] = ArmResult(
                arm_id=arm_id,
                run_id=arm_data["run_id"],
                status=SweepStatus(arm_data["status"]),
                metrics=arm_data.get("metrics", {}),
                benchmark_results=arm_data.get("benchmark_results", {}),
                compute_cost=arm_data.get("compute_cost"),
                error=arm_data.get("error"),
            )

        if data.get("promotion_decision"):
            result.promotion_decision = data["promotion_decision"]

        return result


class SweepExecutor:
    """Orchestrates the execution of an ablation sweep."""

    def __init__(self, config: AblationConfig, baseline_config: EvalConfig):
        self.config = config
        self.baseline_config = baseline_config
        self.sweep_id = _sweep_id(config)
        self.results = SweepResult(
            sweep_id=self.sweep_id,
            config=config,
            baseline_id=config.baseline_id,
        )

    def generate_arms(self) -> list[SweepArm]:
        """Generate all arms to execute based on strategy."""
        strategy = self.config.sweep.strategy

        if isinstance(strategy, GridStrategy):
            return self._grid_arms(strategy)
        elif isinstance(strategy, BayesianStrategy):
            return self._bayesian_arms(strategy)
        else:
            raise ValueError(f"Unknown strategy type: {type(strategy)}")

    def _grid_arms(self, strategy: GridStrategy) -> list[SweepArm]:
        """Generate arms via exhaustive grid search."""
        param_values = []
        param_names = []

        for param in self.config.sweep.parameters:
            param_names.append(param.name)
            if param.type.value == "continuous" and isinstance(param.values, tuple):
                min_val, max_val = param.values
                param_values.append(
                    np.linspace(min_val, max_val, strategy.continuous_samples).tolist()
                )
            else:
                param_values.append(param.choices)

        # Generate cartesian product
        combinations = list(itertools.product(*param_values))

        # Limit if needed
        if len(combinations) > strategy.max_arms:
            log.warning(
                "Grid generated %d arms, limiting to %d",
                len(combinations),
                strategy.max_arms,
            )
            # Sample uniformly
            rng = np.random.default_rng(self.config.sweep.seed)
            idx = rng.choice(len(combinations), strategy.max_arms, replace=False)
            combinations = [combinations[i] for i in sorted(idx)]

        arms = []
        for combo in combinations:
            params = dict(zip(param_names, combo, strict=False))
            arm = SweepArm(
                arm_id=_arm_id(self.sweep_id, params),
                parameters=params,
            )
            arms.append(arm)

        log.info("grid strategy generated %d arms", len(arms))
        return arms

    def _bayesian_arms(self, strategy: BayesianStrategy) -> list[SweepArm]:
        """Generate arms via Bayesian optimization."""
        # Initial random samples
        rng = np.random.default_rng(self.config.sweep.seed)
        arms = []

        # Generate initial random arms
        for _ in range(strategy.n_init):
            params = self._sample_random_params(rng)
            arms.append(
                SweepArm(
                    arm_id=_arm_id(self.sweep_id, params),
                    parameters=params,
                )
            )

        # Note: Full Bayesian optimization requires incremental acquisition
        # and model fitting. This is a placeholder for the initial set.
        # Real implementation would:
        # 1. Execute initial arms
        # 2. Fit GP model to results
        # 3. Generate acquisition candidates
        # 4. Repeat for n_iter

        log.info("bayesian strategy generated %d initial arms (n_init=%d)", len(arms), strategy.n_init)
        return arms

    def _sample_random_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample a random parameter assignment."""
        params: dict[str, Any] = {}
        for param in self.config.sweep.parameters:
            if param.type.value == "categorical":
                choices = param.choices
                params[param.name] = rng.choice(choices).item()
            elif param.type.value == "integer":
                if isinstance(param.values, tuple):
                    min_val, max_val = int(param.values[0]), int(param.values[1])
                    params[param.name] = int(rng.integers(min_val, max_val + 1))
                else:
                    params[param.name] = rng.choice(param.choices).item()
            else:  # continuous
                min_val, max_val = param.values  # type: ignore
                params[param.name] = float(rng.uniform(min_val, max_val))
        return params

    def run(self, resume: bool = True) -> SweepResult:
        """Execute the sweep, resuming if possible."""
        self.results.started_at = datetime.now(UTC)

        # Load previous results if resuming
        if resume:
            prev_path = experiments_dir() / "ablation" / self.sweep_id / "results.json"
            if prev_path.exists():
                log.info("resuming from previous results at %s", prev_path)
                self.results = SweepResult.load(prev_path)

        arms = self.generate_arms()
        manifest_dir = runs_dir() / self.sweep_id
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Execute arms that aren't already completed
        pending = [a for a in arms if a.arm_id not in self.results.arms or
                   self.results.arms[a.arm_id].status != SweepStatus.COMPLETED]

        if resume and len(pending) < len(arms):
            log.info("sweep: %d/%d arms already completed", len(arms) - len(pending), len(arms))

        for i, arm in enumerate(pending):
            log.info("sweep: executing arm %d/%d: %s", i + 1, len(pending), arm.arm_id)

            # Build eval config for this arm
            arm_config = self._apply_params(self.baseline_config, arm.parameters)

            try:
                # Execute eval run
                eval_results = run_eval(arm_config, resume=resume)

                # Extract results
                arm_result = ArmResult(
                    arm_id=arm.arm_id,
                    run_id=self._run_id_for_arm(arm),
                    status=SweepStatus.COMPLETED,
                    metrics=self._extract_metrics(eval_results),
                    benchmark_results=self._extract_benchmark_results(eval_results),
                )

                self.results.arms[arm.arm_id] = arm_result
                log.info("sweep: arm %s completed", arm.arm_id)

            except Exception as exc:
                log.error("sweep: arm %s failed: %s", arm.arm_id, exc)
                self.results.arms[arm.arm_id] = ArmResult(
                    arm_id=arm.arm_id,
                    run_id=self._run_id_for_arm(arm),
                    status=SweepStatus.FAILED,
                    error=str(exc),
                )

            # Save progress after each arm
            self.results.save()

        self.results.finished_at = datetime.now(UTC)

        # Load baseline results for comparison
        baseline_results = self._load_baseline_results()

        # Run promotion analysis
        if baseline_results:
            from medrl.ablation.analysis import analyze_promotion
            self.results.promotion_decision = analyze_promotion(
                self.results, baseline_results, self.config.sweep.decision_rules
            )

        self.results.save()
        return self.results

    def _apply_params(self, base: EvalConfig, params: dict[str, Any]) -> EvalConfig:
        """Apply parameter overrides to baseline config."""
        # Parse dot-notation parameter names
        overrides: dict[str, Any] = {}

        for key, value in params.items():
            parts = key.split(".")
            target = overrides
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

        # Convert nested dict to appropriate config objects
        config_dict = base.model_dump(mode="json")

        for section, values in overrides.items():
            if section in config_dict:
                if isinstance(config_dict[section], dict):
                    config_dict[section].update(values)
                else:
                    config_dict[section] = values
            else:
                config_dict[section] = values

        return EvalConfig.model_validate(config_dict)

    def _run_id_for_arm(self, arm: SweepArm) -> str:
        """Generate run_id for an arm."""
        from medrl.core.manifest import RunManifest
        probe = RunManifest.create(
            Stage.EVAL,
            self._apply_params(self.baseline_config, arm.parameters),
            inputs={"baseline": self.config.baseline_id},
        )
        return probe.run_id

    def _extract_metrics(self, eval_results: dict[str, Any]) -> dict[str, Any]:
        """Extract summary metrics from eval results."""
        metrics = {}
        for bench_name, result in eval_results.items():
            metrics[f"{bench_name}_points"] = result.points
            metrics[f"{bench_name}_ci_low"] = result.ci_low
            metrics[f"{bench_name}_ci_high"] = result.ci_high
        return metrics

    def _extract_benchmark_results(self, eval_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Extract detailed benchmark results."""
        return {
            name: {
                "points": r.points,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "n_items": r.n_items,
                "mde_points": r.mde_points,
            }
            for name, r in eval_results.items()
        }

    def _load_baseline_results(self) -> dict[str, dict[str, Any]]:
        """Load baseline eval results for comparison."""
        baseline_path = experiments_dir() / "eval" / self.config.baseline_id / "results.json"
        if not baseline_path.exists():
            log.warning("baseline results not found at %s", baseline_path)
            return {}

        data = json.loads(baseline_path.read_text())
        return {
            name: {
                "points": r["points"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "mde_points": r["mde_points"],
            }
            for name, r in data.items()
        }


def _sweep_id(config: AblationConfig) -> str:
    """Generate stable sweep ID from config."""
    return f"ablation-{config.fingerprint[:12]}"


def _arm_id(sweep_id: str, params: dict[str, Any]) -> str:
    """Generate stable arm ID from parameters."""
    param_hash = hash_obj(params)[:8]
    return f"{sweep_id}-{param_hash}"


def run_sweep(
    config: AblationConfig,
    baseline_config_path: str | Path | None = None,
    resume: bool = True,
) -> SweepResult:
    """Run an ablation sweep.

    Args:
        config: Ablation sweep configuration
        baseline_config_path: Path to baseline eval config (if not found in experiments)
        resume: Whether to resume from previous runs

    Returns:
        SweepResult with all arm results
    """
    # Load baseline config
    if baseline_config_path:
        baseline_config = load_eval_config(baseline_config_path)
    else:
        # Try to load from experiments
        baseline_path = experiments_dir() / "eval" / config.baseline_id / "manifest.json"
        if baseline_path.exists():
            manifest = RunManifest.load(baseline_path)
            baseline_config = EvalConfig.model_validate(manifest.config)
        else:
            raise ValueError(
                f"Cannot find baseline config at {baseline_path}; "
                "provide baseline_config_path"
            )

    executor = SweepExecutor(config, baseline_config)
    return executor.run(resume=resume)
