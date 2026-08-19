"""Eval run orchestration: phases, resume, artifacts.

The runner owns the shape of a run -- deploy policy, generate to disk, redeploy judge,
grade, score, emit manifest -- so the pattern (sequential / split / sleep-wake) is a
property of the plan rather than of hand-written launch scripts. Generation itself
delegates to the vLLM runtime and is therefore only importable on the GPU host; the
structure below is exercised by dry-runs everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medrl.core.config import EvalConfig
from medrl.core.logging import get_logger
from medrl.core.manifest import RunManifest, Stage
from medrl.eval.serving.vllm import DeploymentPlan, plan_deployment

log = get_logger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """The GPU runtime (vllm / inspect-ai) is not installed on this host."""


@dataclass(frozen=True)
class EvalResult:
    benchmark: str
    points: float
    ci_low: float
    ci_high: float
    n_items: int
    n_repeats: int
    extraction_fail_rate: float
    think_completion_rate: float


def _require_runtime() -> None:
    try:
        import inspect_ai  # noqa: F401
        import vllm  # noqa: F401
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "eval generation requires the GPU runtime; install with "
            "uv pip install -e '.[eval]' on the serving host"
        ) from exc


def run_eval(config: EvalConfig, *, resume: bool = True) -> dict[str, Any]:
    """Execute the phases in order and return per-benchmark results."""
    _require_runtime()

    plan: DeploymentPlan = plan_deployment(config)
    manifest = RunManifest.create(
        Stage.EVAL, config, inputs={"model": config.model.ref}, notes=plan.pattern.value
    )
    manifest.save()
    log.info("run %s: %s phases=%d", manifest.run_id, plan.pattern.value, len(plan.phases))

    # Phase boundaries are the resume points: completions on disk are never regenerated.
    results: dict[str, Any] = {}
    try:
        _generate(plan, config, manifest, resume=resume)
        if config.judge is not None:
            _grade(plan, config, manifest, resume=resume)
        results = _score(config, manifest)
        manifest.complete(**{k: v.points for k, v in results.items()})
    except Exception as exc:
        manifest.fail(str(exc))
        manifest.save()
        raise
    manifest.save()
    return results


def _generate(plan: DeploymentPlan, config: EvalConfig, manifest: RunManifest, *, resume: bool) -> None:
    """Deploy the policy phase and write completions to the run directory."""
    raise RuntimeUnavailableError("generation loop not wired to the vLLM runtime yet")


def _grade(plan: DeploymentPlan, config: EvalConfig, manifest: RunManifest, *, resume: bool) -> None:
    """Deploy the judge phase and grade stored completions."""
    raise RuntimeUnavailableError("grading loop not wired to the judge runtime yet")


def _score(config: EvalConfig, manifest: RunManifest) -> dict[str, EvalResult]:
    """Fold graded completions into BenchmarkScores with CIs and failure rates."""
    raise RuntimeUnavailableError("scoring loop pending the generation artifact format")
