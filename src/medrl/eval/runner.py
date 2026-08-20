"""Eval run orchestration: load -> serve policy -> generate -> grade -> score -> record.

The runner owns the shape of a run. A SEQUENTIAL run is: load every benchmark's items
(CPU), deploy the policy, generate every (item, repeat) completion to an append-only
store, tear the policy down, deploy the judge if any benchmark needs one, grade
everything, score on CPU, and write both the scratch run directory (heavy: server logs,
completions, per-item outcomes) and the committed experiment record (light: config,
metrics, plan). Every phase boundary is a resume point; completions already on disk are
never regenerated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medrl.analysis.stats import score_benchmark
from medrl.core.config import EvalConfig
from medrl.core.logging import get_logger
from medrl.core.manifest import RunManifest, Stage
from medrl.core.paths import experiments_dir
from medrl.eval.generate import CompletionStore, GenRecord, generate_all
from medrl.eval.grade import (
    ItemOutcome,
    extraction_fail_rate,
    grade_benchmark,
    think_completion_rate,
)
from medrl.eval.items import EvalItem
from medrl.eval.loaders import load_items
from medrl.eval.serving.lifecycle import VLMMServer
from medrl.eval.serving.vllm import DeploymentPlan, plan_deployment
from medrl.eval.tasks.spec import TASKS

log = get_logger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """The runtime dependencies (openai client, datasets) are not installed on this host."""


@dataclass(frozen=True)
class EvalResult:
    benchmark: str
    points: float
    ci_low: float
    ci_high: float
    n_items: int
    n_repeats: int
    # (item, repeat) cells that never produced a completion and were scored 0.
    # Zero unless the run is complete; a non-zero value means `points` is biased
    # down and the run is not publishable as-is.
    repeats_missing: int
    extraction_fail_rate: float
    think_completion_rate: float
    mde_points: float


def _require_runtime() -> None:
    try:
        import datasets  # noqa: F401
        import openai  # noqa: F401
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "eval generation requires the runtime extras; install with "
            "uv pip install -e '.[eval]' on the serving host"
        ) from exc


def run_eval(config: EvalConfig, *, resume: bool = True) -> dict[str, EvalResult]:
    """Execute the phases in order and return per-benchmark results."""
    _require_runtime()

    plan: DeploymentPlan = plan_deployment(config)
    manifest = RunManifest.create(
        Stage.EVAL, config, inputs={"model": config.model.ref}, notes=plan.pattern.value
    )
    run_dir = manifest.directory
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.save()
    log.info("run %s: %s phases=%d", manifest.run_id, plan.pattern.value, len(plan.phases))

    try:
        results = _execute(config, plan, manifest, run_dir, resume=resume)
        manifest.complete(**{k: v.points for k, v in results.items()})
    except Exception as exc:
        manifest.fail(str(exc))
        manifest.save()
        raise
    manifest.save()
    _write_experiment_record(config, plan, manifest, run_dir, results=results)
    return results


def _execute(
    config: EvalConfig,
    plan: DeploymentPlan,
    manifest: RunManifest,
    run_dir: Path,
    *,
    resume: bool,
) -> dict[str, EvalResult]:
    # ---- load items (CPU) ------------------------------------------------------------
    specs = {b.name: TASKS.get(b.name) for b in config.benchmarks}
    from medrl.eval.grade import ifeval_available

    if not ifeval_available() and any(
        specs[b.name].verify_style.value == "format_rules" for b in config.benchmarks
    ):
        # Fail before the GPUs light up: generating IFEval completions that cannot be
        # graded afterwards is the most expensive way to learn the checker is missing.
        raise RuntimeUnavailableError(
            "an ifeval benchmark is configured but no IFEval checker is installed; "
            "see medrl.eval.grade._IFEVAL_MISSING"
        )
    loaded: dict[str, list[EvalItem]] = {}
    load_audit: dict[str, dict[str, object]] = {}
    for bench in config.benchmarks:
        spec = specs[bench.name].with_overrides(limit=bench.limit) if bench.limit else specs[bench.name]
        result = load_items(spec)
        loaded[bench.name] = result.items
        load_audit[bench.name] = {**result.meta, "drops": result.drops}
    (run_dir / "load_audit.json").write_text(json.dumps(load_audit, indent=2))

    # ---- policy phase ----------------------------------------------------------------
    # `resume=False` is the CLI's --fresh: regenerate everything, discard the store.
    store = CompletionStore(run_dir / "completions.jsonl", fresh=not resume)
    from openai import OpenAI

    with VLMMServer(plan.policy, run_dir=run_dir) as server:
        client = OpenAI(base_url=server.base_url, api_key="EMPTY", timeout=1200.0, max_retries=0)
        for bench in config.benchmarks:
            generate_all(
                client,
                config.model.ref,
                loaded[bench.name],
                config.resolved_sampling(bench),
                config.resolved_thinking(bench),
                store,
            )

    # ---- judge phase -----------------------------------------------------------------
    judge_benchmarks = [b.name for b in config.benchmarks if specs[b.name].requires_judge]
    judge_obj = None
    if judge_benchmarks:
        assert config.judge is not None  # validated by EvalConfig
        judge_phase = next(p for p in plan.phases if p.role == "judge")
        from medrl.eval.scorers.judge import LLMJudge

        with VLMMServer(judge_phase, run_dir=run_dir) as server:
            judge_client = OpenAI(base_url=server.base_url, api_key="EMPTY", timeout=600.0, max_retries=2)
            judge_obj = LLMJudge(
                model=config.judge.model.ref,
                client=judge_client,
                temperature=config.judge.temperature,
                max_concurrency=config.judge.max_concurrency,
            )
            outcomes = _grade_all(config, specs, loaded, store, judge=judge_obj)
    else:
        outcomes = _grade_all(config, specs, loaded, store, judge=None)

    # ---- score (CPU) -----------------------------------------------------------------
    return _score(config, specs, outcomes, run_dir)


def _grade_all(
    config: EvalConfig,
    specs: dict[str, Any],
    loaded: dict[str, list[EvalItem]],
    store: CompletionStore,
    *,
    judge: Any | None,
) -> dict[str, list[ItemOutcome]]:
    records = CompletionStore.read_all(store.path)
    by_bench: dict[str, list[GenRecord]] = {}
    for record in records:
        by_bench.setdefault(record.benchmark, []).append(record)

    outcomes: dict[str, list[ItemOutcome]] = {}
    for bench in config.benchmarks:
        jc = config.judge
        outcomes[bench.name] = grade_benchmark(
            loaded[bench.name],
            by_bench.get(bench.name, []),
            judge=judge,
            judge_n_consistency=jc.n_consistency if jc else 1,
            judge_position_swap=jc.position_swap if jc else True,
            strict_incomplete=config.resolved_thinking(bench).strict_incomplete,
        )
        log.info("graded %s: %d items", bench.name, len(outcomes[bench.name]))
    (store.path.parent / "outcomes.json").write_text(
        json.dumps({k: [asdict(o) for o in v] for k, v in outcomes.items()}, indent=2)
    )
    return outcomes


def _score(
    config: EvalConfig,
    specs: dict[str, Any],
    outcomes: dict[str, list[ItemOutcome]],
    run_dir: Path,
) -> dict[str, EvalResult]:
    results: dict[str, EvalResult] = {}
    for bench in config.benchmarks:
        bench_outcomes = outcomes[bench.name]
        rows = [o.scores for o in bench_outcomes]
        if not rows:
            log.warning("benchmark %s has no graded outcomes; skipping", bench.name)
            continue
        width = config.resolved_sampling(bench).n_repeats
        matrix = np.zeros((len(rows), width))
        missing = 0
        for i, scores in enumerate(rows):
            if len(scores) != width:
                # Missing repeats score 0 -- visible, not silent, because a run that
                # lost completions mid-flight must not read as a model regression.
                # The count also lands in results.json (repeats_missing) so the
                # experiment record can never misstate a partial run as complete.
                missing += width - len(scores)
                log.warning(
                    "%s/%s: %d of %d repeats present; absent ones score 0",
                    bench.name, bench_outcomes[i].item_id, len(scores), width,
                )
            for r, value in enumerate(scores[:width]):
                matrix[i, r] = value
        score = score_benchmark(bench.name, matrix)
        fail_rate = extraction_fail_rate(bench_outcomes)
        think_rate = think_completion_rate(bench_outcomes)
        is_mcqa = specs[bench.name].verify_style.value == "letter"
        if is_mcqa and fail_rate > config.max_extraction_fail_rate:
            raise RuntimeError(
                f"{bench.name}: extraction fail rate {fail_rate:.1%} exceeds the configured "
                f"{config.max_extraction_fail_rate:.1%} -- this is a harness bug, not a model "
                "result; inspect outcomes.json before publishing anything from this run"
            )
        results[bench.name] = EvalResult(
            benchmark=bench.name,
            points=score.points,
            ci_low=score.ci.low * 100.0,
            ci_high=score.ci.high * 100.0,
            n_items=score.n_items,
            n_repeats=score.n_repeats,
            repeats_missing=missing,
            extraction_fail_rate=round(fail_rate, 4),
            think_completion_rate=round(think_rate, 4),
            mde_points=score.mde_points,
        )
    (run_dir / "results.json").write_text(
        json.dumps({k: asdict(v) for k, v in results.items()}, indent=2)
    )
    return results


def _write_experiment_record(
    config: EvalConfig,
    plan: DeploymentPlan,
    manifest: RunManifest,
    run_dir: Path,
    *,
    results: dict[str, EvalResult] | None,
) -> None:
    """The light, committed record: someone reading the repo can see what ran and why."""
    record_dir = experiments_dir() / "eval" / manifest.run_id
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "plan.txt").write_text(plan.render())
    (record_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    if results is not None:
        (record_dir / "results.json").write_text(
            json.dumps({k: asdict(v) for k, v in results.items()}, indent=2)
        )
