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
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medrl.analysis.stats import score_benchmark
from medrl.core.config import EvalConfig, ServingPattern
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
from medrl.tracking.wandb import make_sink, run_config_for_wandb

log = get_logger(__name__)

# Optional decontamination support (import-guarded for slim hosts)
try:
    from medrl.eval import decontam
    DECONTAM_AVAILABLE = True
except ImportError:
    DECONTAM_AVAILABLE = False
    decontam = None  # type: ignore[assignment]


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
    _lock_run_dir(run_dir)
    manifest.save()
    log.info("run %s: %s phases=%d", manifest.run_id, plan.pattern.value, len(plan.phases))

    # Initialize W&B tracking with resumption support
    resume_run_id = None
    if resume and (run_dir / "wandb_run_id.txt").exists():
        resume_run_id = (run_dir / "wandb_run_id.txt").read_text().strip()
        log.info("resuming W&B run: %s", resume_run_id)

    tracking = make_sink(
        config.tracking,
        job_type="evaluation",
        run_name=manifest.run_id[:8],  # Short name for display
        resume_run_id=resume_run_id,
    )

    if tracking.run_id:
        (run_dir / "wandb_run_id.txt").write_text(tracking.run_id)
        log.info("W&B run ID: %s", tracking.run_id)

    # Log run configuration
    tracking.log_config(run_config_for_wandb(config, extra={
        "run_id": manifest.run_id,
        "serving_pattern": plan.pattern.value,
    }))

    try:
        results = _execute(config, plan, manifest, run_dir, resume=resume, tracking=tracking)
        manifest.complete(**{k: v.points for k, v in results.items()})
    except Exception as exc:
        manifest.fail(str(exc))
        manifest.save()
        tracking.finish()
        raise
    manifest.save()
    _write_experiment_record(config, plan, manifest, run_dir, results=results)

    # Save artifacts and finish tracking
    if config.tracking.enabled and config.tracking.backend == "wandb" and tracking.run_id:
        tracking.save_artifact(f"eval-run-{manifest.run_id[:8]}", run_dir, "evaluation_run")

    tracking.finish()
    return results


def _lock_run_dir(run_dir: Path) -> None:
    """Exclusive hold on the run directory; held for the process lifetime.

    run_id is a content hash, so re-invoking the same config resolves to the
    same directory, the same completions store, and the same two GPUs. A second
    concurrent copy would interleave duplicate keys and launch a second vLLM
    against GPUs the first already holds.
    """
    import fcntl

    fh = (run_dir / "run.lock").open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeError(
            f"another eval process holds {run_dir / 'run.lock'}: the same config "
            "is already running (double-booked GPUs and interleaved stores)"
        ) from exc
    fh.write(f"pid {os.getpid()}\n")
    fh.flush()


def _preflight_context_fits(
    config: EvalConfig, plan: DeploymentPlan, loaded: dict[str, list[EvalItem]]
) -> None:
    """Fail fast when an item's prompt + generation budget exceeds max_model_len.

    vLLM rejects such requests with a 400 on every attempt of every repeat --
    hours of serving later, surfacing only as error records. One CPU pass with
    the tokenizer catches it before the GPUs light up. Template overhead varies
    by mode; the slack absorbs it.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover - slim hosts
        log.warning("preflight skipped: transformers not importable")
        return
    limit = plan.policy.max_model_len
    slack = 256
    try:
        tok = AutoTokenizer.from_pretrained(config.model.hf_id)
    except Exception as exc:
        log.warning("preflight skipped: tokenizer unavailable (%s)", exc)
        return
    for bench in config.benchmarks:
        budget = config.resolved_thinking(bench).total_budget
        over = 0
        worst = ("", 0)
        for item in loaded[bench.name]:
            try:
                n = len(tok.apply_chat_template(
                    [dict(m) for m in item.messages], add_generation_prompt=True,
                ))
            except Exception:  # a template the fast tokenizer cannot render
                return
            if n + budget + slack > limit:
                over += 1
                if n > worst[1]:
                    worst = (item.item_id, n)
        if over:
            raise RuntimeError(
                f"{bench.name}: {over} items' templated prompts + the {budget}-token "
                f"generation budget exceed max_model_len={limit} (longest seen: "
                f"{worst[1]} tokens on {worst[0]!r}); lower the think budget, raise "
                "max_model_len, or drop the items"
            )


def _preflight_decontamination_check(
    config: EvalConfig,
    loaded: dict[str, list[EvalItem]],
    run_dir: Path,
) -> None:
    """Scan evaluation items for contamination with training data.

    Uses n-gram overlap analysis to detect potential data leakage between
    training and evaluation sets. Critical contamination (>50% overlap) fails
    the run by default.

    Training data paths are configured via environment variable:
    MEDRL_DECONTAM_TRAINING_PATHS (comma-separated list)

    This check runs after items are loaded but before any GPU work starts.
    """
    if not DECONTAM_AVAILABLE:
        log.info("Decontamination check skipped: module not available")
        return

    import os

    training_paths = os.environ.get("MEDRL_DECONTAM_TRAINING_PATHS", "")
    if not training_paths:
        log.info("Decontamination check skipped: no training paths configured")
        return

    paths = [p.strip() for p in training_paths.split(",") if p.strip()]
    if not paths:
        log.info("Decontamination check skipped: empty training paths")
        return

    log.info("Running decontamination pre-flight check against %d training sources", len(paths))

    try:
        from medrl.eval.decontam import (
            DecontaminationConfig,
            run_decontamination_check,
            save_decontamination_report,
            summarize_for_human,
        )

        # Configure decontamination
        decontam_config = DecontaminationConfig(
            n_gram_size=13,
            training_data_paths=paths,
            fail_on_critical=True,  # Fail on critical contamination
        )

        # Run the scan
        reports = run_decontamination_check(
            benchmarks=loaded,
            training_paths=paths,
            config=decontam_config,
        )

        # Save detailed report
        report_path = run_dir / "decontamination_report.json"
        save_decontamination_report(reports, report_path)

        # Log summary
        summary = summarize_for_human(reports)
        log.info("Decontamination scan complete:\n%s", summary)

        # Log to tracking
        for bench_name, report in reports.items():
            log.info(
                "Decontamination %s: %d/%d items contaminated (%.1f%%)",
                bench_name,
                report.items_with_overlap,
                report.total_items,
                report.contamination_rate * 100,
            )

    except Exception as exc:
        log.warning("Decontamination check failed: %s", exc)
        # Don't fail the run if decontamination errors, just log


def _execute(
    config: EvalConfig,
    plan: DeploymentPlan,
    manifest: RunManifest,
    run_dir: Path,
    *,
    resume: bool,
    tracking: Any,  # TrackingSink protocol
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
    if plan.pattern is not ServingPattern.SEQUENTIAL:
        # split_gpu/sleep_wake are planned shapes; the runner executes every
        # pattern identically today, and silently paying split_gpu's tp=1 cost
        # for none of its overlap is worse than refusing.
        raise NotImplementedError(
            f"serving={plan.pattern.value} is not implemented by the runner yet; "
            "use serving=sequential"
        )
    loaded: dict[str, list[EvalItem]] = {}
    load_audit: dict[str, dict[str, object]] = {}
    for bench in config.benchmarks:
        spec = specs[bench.name].with_overrides(limit=bench.limit) if bench.limit else specs[bench.name]
        result = load_items(spec)
        loaded[bench.name] = result.items
        load_audit[bench.name] = {**result.meta, "drops": result.drops}
    (run_dir / "load_audit.json").write_text(json.dumps(load_audit, indent=2))
    _preflight_context_fits(config, plan, loaded)
    _preflight_decontamination_check(config, loaded, run_dir)

    # ---- policy phase ----------------------------------------------------------------
    # `resume=False` is the CLI's --fresh: regenerate everything, discard the store.
    store = CompletionStore(run_dir / "completions.jsonl", fresh=not resume)
    from openai import OpenAI

    with VLMMServer(plan.policy, run_dir=run_dir) as server:
        client = OpenAI(base_url=server.base_url, api_key="EMPTY", timeout=1200.0, max_retries=0)
        try:
            for bench in config.benchmarks:
                generate_all(
                    client,
                    config.model.ref,
                    loaded[bench.name],
                    config.resolved_sampling(bench),
                    config.resolved_thinking(bench),
                    store,
                    alive=server.alive,
                )
        finally:
            client.close()

    # ---- judge phase -----------------------------------------------------------------
    judge_benchmarks = [b.name for b in config.benchmarks if specs[b.name].requires_judge]
    judge_obj = None
    if judge_benchmarks:
        assert config.judge is not None  # validated by EvalConfig
        judge_phase = next(p for p in plan.phases if p.role == "judge")
        from medrl.eval.scorers.judge import LLMJudge

        with VLMMServer(judge_phase, run_dir=run_dir) as server:
            judge_client = OpenAI(base_url=server.base_url, api_key="EMPTY", timeout=600.0, max_retries=2)
            try:
                judge_obj = LLMJudge(
                    model=config.judge.model.ref,
                    client=judge_client,
                    temperature=config.judge.temperature,
                    max_concurrency=config.judge.max_concurrency,
                    max_tokens=config.judge.max_tokens,
                # Both judge tiers are Qwen3.5: leave thinking on and the model
                # emits its reasoning into content ahead of the JSON, breaking
                # the parse. The OpenAI backend gets no vLLM-ism in its body.
                    extra_body=(
                        {"chat_template_kwargs": {"enable_thinking": False}}
                        if config.judge.backend == "vllm" else None
                    ),
                )
                outcomes = _grade_all(config, specs, loaded, store, judge=judge_obj, tracking=tracking)
            finally:
                judge_client.close()
    else:
        outcomes = _grade_all(config, specs, loaded, store, judge=None, tracking=tracking)

    # ---- score (CPU) -----------------------------------------------------------------
    # Read records for sample-level tracking
    records = CompletionStore.read_all(store.path)
    return _score(config, specs, outcomes, run_dir, loaded, records, tracking=tracking)


def _grade_all(
    config: EvalConfig,
    specs: dict[str, Any],
    loaded: dict[str, list[EvalItem]],
    store: CompletionStore,
    *,
    judge: Any | None,
    tracking: Any,  # TrackingSink protocol
) -> dict[str, list[ItemOutcome]]:
    records = CompletionStore.read_all(store.path)
    by_bench: dict[str, list[GenRecord]] = {}
    for record in records:
        by_bench.setdefault(record.benchmark, []).append(record)

    outcomes: dict[str, list[ItemOutcome]] = {}
    outcome_path = store.path.parent / "outcomes.json"
    for bench in config.benchmarks:
        jc = config.judge
        outcomes[bench.name] = grade_benchmark(
            loaded[bench.name],
            by_bench.get(bench.name, []),
            judge=judge,
            judge_n_consistency=jc.n_consistency if jc else 1,
            judge_position_swap=jc.position_swap if jc else True,
            strict_incomplete=config.resolved_thinking(bench).strict_incomplete,
            max_workers=jc.max_concurrency if jc else 32,
        )
        log.info("graded %s: %d items", bench.name, len(outcomes[bench.name]))
        # Written per benchmark, not once at the end: judging is the last
        # GPU-adjacent phase, and a crash here must not discard the work of
        # the benchmarks that already graded.
        outcome_path.write_text(
            json.dumps({k: [asdict(o) for o in v] for k, v in outcomes.items()}, indent=2)
        )
    return outcomes


def _score(
    config: EvalConfig,
    specs: dict[str, Any],
    outcomes: dict[str, list[ItemOutcome]],
    run_dir: Path,
    loaded: dict[str, list[EvalItem]],
    records: list[GenRecord],
    *,
    tracking: Any,  # TrackingSink protocol
) -> dict[str, EvalResult]:
    results: dict[str, EvalResult] = {}
    step = 0  # Step counter for W&B logging

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
        # The same gate on the judge side: a rubric benchmark whose repeats went
        # missing systematically has a broken instrument, and zero-filled scores
        # must not be publishable as a model result.
        expected = len(rows) * width
        missing_rate = missing / expected if expected else 0.0
        if not is_mcqa and missing_rate > config.max_extraction_fail_rate:
            raise RuntimeError(
                f"{bench.name}: {missing_rate:.1%} of repeats missing (judge failures) "
                f"exceeds the configured {config.max_extraction_fail_rate:.1%} -- the judge "
                "is systematically unparseable, which is a harness bug, not a model result"
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

        # Log metrics to W&B after each benchmark
        tracking.log_metrics(
            {
                f"{bench.name}/accuracy": score.points,
                f"{bench.name}/acc_ci_low": score.ci.low * 100.0,
                f"{bench.name}/acc_ci_high": score.ci.high * 100.0,
                f"{bench.name}/extraction_fail_rate": fail_rate,
                f"{bench.name}/think_completion_rate": think_rate,
                f"{bench.name}/repeats_missing": missing,
                f"{bench.name}/n_items": score.n_items,
            },
            step=step,
        )
        step += 1

        # Log sample-level data if enabled
        if config.tracking.log_samples and bench_outcomes:
            bench_records = [r for r in records if r.benchmark == bench.name]
            samples = _build_sample_rows(bench.name, bench_outcomes, loaded.get(bench.name, []), bench_records)
            tracking.log_samples(f"samples/{bench.name}", samples)

    (run_dir / "results.json").write_text(
        json.dumps({k: asdict(v) for k, v in results.items()}, indent=2)
    )
    return results


def _build_sample_rows(
    benchmark: str,
    outcomes: list[ItemOutcome],
    items: list[EvalItem],
    records: list[GenRecord],
) -> list[dict[str, Any]]:
    """Build sample-level rows for W&B Tables from graded outcomes and completions."""

    items_by_id: dict[str, EvalItem] = {item.item_id: item for item in items}
    records_by_key: dict[tuple[str, str, int], GenRecord] = {
        (r.benchmark, r.item_id, r.repeat): r for r in records
    }
    rows: list[dict[str, Any]] = []

    for outcome in outcomes:
        item = items_by_id.get(outcome.item_id)
        if item is None:
            continue

        # Get the prompt (last user message)
        prompt = ""
        for msg in reversed(item.messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        for repeat_outcome in outcome.repeats:
            record = records_by_key.get((benchmark, outcome.item_id, repeat_outcome.repeat))
            output = record.content if record else ""
            reasoning = record.reasoning if record else ""

            # Build the output string - include reasoning if present
            full_output = ""
            if reasoning:
                full_output = f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
            full_output += output or ""

            # Get gold answer
            gold = ""
            if item.verify.letters:
                gold = str(item.verify.gold_letter) if item.verify.gold_letter else ""
            elif item.verify.gold_number is not None:
                gold = str(item.verify.gold_number)

            # Get judge rationale if available (from repeat_outcome)
            judge_rationale = getattr(repeat_outcome, 'rationale', None)

            rows.append({
                "benchmark": benchmark,
                "item_id": outcome.item_id,
                "prompt": prompt,
                "output": full_output,
                "parsed_answer": None,  # Would need extraction from output
                "gold": gold,
                "correct": repeat_outcome.score,
                "path_used": repeat_outcome.extraction_path,
                "judge_rationale": judge_rationale,
                "repeat": repeat_outcome.repeat,
            })

    return rows


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
