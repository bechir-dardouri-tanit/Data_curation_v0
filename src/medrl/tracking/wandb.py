"""Experiment tracking schema.

Every run records the same config keys and emits the same metric names, so runs are
comparable across weeks and the W&B view is buildable once. Samples are logged as tables
(prompt, output, parsed answer, gold, judge rationale) specifically so grader behavior can
be audited after the fact -- an unauditable judge is indistinguishable from a random one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from medrl.core.config import EvalConfig, TrackingConfig

CONFIG_KEYS = (
    "model_ref", "model_sha", "base_model", "dataset_names", "dataset_shas",
    "prompt_hash", "framework_version", "vllm_version", "thinking_mode",
    "think_budget", "answer_budget", "temperature", "top_p", "top_k",
    "n_repeats", "seed", "judge_tier", "judge_model", "code_sha", "config_hash",
)
METRIC_KEYS = (
    "acc", "acc_ci_low", "acc_ci_high", "extraction_fail_rate",
    "think_completion_rate", "mean_tokens", "gpu_seconds",
)
SAMPLE_COLUMNS = (
    "benchmark", "item_id", "prompt", "output", "parsed_answer", "gold",
    "correct", "path_used", "judge_rationale", "repeat",
)


class TrackingSink(Protocol):
    def log_config(self, config: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_samples(self, table_name: str, rows: list[dict[str, Any]]) -> None: ...
    def save_artifact(self, name: str, path: Path, artifact_type: str) -> None: ...
    def finish(self) -> None: ...
    @property
    def run_id(self) -> str | None: ...


class NullSink:
    """Default sink: swallows everything, so code paths never branch on tracking."""

    def log_config(self, config: dict[str, Any]) -> None: pass
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: pass
    def log_samples(self, table_name: str, rows: list[dict[str, Any]]) -> None: pass
    def save_artifact(self, name: str, path: Path, artifact_type: str) -> None: pass
    def finish(self) -> None: pass
    @property
    def run_id(self) -> str | None: return None


class WandbSink:
    """W&B-backed sink; constructed only when tracking is enabled and the lib exists."""

    def __init__(self, cfg: TrackingConfig, job_type: str, run_name: str, *, resume_run_id: str | None = None) -> None:
        import wandb  # imported lazily: the CPU dev box does not install it

        self._cfg = cfg
        self._run = wandb.init(
            project=cfg.project, entity=cfg.entity, group=cfg.group,
            job_type=job_type, name=run_name, tags=list(cfg.tags),
            config=dict.fromkeys(CONFIG_KEYS), reinit=True,
            id=resume_run_id, resume="allow" if resume_run_id else None,
        )

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update({k: config.get(k) for k in CONFIG_KEYS}, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._run.log(metrics, step=step)

    def log_samples(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        import wandb

        if not rows:
            return
        columns = list(rows[0].keys())
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row.get(c) for c in columns])
        self._run.log({table_name: table})

    def save_artifact(self, name: str, path: Path, artifact_type: str) -> None:
        import wandb

        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_dir(str(path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()

    @property
    def run_id(self) -> str:
        return str(self._run.id)


def make_sink(tracking: TrackingConfig, job_type: str, run_name: str, *, resume_run_id: str | None = None) -> TrackingSink:
    if not tracking.enabled or tracking.backend == "none":
        return NullSink()
    try:
        return WandbSink(tracking, job_type, run_name, resume_run_id=resume_run_id)
    except ImportError:
        return NullSink()


def run_config_for_wandb(eval_config: EvalConfig, extra: dict[str, Any]) -> dict[str, Any]:
    """Flatten an EvalConfig into the tracking schema's key set."""
    from medrl.core.hashing import hash_text

    return {
        "model_ref": eval_config.model.ref,
        "base_model": eval_config.model.hf_id,
        "dataset_names": [b.name for b in eval_config.benchmarks],
        "thinking_mode": eval_config.thinking.mode.value,
        "think_budget": eval_config.thinking.think_budget,
        "answer_budget": eval_config.thinking.answer_budget,
        "temperature": eval_config.sampling.temperature,
        "top_p": eval_config.sampling.top_p,
        "top_k": eval_config.sampling.top_k,
        "n_repeats": eval_config.sampling.n_repeats,
        "seed": eval_config.sampling.seed,
        "judge_tier": eval_config.judge.tier.value if eval_config.judge else None,
        "judge_model": eval_config.judge.model.ref if eval_config.judge else None,
        "config_hash": eval_config.fingerprint,
        "prompt_hash": hash_text("".join(sorted(b.name for b in eval_config.benchmarks))),
        **extra,
    }
