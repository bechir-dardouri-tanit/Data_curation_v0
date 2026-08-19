"""Run manifests and the content-addressed artifact store.

Every stage (convert, generate, train, eval, ablate) writes a :class:`RunManifest`
recording exactly what went in: code revision, config hash, upstream artifact hashes, and
the environment. Two consequences follow:

* An ablation sweep can skip an arm whose fingerprint already has a completed manifest,
  so re-running a sweep after adding one arm costs one arm.
* Any reported number can be traced back to the precise inputs that produced it, which is
  the difference between an experiment log and a pile of directories.
"""

from __future__ import annotations

import os
import platform as _platform
import subprocess
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from medrl.core.hashing import hash_obj
from medrl.core.paths import artifacts_dir, repo_root, runs_dir


class Stage(StrEnum):
    CONVERT = "convert"
    DATA = "data"
    SFT = "sft"
    PREF = "pref"
    RL = "rl"
    DISTILL = "distill"
    EVAL = "eval"
    ABLATION = "ablation"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        if not sha:
            return default
        dirty = subprocess.run(
            ["git", "-C", str(repo_root()), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return f"{sha[:12]}{'-dirty' if dirty.stdout.strip() else ''}"
    except (OSError, subprocess.SubprocessError):
        return default


def _package_versions() -> dict[str, str]:
    """Versions of the packages that can change a numeric result."""
    from importlib.metadata import PackageNotFoundError, version

    watched = (
        "torch", "transformers", "vllm", "verl", "inspect-ai",
        "datasets", "numpy", "flash-attn", "liger-kernel",
    )
    found: dict[str, str] = {}
    for pkg in watched:
        try:
            found[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return found


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "cuda": torch.version.cuda,
        }
    except (ImportError, RuntimeError) as exc:
        return {"available": False, "error": str(exc)}


class Environment(BaseModel):
    """Everything about the machine that could move a number."""

    model_config = ConfigDict(frozen=True)

    python: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=_platform.platform)
    hostname: str = Field(default_factory=_platform.node)
    packages: dict[str, str] = Field(default_factory=_package_versions)
    gpu: dict[str, Any] = Field(default_factory=_gpu_info)
    # Env vars that silently change kernels or determinism.
    relevant_env: dict[str, str] = Field(
        default_factory=lambda: {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA_", "NCCL_", "VLLM_", "TORCH", "HF_", "CUBLAS_"))
        }
    )


class RunManifest(BaseModel):
    """The record of one stage execution."""

    model_config = ConfigDict(frozen=False)  # status/outputs are filled in as the run proceeds

    run_id: str
    stage: Stage
    config_hash: str
    config: dict[str, Any]
    # Fingerprints of upstream artifacts this run consumed, keyed by role.
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    code_sha: str = Field(default_factory=_git_sha)
    environment: Environment = Field(default_factory=Environment)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def create(
        cls,
        stage: Stage,
        config: Any,
        *,
        inputs: dict[str, str] | None = None,
        notes: str | None = None,
    ) -> Self:
        """Build a manifest whose ``run_id`` is derived from stage + config + inputs.

        The id is deterministic on purpose: an identical re-run resolves to the same
        directory, which is what makes the cache work.
        """
        cfg = config.model_dump(mode="json") if hasattr(config, "model_dump") else dict(config)
        cfg_hash = hash_obj(cfg)
        ins = inputs or {}
        run_id = f"{stage.value}-{hash_obj({'c': cfg_hash, 'i': ins})}"
        return cls(
            run_id=run_id,
            stage=stage,
            config_hash=cfg_hash,
            config=cfg,
            inputs=ins,
            notes=notes,
        )

    @property
    def fingerprint(self) -> str:
        """Identity of this run's *inputs* (not its outputs)."""
        return hash_obj({"stage": self.stage, "config": self.config_hash, "inputs": self.inputs})

    @property
    def directory(self) -> Path:
        return runs_dir() / self.run_id

    def complete(self, **metrics: Any) -> None:
        self.metrics.update(metrics)
        self.status = RunStatus.COMPLETED
        self.finished_at = datetime.now(UTC)

    def fail(self, error: str) -> None:
        self.error = error
        self.status = RunStatus.FAILED
        self.finished_at = datetime.now(UTC)

    @property
    def duration_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def save(self, directory: Path | None = None) -> Path:
        target = directory or self.directory
        target.mkdir(parents=True, exist_ok=True)
        path = target / "manifest.json"
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        target = Path(path)
        if target.is_dir():
            target = target / "manifest.json"
        return cls.model_validate_json(target.read_text())


class ArtifactStore:
    """Lookup of completed runs by fingerprint.

    ``find`` is what lets a sweep launcher answer "have I already computed this arm?"
    without re-deriving anything.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or runs_dir()

    def find(self, fingerprint: str) -> RunManifest | None:
        """Return the completed run with this input fingerprint, if any."""
        if not self.root.exists():
            return None
        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                manifest = RunManifest.load(manifest_path)
            except (ValueError, OSError):
                continue
            if manifest.fingerprint == fingerprint and manifest.status is RunStatus.COMPLETED:
                return manifest
        return None

    def find_for(self, stage: Stage, config: Any, inputs: dict[str, str] | None = None) -> RunManifest | None:
        probe = RunManifest.create(stage, config, inputs=inputs)
        return self.find(probe.fingerprint)

    def list_runs(self, stage: Stage | None = None) -> list[RunManifest]:
        if not self.root.exists():
            return []
        out: list[RunManifest] = []
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            try:
                manifest = RunManifest.load(manifest_path)
            except (ValueError, OSError):
                continue
            if stage is None or manifest.stage is stage:
                out.append(manifest)
        return sorted(out, key=lambda m: m.started_at, reverse=True)

    def path_for(self, name: str) -> Path:
        return artifacts_dir() / name
