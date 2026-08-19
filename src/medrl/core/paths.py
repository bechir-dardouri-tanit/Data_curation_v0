"""Filesystem layout.

Code lives on the small root volume; every byte of model/data/artifact traffic goes to
``/scratch`` (2.9 TB) because ``/`` has ~105 GB free and a single Qwen3.5-9B checkpoint is
19.3 GB. Nothing in this package should ever write a large file to a path it did not get
from here.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SCRATCH = Path("/scratch/medrl")


def scratch_root() -> Path:
    """Root for all heavy artifacts. Override with ``MEDRL_SCRATCH``."""
    return Path(os.environ.get("MEDRL_SCRATCH", str(DEFAULT_SCRATCH)))


def repo_root() -> Path:
    """Repository root (the directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(scratch_root() / "hf")))


def artifacts_dir() -> Path:
    """Content-addressed artifact store."""
    return scratch_root() / "artifacts"


def datasets_dir() -> Path:
    return scratch_root() / "datasets"


def checkpoints_dir() -> Path:
    return scratch_root() / "checkpoints"


def runs_dir() -> Path:
    """One directory per run, keyed by run id."""
    return scratch_root() / "runs"


def logs_dir() -> Path:
    return scratch_root() / "logs"


def experiments_dir() -> Path:
    """Committed experiment records (config + manifest + metrics). Small, lives in git."""
    return repo_root() / "experiments"


def ensure_layout() -> None:
    """Create the scratch layout. Idempotent."""
    for path in (
        hf_home(),
        artifacts_dir(),
        datasets_dir(),
        checkpoints_dir(),
        runs_dir(),
        logs_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
