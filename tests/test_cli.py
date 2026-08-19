"""End-to-end CLI smoke tests: every command must run on a CPU box with no ML stack."""

from __future__ import annotations

import subprocess
import sys

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()


def medrl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "medrl.cli.main", *args],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )


def test_version() -> None:
    out = medrl("version")
    assert out.returncode == 0
    assert "medrl 0." in out.stdout


def test_eval_tasks_lists_registry() -> None:
    out = medrl("eval", "tasks")
    assert out.returncode == 0
    for name in ("medqa", "mediqal", "healthbench_hard", "gpqa_diamond"):
        assert name in out.stdout


def test_eval_plan_renders_deployment() -> None:
    out = medrl("eval", "plan", "-c", "decision_grade")
    assert out.returncode == 0, out.stderr
    assert "vllm serve" in out.stdout
    assert "--language-model-only" in out.stdout
    assert "phase 1: policy" in out.stdout


def test_eval_plan_fast_preset() -> None:
    out = medrl("eval", "plan", "-c", "fast")
    assert out.returncode == 0, out.stderr
    assert "split_gpu" in out.stdout.replace("-", "_")


def test_convert_fails_cleanly_without_train_stack() -> None:
    out = medrl("model", "convert")
    # torch is absent on the CPU box: the guard must exit 2 with guidance, not traceback.
    assert out.returncode == 2, out.stderr
    assert "train" in (out.stdout + out.stderr)


def test_probe_help_present() -> None:
    out = medrl("probe", "--help")
    assert out.returncode == 0
    assert "invariants" in out.stdout.lower()
