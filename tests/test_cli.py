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


def test_convert_help_documents_contract() -> None:
    # The conversion itself moves ~19 GB and is a GPU-host operation (covered by
    # test_surgery's stubbed equivalence tests); the CLI contract worth pinning here
    # is that the command exists, documents itself, and defaults to the 9B.
    out = medrl("model", "convert", "--help")
    assert out.returncode == 0, out.stderr
    assert "--verify-logits" in out.stdout
    assert "Qwen/Qwen3.5-9B" in out.stdout


def test_probe_help_present() -> None:
    out = medrl("probe", "--help")
    assert out.returncode == 0
    assert "invariants" in out.stdout.lower()
