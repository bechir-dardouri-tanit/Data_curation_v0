"""Surgery gate tests: verification must block the save, not just comment on it.

torch/transformers are stubbed at the ``sys.modules`` level with fakes that implement
exactly the surface :func:`medrl.model.surgery.convert` touches, so the *ordering* --
verify, then save -- is pinned on a CPU box with no ML stack installed.
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from medrl.model.surgery import SurgeryVerificationError, _equivalence_ok, convert


class FakeLogits:
    """Scalar stand-in for a logits tensor: supports ``-``, ``abs()``, ``max()``."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __sub__(self, other: FakeLogits) -> FakeLogits:
        return FakeLogits(self.value - other.value)

    def abs(self) -> FakeLogits:
        return FakeLogits(abs(self.value))

    def max(self) -> float:
        return self.value

    def __float__(self) -> float:
        return self.value


class FakeModel:
    """Constant-logits model; records every ``save_pretrained`` target."""

    def __init__(self, logits_value: float, state: dict[str, Any] | None = None) -> None:
        self._logits_value = logits_value
        self._state = state or {}
        self.config = SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"])
        self.saved_to: list[str] = []

    def state_dict(self) -> dict[str, Any]:
        return self._state

    def __call__(self, **_inputs: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=FakeLogits(self._logits_value))

    def save_pretrained(self, path: str | Path) -> None:
        self.saved_to.append(str(path))


class FakeTokenizer:
    def __init__(self) -> None:
        self.saved_to: list[str] = []

    def __call__(self, _prompt: str, return_tensors: str | None = None) -> dict[str, Any]:
        return {"input_ids": [[0]]}

    def save_pretrained(self, path: str | Path) -> None:
        self.saved_to.append(str(path))


class _Factory:
    """Stand-in for an ``AutoX.from_pretrained``-style transformers class."""

    def __init__(self, product: Any) -> None:
        self._product = product

    def from_pretrained(
        self,
        _source: str,
        revision: str | None = None,
        torch_dtype: str | None = None,
        device_map: str | None = None,
    ) -> Any:
        return self._product


class TorchStack:
    """Installed stub modules plus the handles the tests assert on."""

    def __init__(self, reference: FakeModel, text_model: FakeModel, tokenizer: FakeTokenizer) -> None:
        self.reference = reference
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.mtp_saved_to: list[str] = []


def install_stack(
    monkeypatch: pytest.MonkeyPatch, ref_logit: float, text_logit: float
) -> TorchStack:
    stack = TorchStack(
        reference=FakeModel(ref_logit, state={"mtp.lm_head.weight": object()}),
        text_model=FakeModel(text_logit),
        tokenizer=FakeTokenizer(),
    )

    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = _Factory(stack.tokenizer)  # type: ignore[attr-defined]
    transformers_mod.Qwen3_5ForConditionalGeneration = _Factory(stack.reference)  # type: ignore[attr-defined]
    transformers_mod.Qwen3_5ForCausalLM = _Factory(stack.text_model)  # type: ignore[attr-defined]

    st_mod = types.ModuleType("safetensors.torch")
    st_mod.save_file = lambda _state, path: stack.mtp_saved_to.append(str(path))  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "safetensors", types.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", st_mod)
    return stack


@pytest.mark.parametrize(
    ("max_diff", "max_ref_abs", "ok"),
    [
        # Benign bf16 kernel-path noise: ~1 ULP at |logit|=30 (ULP is 2^-8 relative).
        (0.25, 30.0, True),
        # A couple of ULP still passes: 2% of 30 is 0.6.
        (0.5, 30.0, True),
        # A misrouted tensor produces diffs of the logits' own order.
        (28.0, 30.0, False),
        # Near-zero logits fall back to the absolute floor.
        (0.0005, 0.0, True),
        (0.002, 0.0, False),
    ],
)
def test_equivalence_ok_gate(max_diff: float, max_ref_abs: float, ok: bool) -> None:
    assert _equivalence_ok(max_diff, max_ref_abs) is ok


def test_benign_kernel_noise_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack = install_stack(monkeypatch, ref_logit=30.0, text_logit=30.25)
    out = tmp_path / "out"

    summary = convert("Qwen/Qwen3.5-9B", out)

    assert summary["max_logit_diff"] == pytest.approx(0.25)
    assert summary["mtp_tensors_archived"] == 1
    assert stack.text_model.saved_to == [str(out)]
    assert stack.tokenizer.saved_to == [str(out)]
    assert stack.mtp_saved_to  # sidecar archived


def test_misroute_raises_and_writes_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = install_stack(monkeypatch, ref_logit=30.0, text_logit=2.0)
    out = tmp_path / "out"

    with pytest.raises(SurgeryVerificationError, match="NOT saved"):
        convert("Qwen/Qwen3.5-9B", out)

    # The gate is the save: nothing reaches disk under the output path.
    assert stack.text_model.saved_to == []
    assert stack.tokenizer.saved_to == []
    assert not out.exists()
    # The MTP sidecar IS written: it is copied verbatim from the source, not converted,
    # so it stays useful even when the conversion itself is rejected.
    assert stack.mtp_saved_to


def test_verification_disabled_skips_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = install_stack(monkeypatch, ref_logit=30.0, text_logit=2.0)

    summary = convert("Qwen/Qwen3.5-9B", tmp_path / "out", verify_logits=False)

    assert summary["max_logit_diff"] is None
    assert stack.text_model.saved_to  # explicit opt-out is honored


def test_cli_maps_verification_failure_to_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    from medrl.cli.main import model_convert

    install_stack(monkeypatch, ref_logit=30.0, text_logit=2.0)

    with pytest.raises(typer.Exit) as exc:
        model_convert(source="Qwen/Qwen3.5-9B", out=str(tmp_path / "out"))

    assert exc.value.exit_code == 1
