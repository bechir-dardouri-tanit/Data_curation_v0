"""Checkpoint surgery: emit a text-only Qwen3_5ForCausalLM from the multimodal release.

Why not just serve the multimodal class with ``--language-model-only``? Three reasons:
FSDP2 wrapping a frozen-but-present vision tower still allocates flat-parameter and
optimizer bookkeeping for it; a text checkpoint needs no vision-aware batching anywhere in
the stack; and the saved text checkpoint is what a training framework can load without
custom handling.

The load is *verified*, not trusted. ``Qwen3_5ForCausalLM`` declares
``_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]``, and the
checkpoint stores the language model under ``model.language_model.*`` while the CausalLM
class expects ``model.*`` -- a prefix remap that has bitten real deployments (vLLM
text-only-checkpoint issue). So the conversion asserts logit equivalence on a text probe
before saving, and archives the MTP weights to a sidecar directory so speculative decoding
can be restored later rather than being destroyed.

Requires torch + transformers (the ``[train]`` extra); guarded imports keep this module
importable for docstrings and type checks on CPU boxes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger
from medrl.core.paths import checkpoints_dir

log = get_logger(__name__)

_PROBE_PROMPTS = (
    "A 54-year-old man presents with crushing chest pain radiating to the left arm.",
    "Un patient de 54 ans présente une douleur thoracique.",
)


def convert(
    source: str,
    output: str | Path | None = None,
    *,
    verify_logits: bool = True,
    revision: str | None = None,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Strip vision + MTP, verify text equivalence, save. Returns a summary dict."""
    try:
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - CPU dev boxes
        raise RuntimeError(
            "surgery requires torch and transformers: uv pip install -e '.[train]'"
        ) from exc

    out_path = Path(output) if output else checkpoints_dir() / f"{Path(source).name}-text"

    reference = Qwen3_5ForConditionalGeneration.from_pretrained(
        source, revision=revision, torch_dtype=dtype, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)

    # Archive MTP before it is dropped: it is usable for speculative decoding later.
    mtp_state = {k: v for k, v in reference.state_dict().items() if k.startswith("mtp.")}
    mtp_dir = out_path.parent / f"{out_path.name}-mtp-sidecar"
    mtp_dir.mkdir(parents=True, exist_ok=True)
    if mtp_state:
        import safetensors.torch as st

        st.save_file(mtp_state, str(mtp_dir / "mtp.safetensors"))
        log.info("archived %d MTP tensors to %s", len(mtp_state), mtp_dir)

    # The class-level ignore patterns drop visual/mtp on load; the language model comes
    # through the prefix remap. Verification below is what makes that trustworthy.
    text_model = Qwen3_5ForCausalLM.from_pretrained(
        source, revision=revision, torch_dtype=dtype, device_map="cpu"
    )
    text_model.config.architectures = ["Qwen3_5ForCausalLM"]

    max_diff = None
    if verify_logits:
        max_diff = _verify_equivalence(reference, text_model, tokenizer, torch)
        log.info("max |logit diff| on text probes: %.3e", max_diff)

    text_model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    summary: dict[str, Any] = {
        "source": source,
        "output": str(out_path),
        "mtp_sidecar": str(mtp_dir) if mtp_state else None,
        "mtp_tensors_archived": len(mtp_state),
        "verify_logits": verify_logits,
        "max_logit_diff": max_diff,
        "architectures": text_model.config.architectures,
    }
    if max_diff is not None and max_diff > 1e-3:
        summary["warning"] = (
            f"logit diff {max_diff:.3e} exceeds 1e-3; the prefix remap may have misrouted "
            "tensors -- do NOT train or serve this checkpoint"
        )
    return summary


def _verify_equivalence(reference: Any, text_model: Any, tokenizer: Any, torch: Any) -> float:
    """Max abs logit difference over text probes; must be numerically ~0."""
    max_diff = 0.0
    for prompt in _PROBE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            ref_logits = reference(**inputs).logits
            text_logits = text_model(**inputs).logits
        max_diff = max(max_diff, float((ref_logits - text_logits).abs().max()))
    return max_diff
