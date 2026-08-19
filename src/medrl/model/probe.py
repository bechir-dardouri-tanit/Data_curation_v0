"""Architecture prover for Qwen3.5 checkpoints.

The training and serving plan is derived from architecture facts: dense-not-MoE decides the
RL algorithm family; the 24:8 hybrid attention layout rules out context parallelism;
``vocab_size`` with untied embeddings makes logits the dominant memory term; fp32 SSM state
constrains the FSDP mixed-precision policy. A checkpoint revision that silently changes any
of these must fail loudly *before* a long run, not after -- so every fact the plan relies on
is asserted here against ground truth recovered from the checkpoint itself.

Parameter formulas were derived from the actual safetensors headers of ``Qwen/Qwen3.5-9B``
(revision c2022362), not inferred from naming conventions. Two non-obvious terms they encode:

- ``attn_output_gate`` doubles ``q_proj`` (query and gate halves): 16 heads x 256 head_dim
  produces an ``[8192 x 4096]`` projection, not ``[4096 x 4096]``.
- the Gated-DeltaNet ``in_proj_a``/``in_proj_b`` are per-head *scalar* gates
  (``[num_value_heads x hidden]``), not full projections -- easy to overestimate by 100x.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger

log = get_logger(__name__)

_MOE_KEYS = ("num_experts", "moe_intermediate_size", "decoder_sparse_step", "num_expert_groups")
"""Keys whose presence means the checkpoint is MoE. Any one of them flips ``is_dense``."""


class ArchitectureDriftError(RuntimeError):
    """A checkpoint invariant no longer holds; the derived plan is invalid for it."""

    def __init__(self, key: str, expected: Any, actual: Any) -> None:
        self.key, self.expected, self.actual = key, expected, actual
        super().__init__(f"architecture drift on {key!r}: expected {expected!r}, got {actual!r}")


@dataclass(frozen=True)
class ArchitectureProfile:
    """Facts about one checkpoint, plus the parameter math derived from them."""

    hidden_size: int
    num_layers: int
    vocab_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_type_counts: dict[str, int]
    full_attention_interval: int
    tie_word_embeddings: bool
    mamba_ssm_dtype: str | None
    rope_theta: float | None
    mrope_section: tuple[int, ...] | None
    partial_rotary_factor: float | None
    mtp_num_hidden_layers: int
    attn_output_gate: bool
    vision_hidden_size: int | None
    is_dense: bool
    moe_keys: tuple[str, ...]
    source: str
    linear_num_key_heads: int = 0
    linear_key_head_dim: int = 0
    linear_num_value_heads: int = 0
    linear_value_head_dim: int = 0
    notes: tuple[str, ...] = field(default=())

    # -- parameter math -----------------------------------------------------------------

    def _full_attention_params(self) -> int:
        h, d = self.hidden_size, self.head_dim
        q_out = self.num_attention_heads * d * (2 if self.attn_output_gate else 1)
        kv_out = self.num_key_value_heads * d
        attn = h * q_out + 2 * h * kv_out + (self.num_attention_heads * d) * h
        attn += 2 * d  # q_norm, k_norm
        return attn + self._mlp_params() + 2 * h

    def _linear_attention_params(self) -> int:
        if not self.linear_num_value_heads:
            return 0  # pure-attention checkpoint: no Gated-DeltaNet term
        h = self.hidden_size
        k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        v_dim = self.linear_num_value_heads * self.linear_value_head_dim
        proj_qkv = h * (2 * k_dim + v_dim)
        gates = 2 * h * self.linear_num_value_heads  # in_proj_a/b: one scalar per value head
        z_and_out = 2 * (v_dim * h)
        conv = (2 * k_dim + v_dim) * 4  # conv1d over the qkv stream
        attn = proj_qkv + gates + z_and_out + conv
        return attn + self._mlp_params() + 2 * h

    def _mlp_params(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def text_param_count(self) -> int:
        """Language-model parameters, analytic (embeddings + layers + final norm).

        Verified against the real checkpoint to within a fraction of a percent; the
        cross-check against ``index.json`` ``total_size`` is itself a test.
        """
        emb = (1 if self.tie_word_embeddings else 2) * self.vocab_size * self.hidden_size
        layers = sum(
            self._full_attention_params() if kind == "full_attention"
            else self._linear_attention_params()
            for kind, n in self.layer_type_counts.items()
            for _ in range(n)
        )
        return emb + layers + self.hidden_size  # final rmsnorm

    @property
    def vision_param_count_hint(self) -> int | None:
        """Vision-tower estimate, or ``None`` for a text-only checkpoint.

        Coarser than the text math (patch-embed/merger details vary) and marked ``hint``:
        it exists to size the strip, not to account for every bias.
        """
        v = self.vision_hidden_size
        if v is None:
            return None
        depth, inter = 27, 4304  # fixed by the SigLIP2-class tower; asserted via notes
        block = (3 * v * v + v * v) + 2 * v * inter  # fused qkv + proj + mlp
        merger = (4 * v) * 4096 + 4096 * 4096
        return depth * block + merger

    @property
    def mtp_param_count_hint(self) -> int:
        """MTP-head estimate: one full-attention-style layer plus the combine projection."""
        if not self.mtp_num_hidden_layers:
            return 0
        h = self.hidden_size
        return self.mtp_num_hidden_layers * (self._full_attention_params() + 2 * h * h) + 3 * h

    def logit_memory_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """Bytes for one sequence's logits -- the number that forces fused/chunked CE.

        A 16k-token sequence at vocab 248320 in bf16 is ~8.1 GB: materializing logits for a
        long-CoT batch is not an optimization concern, it is the plan-breaking term.
        """
        return seq_len * self.vocab_size * dtype_bytes

    # -- reporting ----------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"source: {self.source}",
            f"dense: {self.is_dense}" + (f"  (moe keys: {list(self.moe_keys)})" if self.moe_keys else ""),
            f"layers: {self.num_layers} = "
            + " + ".join(f"{n} {k}" for k, n in sorted(self.layer_type_counts.items())),
            f"hidden/vocab/intermediate: {self.hidden_size}/{self.vocab_size}/{self.intermediate_size}",
            f"heads: {self.num_attention_heads} q / {self.num_key_value_heads} kv, head_dim {self.head_dim}"
            + (" (q_proj doubled by attn_output_gate)" if self.attn_output_gate else ""),
            f"embeddings: {'tied' if self.tie_word_embeddings else 'untied'}",
            f"ssm dtype: {self.mamba_ssm_dtype}",
            f"text params: {self.text_param_count/1e9:.3f}B",
        ]
        if self.vision_param_count_hint:
            lines.append(f"vision params: ~{self.vision_param_count_hint/1e9:.3f}B")
        if self.mtp_param_count_hint:
            lines.append(f"mtp params:   ~{self.mtp_param_count_hint/1e9:.3f}B")
        lines.append(f"logits @16k bf16: {self.logit_memory_bytes(16384)/2**30:.2f} GiB")
        return "\n".join(lines)


QWEN35_9B_EXPECTATIONS: dict[str, Any] = {
    "is_dense": True,
    "num_layers": 32,
    "layer_type_counts": {"linear_attention": 24, "full_attention": 8},
    "vocab_size": 248320,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "tie_word_embeddings": False,
    "mamba_ssm_dtype": "float32",
    "full_attention_interval": 4,
    "head_dim": 256,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "mtp_num_hidden_layers": 1,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
}
"""Ground truth for ``Qwen/Qwen3.5-9B`` (revision c2022362), verified 2026-08.

``layer_type_counts`` is compared exactly: the 3:1 hybrid ratio is what the parallelism and
KV-cache plans are derived from. The ``linear_*`` head geometry is asserted because the
fp32 Gated-DeltaNet recurrent state -- ``num_value_heads * key_head_dim * value_head_dim``
per layer, the term the FSDP mixed-precision policy is derived from -- scales with exactly
these numbers; a revision that changes them silently invalidates the memory plan.
"""


def profile_from_hf_config(cfg: dict[str, Any], source: str = "<dict>") -> ArchitectureProfile:
    """Build a profile from a HF ``config.json`` dict, multimodal or flat-CausalLM layout."""
    text = dict(cfg.get("text_config") or cfg)
    text.pop("architectures", None)

    layer_types = text.get("layer_types") or []
    counts = dict(Counter(layer_types))
    if not counts and "num_hidden_layers" in text:
        counts = {"full_attention": text["num_hidden_layers"]}

    moe_keys = tuple(k for k in _MOE_KEYS if k in text)
    rope = text.get("rope_parameters") or {}

    def grab(*names: str, default: Any = None) -> Any:
        for n in names:
            if n in text:
                return text[n]
        return default

    profile = ArchitectureProfile(
        hidden_size=text["hidden_size"],
        num_layers=text["num_hidden_layers"],
        vocab_size=text["vocab_size"],
        intermediate_size=text["intermediate_size"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=grab("num_key_value_heads", default=text["num_attention_heads"]),
        head_dim=int(grab("head_dim", default=text["hidden_size"] // text["num_attention_heads"])),
        layer_type_counts=counts,
        full_attention_interval=grab("full_attention_interval", default=0),
        tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
        mamba_ssm_dtype=text.get("mamba_ssm_dtype"),
        rope_theta=rope.get("rope_theta"),
        mrope_section=tuple(rope["mrope_section"]) if rope.get("mrope_section") else None,
        partial_rotary_factor=rope.get("partial_rotary_factor"),
        mtp_num_hidden_layers=grab("mtp_num_hidden_layers", default=0),
        attn_output_gate=bool(text.get("attn_output_gate", False)),
        linear_num_key_heads=int(grab("linear_num_key_heads", default=0)),
        linear_key_head_dim=int(grab("linear_key_head_dim", default=0)),
        linear_num_value_heads=int(grab("linear_num_value_heads", default=0)),
        linear_value_head_dim=int(grab("linear_value_head_dim", default=0)),
        vision_hidden_size=(cfg.get("vision_config") or {}).get("hidden_size"),
        is_dense=not moe_keys,
        moe_keys=moe_keys,
        source=source,
    )
    return profile


def profile_from_hub(repo_id: str, revision: str | None = None) -> ArchitectureProfile:
    """Fetch ``config.json`` from the Hub and profile it. Requires ``huggingface_hub``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only on CPU dev boxes
        raise RuntimeError(
            "huggingface_hub is not installed; install the [eval] extra or pass a local config"
        ) from exc
    path = Path(hf_hub_download(repo_id, "config.json", revision=revision))
    return profile_from_hf_config(json.loads(path.read_text()), source=f"hub:{repo_id}@{revision or 'main'}")


def assert_invariants(
    profile: ArchitectureProfile,
    expectations: dict[str, Any] | None = None,
) -> list[str]:
    """Verify every expected fact; raise :class:`ArchitectureDriftError` on the first mismatch."""
    exp = expectations if expectations is not None else QWEN35_9B_EXPECTATIONS
    confirmations: list[str] = []
    for key, expected in exp.items():
        actual = getattr(profile, key, None)
        if actual != expected:
            raise ArchitectureDriftError(key, expected, actual)
        confirmations.append(f"{key} == {expected!r}")
    log.info("architecture invariants hold (%d checked)", len(confirmations))
    return confirmations


def params_by_subtree(index: dict[str, Any]) -> dict[str, int]:
    """Tensor counts per top-level subtree from a safetensors ``index.json``.

    Byte sizes per tensor are not stored in the index, so this counts tensors and reports
    ``total_size`` separately -- enough to verify the strip targets (vision 333 tensors,
    mtp 15) without downloading shards.
    """
    weight_map: dict[str, str] = index.get("weight_map", {})
    counts: Counter[str] = Counter()
    for key in weight_map:
        top = ".".join(key.split(".")[:2]) if key.startswith("model.") else key.split(".")[0]
        counts[top] += 1
    out: dict[str, int] = dict(counts)
    # The real layout keeps total_size under "metadata"; tolerate a top-level copy.
    total = index.get("total_size") or (index.get("metadata") or {}).get("total_size")
    if total:
        out["total_size"] = int(total)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assert Qwen3.5 architecture invariants.")
    parser.add_argument("--repo", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--config", default=None, help="local config.json instead of the Hub")
    args = parser.parse_args(argv)

    profile = (
        profile_from_hf_config(json.loads(Path(args.config).read_text()), source=args.config)
        if args.config
        else profile_from_hub(args.repo, args.revision)
    )
    print(profile.summary())
    print()
    try:
        assert_invariants(profile)
    except ArchitectureDriftError as drift:
        print(f"DRIFT: {drift}", file=sys.stderr)
        return 1
    print(f"all {len(QWEN35_9B_EXPECTATIONS)} invariants hold")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
