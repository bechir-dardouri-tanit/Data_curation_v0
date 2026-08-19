"""Architecture prover tests, run against the real Qwen3.5-9B config as a fixture."""

from __future__ import annotations

import copy

import pytest

from medrl.model.probe import (
    QWEN35_9B_EXPECTATIONS,
    ArchitectureDriftError,
    assert_invariants,
    params_by_subtree,
    profile_from_hf_config,
)

TEXT_CONFIG: dict = {
    "attention_bias": False,
    "attn_output_gate": True,
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "num_hidden_layers": 32,
    "rms_norm_eps": 1e-06,
    "vocab_size": 248320,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
    "tie_word_embeddings": False,
}
VISION_CONFIG: dict = {
    "depth": 27,
    "hidden_size": 1152,
    "in_channels": 3,
    "intermediate_size": 4304,
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
FULL_CONFIG: dict = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": dict(TEXT_CONFIG),
    "vision_config": dict(VISION_CONFIG),
}
# Ground truth recovered from the checkpoint itself (revision c2022362).
TOTAL_PARAMS = 19_306_216_416 // 2  # bf16


@pytest.fixture()
def profile():
    return profile_from_hf_config(copy.deepcopy(FULL_CONFIG), source="fixture")


def test_dense_detection(profile) -> None:
    assert profile.is_dense
    assert profile.moe_keys == ()
    assert_invariants(profile)


def test_moe_fixture_is_flagged() -> None:
    cfg = copy.deepcopy(FULL_CONFIG)
    cfg["text_config"]["num_experts"] = 8
    cfg["text_config"]["moe_intermediate_size"] = 768
    moe = profile_from_hf_config(cfg)
    assert not moe.is_dense
    assert set(moe.moe_keys) == {"num_experts", "moe_intermediate_size"}
    with pytest.raises(ArchitectureDriftError) as exc:
        assert_invariants(moe)
    assert exc.value.key == "is_dense"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vocab_size", 151936),
        ("tie_word_embeddings", True),
        ("mamba_ssm_dtype", "bfloat16"),
        ("head_dim", 128),
        ("mtp_num_hidden_layers", 0),
        ("attn_output_gate", False),
        # The fp32 SSM recurrent state scales with exactly these; halving one must fail
        # the probe, not silently invalidate the memory plan (regression: all four were
        # extracted but never asserted).
        ("linear_num_key_heads", 8),
        ("linear_key_head_dim", 64),
        ("linear_num_value_heads", 16),
        ("linear_value_head_dim", 64),
    ],
)
def test_drift_raised_per_key(profile, key: str, value: object) -> None:
    cfg = copy.deepcopy(FULL_CONFIG)
    cfg["text_config"][key] = value
    drifted = profile_from_hf_config(cfg)
    with pytest.raises(ArchitectureDriftError) as exc:
        assert_invariants(drifted)
    assert exc.value.key == key
    assert exc.value.expected == QWEN35_9B_EXPECTATIONS[key]


def test_hybrid_ratio_drift_is_detected() -> None:
    cfg = copy.deepcopy(FULL_CONFIG)
    layers = cfg["text_config"]["layer_types"]
    cfg["text_config"]["layer_types"] = layers[:-8] + ["full_attention"] * 8  # 24:16, not 24:8
    with pytest.raises(ArchitectureDriftError) as exc:
        assert_invariants(profile_from_hf_config(cfg))
    assert exc.value.key == "layer_type_counts"


def test_flat_causal_lm_layout_matches_nested() -> None:
    """A converted text-only config must profile identically to the nested one."""
    flat = profile_from_hf_config(copy.deepcopy(TEXT_CONFIG), source="flat")
    nested = profile_from_hf_config(copy.deepcopy(FULL_CONFIG), source="nested")
    for attr in (
        "hidden_size", "num_layers", "vocab_size", "layer_type_counts",
        "tie_word_embeddings", "mamba_ssm_dtype", "is_dense", "text_param_count",
    ):
        assert getattr(flat, attr) == getattr(nested, attr), attr
    assert nested.vision_param_count_hint is not None
    assert flat.vision_param_count_hint is None


def test_text_param_count_within_two_percent() -> None:
    """The analytic formula must match the real checkpoint's language-model size.

    Ground truth: total 9.653B params, vision 333 tensors ~0.45B, mtp 15 tensors ~0.24B
    (both bounded above by their analytic estimates), leaving ~8.95B text.
    """
    profile = profile_from_hf_config(copy.deepcopy(FULL_CONFIG))
    text = profile.text_param_count
    vision = profile.vision_param_count_hint or 0
    mtp = profile.mtp_param_count_hint
    assert text == pytest.approx(8.95e9, rel=0.02), f"text params {text/1e9:.3f}B"
    assert 0.40e9 <= vision <= 0.50e9, f"vision estimate {vision/1e9:.3f}B"
    assert 0.20e9 <= mtp <= 0.28e9, f"mtp estimate {mtp/1e9:.3f}B"
    # The three components must reconstruct the checkpoint total to within 2%.
    assert text + vision + mtp == pytest.approx(TOTAL_PARAMS, rel=0.02)


def test_param_math_scales_with_gate_flag() -> None:
    """attn_output_gate doubles q_proj -- 16M params per full-attention layer at this size."""
    on = profile_from_hf_config(copy.deepcopy(FULL_CONFIG))
    off = profile_from_hf_config(copy.deepcopy(FULL_CONFIG))
    object.__setattr__(off, "attn_output_gate", False)
    delta = on.text_param_count - off.text_param_count
    per_layer = 4096 * (16 * 256)  # the gate half of q_proj
    assert delta == pytest.approx(8 * per_layer)


def test_logit_memory_is_the_plan_breaking_term(profile) -> None:
    assert profile.logit_memory_bytes(16384, 2) == 16384 * 248320 * 2
    assert profile.logit_memory_bytes(16384, 2) / 2**30 == pytest.approx(7.56, abs=0.05)


def test_params_by_subtree_counts_and_total() -> None:
    index = {
        "metadata": {"total_size": 19_306_216_416},  # the real index.json layout
        "weight_map": {
            "lm_head.weight": "s1",
            **{f"model.language_model.layers.{i}.mlp.up_proj.weight": "s2" for i in range(4)},
            **{f"model.visual.blocks.{i}.attn.proj.weight": "s4" for i in range(3)},
            "mtp.layers.0.mlp.gate_proj.weight": "s2",
        },
    }
    out = params_by_subtree(index)
    assert out["model.language_model"] == 4
    assert out["model.visual"] == 3
    assert out["mtp"] == 1
    assert out["lm_head"] == 1
    assert out["total_size"] == 19_306_216_416


def test_summary_mentions_the_load_bearing_facts(profile) -> None:
    summary = profile.summary()
    assert "dense: True" in summary
    assert "24 linear_attention" in summary and "8 full_attention" in summary
    assert "untied" in summary
