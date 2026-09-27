"""Single source of truth for every numeric threshold in the curation pipeline.

A lint test (tests/curation/test_thresholds_discipline.py) greps the stage modules
for bare numeric literals that look like thresholds -- if a stage needs a new knob,
it gets a name here first, then a calibration entry. Thresholds imported anywhere
else are a review failure.

Calibration status is recorded per constant: values marked PILOT must be
recalibrated at the B7 8% pilot before any full-scale run (the semantic thresholds
were bge-m3-calibrated; Qwen3-Embedding's cosine distribution differs -- see the
plan, section 1.1).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Calibration(StrEnum):
    SET = "set"  # decided; change only with a manifest-recorded reason
    PILOT = "pilot"  # provisional -- recalibrate at the 8% pilot before full scale


class Thresholds(BaseModel, frozen=True):
    """Every knob, one place. Frozen so a stage cannot drift its own config."""

    # ---- S2 structural ----------------------------------------------------
    min_tokens: int = 8
    max_tokens: int = 32768  # cluster max_model_len is 40960; leave headroom
    repetition_span_chars: int = 50
    repetition_max_occurrences: int = 4
    encoding_damage_max_ratio: float = 0.001

    # ---- S3 dedup ----------------------------------------------------------
    minhash_num_perm: int = 128
    minhash_shingle_n: int = 5  # word shingles
    minhash_jaccard: float = 0.80  # plan value; repo default was 0.9 -- plan wins, pilot confirms
    ngram_dup_n: int = 13  # word-level, matching the eval-side decontam convention
    ngram_dup_jaccard: float = 0.90

    # ---- S4 lexical decontamination ----------------------------------------
    contam_ngram_n: int = 13
    contam_ngram_threshold: float = 0.80

    # ---- S5/S6 embeddings (PILOT: bge-m3-calibrated values, model is Qwen3-Embedding-4B)
    embed_dim_store: int = 2560  # full-dim fp16 in Parquet
    embed_dim_index: int = 1024  # MRL-truncated for usearch; recall curve logged at pilot
    semantic_dup_threshold: float = Field(
        default=0.95, description="PILOT: recalibrate for Qwen3-Embedding"
    )
    semantic_contam_threshold: float = Field(
        default=0.90, description="PILOT: recalibrate for Qwen3-Embedding"
    )
    lid_confidence_floor: float = 0.90
    lid_min_chars: int = 20  # below this, straight to GlotLID second opinion
    lid_concat_chars: int = 200  # LID runs on a >=200-char question concatenation

    # ---- S8 concept ---------------------------------------------------------
    concept_jaccard: float = 0.85

    # ---- S9 answers ----------------------------------------------------------
    numeric_rtol: float = 0.005  # mirrors eval/verifiers/numbers.py
    numeric_atol: float = 1e-8

    # ---- S10 grounding ---------------------------------------------------------
    grounding_drop_only_contradicted: bool = True

    # ---- S12 difficulty ---------------------------------------------------------
    passk_samples: int = 8
    passk_temperature: float = 0.6
    passk_max_tokens: int = 10240  # decision_grade.yaml precedent
    passk_think_budget: int = 8192
    passk_topup_k: tuple[int, ...] = (3, 4, 7, 8)  # boundary counts that trigger +8 samples
    passk_topup_samples: int = 8
    band_downsample: float = 1.0  # pass == 1.0 -> downsample
    band_sft1_low: float = 0.5
    band_rl_low: float = 0.1  # 0.1-0.4 -> RL pool (the gradient band)
    band_rl_high: float = 0.4
    # 0.0 -> hold. Boundaries are inclusive-lower/exclusive-upper except the ends.

    # ---- S14 generation --------------------------------------------------------
    gen_temperature: float = 0.7
    gen_top_p: float = 0.95
    gen_samples_per_prompt: int = 6
    gen_max_kept_per_prompt: int = 2


THRESHOLDS = Thresholds()


def snapshot(thresholds: Thresholds = THRESHOLDS) -> dict[str, object]:
    """Flat dict for StageManifest.thresholds -- proof of what a stage actually used."""
    return thresholds.model_dump()


__all__ = ["THRESHOLDS", "Calibration", "Thresholds", "snapshot"]
