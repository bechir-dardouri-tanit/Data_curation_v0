"""Canonical curation-corpus schema.

One row type for the whole S0-S15 pipeline. Every column a later stage fills is
reserved here from day one: stages never migrate the schema, they only ever write
into their reserved slots. Deletion is a query over ``flags``; nothing is dropped.

Two audit invariants:

* ``Flags`` only ever transitions False -> True (or None -> value). There is no
  unflag. A correction means re-running the stage from its snapshot, not editing
  rows in place.
* Every stage writes a :class:`StageManifest` recording parameters, row counts,
  per-source flag rates and content hashes, so the published reports are derived,
  never hand-written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

Message = dict[str, str]
"""OpenAI-shaped chat message; kept as a plain dict to stay lossless with sources."""

AnswerType = Literal["mcqa", "numeric", "free_text", "none"]
DifficultyBand = Literal["hold", "rl", "sft1", "downsample"]


class Flags(BaseModel):
    """Stage-set booleans. All default False; stages only ever set, never clear."""

    # S2 structural
    f_empty: bool = False
    f_length: bool = False
    f_truncated: bool = False
    f_repetition: bool = False
    f_lang: bool = False
    f_encoding: bool = False
    f_refusal: bool = False
    # S3-S6 dedup / decontamination
    f_dup_exact: bool = False
    f_dup_minhash: bool = False
    f_dup_semantic: bool = False
    f_dup_concept: bool = False
    f_contam_ngram: bool = False
    f_contam_semantic: bool = False
    f_contam_concept: bool = False
    # S9 answer verification
    f_answer_wrong: bool = False
    f_answer_right_reasoning_contradicts: bool = False
    # S10 grounding
    f_kg_contradicted: bool = False

    def any(self) -> bool:
        """True when any flag is set (the corpus-wide 'survivor' predicate)."""
        return any(self.model_dump().values())

    def set_names(self) -> tuple[str, ...]:
        """Names of every set flag, stable order -- for manifests and diffs."""
        return tuple(k for k, v in self.model_dump().items() if v)


class CorpusItem(BaseModel):
    """One corpus row. Reserved columns exist from S1; later stages fill them."""

    id: str
    """``<source_id>:<row_id>`` -- stable across pipeline runs."""

    source: str
    lang: str = "en"
    """LID result at S1; the S2 keep-rule decides membership, LID never drops."""
    lang_score: float | None = None
    """fastText probability; GlotLID second-opinion runs below the confidence floor."""

    licence: str = "unknown"
    redistributable: bool | None = None

    messages: list[Message] = Field(default_factory=list)
    thinking: str | None = None
    tools: list[Any] | None = None
    answer: str | None = None
    """Gold answer when the source has one; None is the common case for Pool B."""
    answer_type: AnswerType = "none"

    meta: dict[str, Any] = Field(default_factory=dict)
    """Source-row payload plus carried labels (e.g. FineMed quality/complexity verbatim)."""

    # ---- reserved: S3-S6 -------------------------------------------------
    flags: Flags = Field(default_factory=Flags)
    dup_of: str | None = None
    """Canonical id this row duplicates (S3/S6/S8)."""
    contam_benchmark: str | None = None
    """Which eval benchmark an n-gram/semantic/concept hit matched (first, strongest)."""

    # ---- reserved: S5 ----------------------------------------------------
    embedding: bytes | None = None
    """fp16-quantized little-endian vector; dim recorded in the run manifest."""

    # ---- reserved: S7/S8/S10 ----------------------------------------------
    cuis: list[str] = Field(default_factory=list)
    """Question concept ids; namespace recorded per-run in the manifest (cui|sctid|omop_id)."""
    answer_cui: str | None = None
    kg_triples: list[tuple[str, str, str]] = Field(default_factory=list)
    support_frac: float | None = None
    unknown_frac: float | None = None

    # ---- reserved: S11 ----------------------------------------------------
    q_coherence: int | None = None
    q_clinical: int | None = None
    q_format: int | None = None
    """Binary axis outcomes (0/1), not holistic scores."""

    # ---- reserved: S12 ----------------------------------------------------
    difficulty: float | None = None
    """Pass rate on [0,1] at the labelling model; None = not labelled or unretriable."""
    difficulty_band: DifficultyBand | None = None

    # ---- reserved: S13 ----------------------------------------------------
    n_covered_by: int = 0


class SourceRecord(BaseModel):
    """One entry of registry.jsonl (S0). Provenance for everything downstream."""

    source_id: str
    url_or_hf_id: str
    hf_revision: str | None = None
    """Pinned commit sha of the dataset repo at acquisition."""
    content_sha256: str
    """sha256 over the sorted (row_id, canonical-json) stream of the source."""
    n_rows: int
    n_rows_after_filters: int | None = None
    """Filled when a source-level filter is applied; None means the filter was a no-op
    or none applied -- the ChatDoctor-112k case is recorded honestly this way."""
    licence: str = "unknown"
    licence_source: str | None = None
    """Where the licence string came from: hub tag, dataset card, or upstream issue url."""
    redistributable: bool | None = None
    lang: str = "en"
    has_cot: bool = False
    has_tools: bool = False
    known_contamination_risk: str | None = None
    downloaded_at: datetime
    size_bytes: int = 0


class StageManifest(BaseModel):
    """Per-stage audit record, one JSON per stage per run. Reports derive from these."""

    run_id: str
    stage: str
    started_at: datetime
    finished_at: datetime | None = None
    wall_s: float | None = None

    config: dict[str, Any] = Field(default_factory=dict)
    """The effective StageConfig dump -- parameters as run, not as intended."""
    thresholds: dict[str, Any] = Field(default_factory=dict)
    """Snapshot of thresholds.py values the stage consumed (threshold drift is a diff away)."""

    rows_in: int = 0
    rows_out: int = 0
    """rows_out == rows_in always (flags-not-deletes); kept for the invariant check."""
    flag_rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    """flag name -> per-source set-rate; '_all' key for corpus-wide."""

    input_sha256: str | None = None
    output_sha256: str | None = None
    code_sha: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)
    """Stage-specific extras: dup pair counts, canary results, ANN recall curves..."""


class StageError(RuntimeError):
    """Raised when a stage cannot honour its contract; the runner records and re-raises."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "AnswerType",
    "CorpusItem",
    "DifficultyBand",
    "Flags",
    "Message",
    "SourceRecord",
    "StageError",
    "StageManifest",
    "utcnow",
]
