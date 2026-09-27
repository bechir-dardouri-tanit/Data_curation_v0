"""S0 -- acquire and register the source pools.

For each configured source: download via HF hub (pinned revision when known),
count rows, compute a canonical content sha256, resolve licence provenance, and
emit ``registry.jsonl``. This stage never touches the corpus table -- it produces
the provenance ledger the rest of the pipeline cites.

Row counts are recorded honestly: ``n_rows_after_filters`` stays None unless a
source-level filter actually changed the count (the ChatDoctor-112k "filter" was
a no-op and the registry says so, rather than inheriting a phantom step).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from medrl.curation.schema import SourceRecord, StageError, StageManifest, utcnow
from medrl.curation.store import content_sha256, rows_sha256, stage_dir, write_registry

REGISTRY_SOURCES: dict[str, dict[str, Any]] = {
    # source_id -> hub id + expected shape. Counts/licences verified 2026-09-26
    # (dataset-audit workflow); licences tagged "unknown" are recorded as unknown
    # with the hub page as licence_source until resolved upstream.
    "ii_medical_reasoning_sft": {
        "hf_id": "Intelligent-Internet/II-Medical-Reasoning-SFT", "expected_rows": 2_197_741,
        "lang": "en", "has_cot": True, "licence": "unknown"},
    "finemed_sft": {
        "hf_id": "hongzhouyu/FineMed-SFT", "expected_rows": 731_992,
        "lang": "mixed", "has_cot": True, "licence": "mit"},
    "chatdoctor_healthcaremagic": {
        "hf_id": "lavita/ChatDoctor-HealthCareMagic-100k", "expected_rows": 112_165,
        "lang": "en", "has_cot": False, "licence": "unknown"},
    "generalthought_biology": {
        "hf_id": "GeneralReasoning/GeneralThought-430K", "expected_rows": 430_788,
        "lang": "en", "has_cot": True, "licence": "mit",
        "subset": "human-biology"},  # filtered to the biology slice at S1, count recorded
    "medical_r1_distill": {
        "hf_id": "FreedomIntelligence/Medical-R1-Distill-Data", "expected_rows": 22_000,
        "lang": "en", "has_cot": True, "licence": "apache-2.0"},
    "m23k_tokenized": {
        "hf_id": "UCSC-VLAA/m23k-tokenized", "expected_rows": 23_493,
        "lang": "en", "has_cot": True, "licence": "unknown"},
    "medreason": {
        "hf_id": "UCSC-VLAA/MedReason", "expected_rows": 32_682,
        "lang": "en", "has_cot": True, "licence": "apache-2.0"},
    "huatuo_o1_reasoning": {
        "hf_id": "FreedomIntelligence/medical-o1-reasoning-SFT", "expected_rows": 44_600,
        "lang": "en", "has_cot": True, "licence": "apache-2.0",
        "note": "en (19.7k) + en_mix (24.9k) subsets; zh excluded"},
    "finemed_dpo": {
        "hf_id": "hongzhouyu/FineMed-DPO", "expected_rows": 32_919,
        "lang": "en", "has_cot": True, "licence": "apache-2.0"},
    "ii_medical_rl": {
        "hf_id": "Intelligent-Internet/II-Medical-RL", "expected_rows": 15_910,
        "lang": "en", "has_cot": True, "licence": "unknown"},
    "chatdoctor_rl": {
        "hf_id": "Intelligent-Internet/ChatDoctor-RL", "expected_rows": 16_749,
        "lang": "en", "has_cot": False, "licence": "unknown"},
}

HUB_DATASET_URL = "https://huggingface.co/api/datasets/{hf_id}"


class SourcePlan(BaseModel, frozen=True):
    """One source to acquire, resolved from REGISTRY_SOURCES (+ optional overrides)."""

    source_id: str
    hf_id: str
    expected_rows: int
    lang: str = "en"
    has_cot: bool = False
    licence: str = "unknown"
    subset: str | None = None
    note: str | None = None


def plans() -> list[SourcePlan]:
    return [SourcePlan(source_id=k, **v) for k, v in REGISTRY_SOURCES.items()]


def fetch_hub_metadata(hf_id: str) -> dict[str, Any]:
    """Revision + licence tag straight from the hub API (public, token-optional)."""
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["authorization"] = f"Bearer {token}"
    r = httpx.get(HUB_DATASET_URL.format(hf_id=hf_id), headers=headers, timeout=30,
                  follow_redirects=True)
    r.raise_for_status()
    meta = r.json()
    return {
        # hub may 307 to a renamed repo (GeneralThought-430K -> RJT1990/GeneralThoughtArchive);
        # record the RESOLVED id so downstream snapshots cite what was actually read
        "resolved_id": meta.get("id", hf_id),
        "sha": meta.get("sha"),
        "licence_tag": (meta.get("cardData") or {}).get("license")
        or (meta.get("cardData") or {}).get("licence"),
        "gated": bool(meta.get("gated")),
    }


def acquire(plan: SourcePlan, out_dir: Path) -> SourceRecord:
    """Download (or reuse) one source's data files and build its SourceRecord."""
    from huggingface_hub import snapshot_download

    meta = fetch_hub_metadata(plan.hf_id)
    path = Path(
        snapshot_download(
            plan.hf_id,
            repo_type="dataset",
            allow_patterns=["*.jsonl", "*.json", "*.parquet"],
            # Pin to the revision the ledger is about to record: floating on
            # latest main would let an upstream force-push between S0 and S1
            # make hf_revision/content_sha256 describe bytes nobody read.
            revision=meta["sha"],
        )
    )

    data_files = sorted(
        p for p in path.rglob("*") if p.suffix in {".jsonl", ".json", ".parquet"}
    )
    if not data_files:
        raise StageError(f"{plan.source_id}: no data files under {path}")
    size_bytes = sum(p.stat().st_size for p in data_files)

    # Content hash over the downloaded files in sorted order -- cheap, and the
    # row-level hash comes at S1 where rows are materialized. Chunked (1 MiB)
    # so multi-GB shards are hashed too: a size cutoff here once hashed those
    # files as a constant placeholder, so a corrupted re-download verified clean.
    fh = hashlib.sha256()
    for p in data_files:
        fh.update(p.name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                fh.update(chunk)

    licence = plan.licence
    licence_source = "plan-registry"
    if plan.licence == "unknown" and meta["licence_tag"]:
        licence = str(meta["licence_tag"])
        licence_source = "hub-tag"

    return SourceRecord(
        source_id=plan.source_id,
        url_or_hf_id=meta.get("resolved_id", plan.hf_id),
        hf_revision=meta["sha"],
        content_sha256=fh.hexdigest(),
        n_rows=plan.expected_rows,  # replaced by the real count at S1 materialization
        licence=licence,
        licence_source=licence_source,
        redistributable=None if licence == "unknown" else True,
        lang=plan.lang,
        has_cot=plan.has_cot,
        known_contamination_risk="synthetic-traces" if plan.has_cot else None,
        downloaded_at=utcnow(),
        size_bytes=size_bytes,
    )


def run_registry(run_id: str, plans_override: list[SourcePlan] | None = None) -> StageManifest:
    started = utcnow()
    out = stage_dir(run_id, "00_registry")
    manifest = StageManifest(
        run_id=run_id, stage="00_registry", started_at=started,
        config={"n_sources": len(plans_override or plans())},
    )
    records: list[SourceRecord] = []
    errors: dict[str, str] = {}
    for plan in plans_override or plans():
        try:
            records.append(acquire(plan, out))
        except Exception as exc:  # noqa: BLE001 -- manifest records every failure
            errors[plan.source_id] = str(exc)[:300]
    out_path = write_registry(records, run_id)
    manifest.notes = {
        "acquired": [r.source_id for r in records],
        "failed": errors,
        "registry_path": str(out_path),
        "rows_sha256": rows_sha256_placeholder(records),
    }
    manifest.rows_out = len(records)
    return manifest


def rows_sha256_placeholder(records: list[SourceRecord]) -> str:
    import hashlib

    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x.source_id):
        h.update(r.model_dump_json().encode())
    return h.hexdigest()


def iter_source_files(record: SourceRecord) -> Iterator[Path]:
    """Yield the downloaded data files for one acquired source.

    Same revision + allow_patterns as :func:`acquire`: a floating revision
    would let S1 read different bytes than the ones S0 hashed, and a wider
    file set would read data the ledger never saw (a repo shipping both .json
    and .jsonl exports of the same data would be read twice, duplicating every
    row under distinct row-position ids).
    """
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            record.url_or_hf_id,
            repo_type="dataset",
            allow_patterns=["*.jsonl", "*.json", "*.parquet"],
            revision=record.hf_revision,
        )
    )
    for p in sorted(path.rglob("*")):
        if p.suffix in {".jsonl", ".json", ".parquet"}:
            yield p


__all__ = [
    "REGISTRY_SOURCES",
    "SourcePlan",
    "acquire",
    "fetch_hub_metadata",
    "iter_source_files",
    "plans",
    "run_registry",
]
