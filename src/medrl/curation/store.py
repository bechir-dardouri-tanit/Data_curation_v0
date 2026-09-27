"""Parquet storage for the corpus table + content hashing.

Layout per run (heavy, never sole-copy)::

    /scratch/medrl/curation/<run_id>/00_registry/   registry.jsonl + raw source refs
    /scratch/medrl/curation/<run_id>/01_normalize/  part-*.parquet   (S1 snapshot)
    /scratch/medrl/curation/<run_id>/02_structural/ ...
    ...

Light mirrors (manifests, reports) live under ``experiments/curation/<run_id>/``
and are committed; :func:`mirror_light` performs the copy so the runner has one call.

The row schema is exactly :class:`medrl.curation.schema.CorpusItem` flattened via
pydantic -- one ``to_table``/``iter_items`` pair is the only parquet boundary in
the codebase, so a schema change is a one-file change plus a migration note.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from medrl.curation.schema import CorpusItem, Flags, SourceRecord, StageManifest, utcnow

SCRATCH_ROOT = Path("/scratch/medrl/curation")
EXPERIMENTS_ROOT = Path("/root/medrl/experiments/curation")


def run_dir(run_id: str) -> Path:
    return SCRATCH_ROOT / run_id


def stage_dir(run_id: str, stage: str) -> Path:
    """Output directory for one stage snapshot: ``NN_<stage>`` prefix-free, plain name."""
    d = run_dir(run_id) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_manifest_path(run_id: str, stage: str) -> Path:
    d = EXPERIMENTS_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stage}.manifest.json"


_FLAG_FIELDS = list(Flags.model_fields)  # stable flag order, single definition point

_ARROW_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("lang", pa.string()),
        ("lang_score", pa.float64()),
        ("licence", pa.string()),
        ("redistributable", pa.bool_()),
        # messages as a JSON string, like tools/meta: a typed struct(role, content)
        # column SILENTLY DROPS every other key a source carries (verified on
        # pyarrow 25: OpenAI `name`/`tool_call_id` vanish on the round trip), and
        # the schema contract is lossless with sources.
        ("messages", pa.string()),
        ("thinking", pa.string()),
        ("tools", pa.string()),
        ("answer", pa.string()),
        ("answer_type", pa.string()),
        ("meta", pa.string()),
        # Flags as an explicit bool struct -- queryable per-flag from DuckDB.
        *[("flags_" + name.removeprefix("f_"), pa.bool_()) for name in _FLAG_FIELDS],
        ("dup_of", pa.string()),
        ("contam_benchmark", pa.string()),
        ("embedding", pa.binary()),
        ("cuis", pa.list_(pa.string())),
        ("answer_cui", pa.string()),
        ("kg_triples", pa.string()),
        ("support_frac", pa.float64()),
        ("unknown_frac", pa.float64()),
        ("q_coherence", pa.int8()),
        ("q_clinical", pa.int8()),
        ("q_format", pa.int8()),
        ("difficulty", pa.float64()),
        ("difficulty_band", pa.string()),
        ("n_covered_by", pa.int64()),
    ]
)


def _to_arrow_row(it: CorpusItem) -> dict[str, Any]:
    row = it.model_dump()
    row["messages"] = (
        json.dumps(row["messages"], ensure_ascii=False, default=str) if row["messages"] else None
    )
    row["meta"] = json.dumps(row["meta"], sort_keys=True, default=str) if row["meta"] else None
    row["tools"] = json.dumps(row["tools"], default=str) if row["tools"] else None
    row["kg_triples"] = (
        json.dumps([list(t) for t in row["kg_triples"]]) if row["kg_triples"] else None
    )
    flags = row.pop("flags")
    for name, val in flags.items():
        row["flags_" + name.removeprefix("f_")] = val
    return row


def _from_arrow_row(row: dict[str, Any]) -> CorpusItem:
    flags = {}
    for name in _FLAG_FIELDS:
        flags[name] = row.pop("flags_" + name.removeprefix("f_")) or False
    row["flags"] = flags
    msg = row.get("messages")
    if isinstance(msg, str):  # current writer: JSON string
        row["messages"] = json.loads(msg) if msg else []
    elif msg is None:
        row["messages"] = []
    # else: legacy struct-decoded list-of-dicts -- already the right shape
    row["meta"] = json.loads(row["meta"]) if row.get("meta") else {}
    row["tools"] = json.loads(row["tools"]) if row.get("tools") else None
    row["kg_triples"] = (
        [tuple(t) for t in json.loads(row["kg_triples"])] if row.get("kg_triples") else []
    )
    for nullable in (
        "thinking",
        "answer",
        "dup_of",
        "contam_benchmark",
        "answer_cui",
        "difficulty_band",
        "lang_score",
        "support_frac",
        "unknown_frac",
        "q_coherence",
        "q_clinical",
        "q_format",
        "difficulty",
        "redistributable",
    ):
        if row.get(nullable) is None:
            row[nullable] = None
    return CorpusItem.model_validate(row)


def reset_dir(directory: Path) -> Path:
    """Remove stage outputs (parts, ANN files) so a re-run starts clean.

    Stages call this on their OUTPUT dir only -- never on an input snapshot.
    Append-across-calls semantics of write_items apply *within* one stage run.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.iterdir():
        if f.is_file() and (
            f.name.startswith("part-")
            or (f.suffix in {".usearch", ".json"} and f.stem.startswith("ann"))
        ):
            f.unlink()
    return directory


def write_items(
    items: Iterable[CorpusItem], directory: Path, max_rows_per_file: int = 50_000
) -> int:
    """Write items as part files; returns count written. Sorted by id for stable hashes.

    Multiple calls into the same directory APPEND (part numbering continues from
    the existing files), so chunked writes are safe; each call must receive a
    disjoint, internally-unique id set.
    """
    buf: list[CorpusItem] = []
    n = 0
    existing = [int(f.stem.split("-")[1]) for f in directory.glob("part-*.parquet")]
    part = (max(existing) + 1) if existing else 0

    def flush(rows: list[CorpusItem]) -> None:
        nonlocal part, n
        if not rows:
            return
        rows.sort(key=lambda it: it.id)
        table = pa.Table.from_pylist([_to_arrow_row(it) for it in rows], schema=_ARROW_SCHEMA)
        pq.write_table(table, directory / f"part-{part:05d}.parquet")
        part += 1
        n += len(rows)
        rows.clear()

    seen = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate id within a stage snapshot: {it.id}")
        seen.add(it.id)
        buf.append(it)
        if len(buf) >= max_rows_per_file:
            flush(buf)
    flush(buf)
    return n


def iter_items(directory: Path) -> Iterator[CorpusItem]:
    """Stream items back from a stage snapshot in id order."""
    for f in sorted(directory.glob("part-*.parquet")):
        table = pq.read_table(f)
        for row in table.to_pylist():
            yield _from_arrow_row(row)


def count_items(directory: Path) -> int:
    total = 0
    for f in sorted(directory.glob("part-*.parquet")):
        total += pq.read_metadata(f).num_rows
    return total


def content_sha256(directory: Path) -> str:
    """Stable dataset hash: sha256 over each part's bytes in sorted name order.

    Uses the parquet bytes directly -- fast, and equal snapshots of equal items
    hash equal regardless of write batching because part boundaries are
    deterministic (sorted ids, fixed max_rows_per_file).
    """
    h = hashlib.sha256()
    for f in sorted(directory.glob("part-*.parquet")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def source_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_sha256(items: Iterable[CorpusItem]) -> str:
    """sha256 over canonical row JSON -- provenance hash independent of part layout."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.id):
        h.update(it.id.encode())
        h.update(json.dumps(it.model_dump(), sort_keys=True, default=str).encode())
    return h.hexdigest()


def seal_manifest(manifest: StageManifest) -> StageManifest:
    """Fill timing fields; call right before saving."""
    now = utcnow()
    return manifest.model_copy(
        update={
            "finished_at": now,
            "wall_s": round((now - manifest.started_at).total_seconds(), 3),
        }
    )


def save_manifest(manifest: StageManifest) -> Path:
    p = stage_manifest_path(manifest.run_id, manifest.stage)
    p.write_text(manifest.model_dump_json(indent=2))
    return p


def mirror_light(run_id: str) -> list[Path]:
    """Copy every light record (manifests, reports) to experiments/ for commit."""
    src = run_dir(run_id)
    dst = EXPERIMENTS_ROOT / run_id
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for pattern in ("*.manifest.json", "*.md", "*.json", "*.sql"):
        for f in src.glob(pattern):
            if f.name.startswith("part-"):
                continue
            target = dst / f.name
            shutil.copy2(f, target)
            copied.append(target)
    return copied


def registry_path(run_id: str) -> Path:
    return run_dir(run_id) / "00_registry"


def write_registry(records: Iterable[SourceRecord], run_id: str) -> Path:
    d = registry_path(run_id)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "registry.jsonl"
    with out.open("w") as f:
        for rec in sorted(records, key=lambda r: r.source_id):
            f.write(rec.model_dump_json() + "\n")
    return out


def read_registry(run_id: str) -> list[SourceRecord]:
    out = registry_path(run_id) / "registry.jsonl"
    if not out.exists():
        return []
    return [
        SourceRecord.model_validate_json(line)
        for line in out.read_text().splitlines()
        if line.strip()
    ]


__all__ = [
    "EXPERIMENTS_ROOT",
    "SCRATCH_ROOT",
    "content_sha256",
    "count_items",
    "iter_items",
    "mirror_light",
    "read_registry",
    "reset_dir",
    "rows_sha256",
    "run_dir",
    "save_manifest",
    "seal_manifest",
    "source_sha256",
    "stage_dir",
    "stage_manifest_path",
    "write_items",
    "write_registry",
]
