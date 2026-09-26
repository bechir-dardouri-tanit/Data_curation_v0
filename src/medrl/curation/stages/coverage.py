"""S13 -- coverage map: aggregate linked concepts into a measurable generation plan.

S14 generates "gap-targeted only" (plan section 4). Without this stage that
targeting is taste; with it, it is measurement: the bipartite item-concept
incidence built from ``CorpusItem.cuis`` is aggregated into a per-concept
coverage record (degree, sources, languages, difficulty bands), a
per-source/per-band coverage summary, and a ranked list of high-degree
concepts that no row of at least one corpus language covers -- exactly the
concepts where generating new rows buys the most coverage per GPU-hour.

Choices worth a reviewer's attention:

* **Degree proxy.** There is no real KG graph until S7/S10 land, so concept
  degree is proxied by ``n_covered_by`` = the number of snapshot rows whose
  ``cuis`` contain the concept. The schema reserves the same field name on the
  row, where it is filled with the degree of the row's most-covered concept
  (self included, so row-level and concept-level numbers reconcile exactly);
  0 for rows without cuis.
* **Gap = exact absence.** A concept "has a gap" in a language when zero rows
  of that language in the snapshot cover it. A fractional "near-zero" rule
  would need a coverage-ratio knob; THRESHOLDS has none and adding one is not
  a stage's call -- exact absence is the plan's own
  "high-degree-zero-coverage" wording (docs/CURATION_IMPLEMENTATION_PLAN.md,
  S13). Gap languages are the snapshot's OBSERVED languages, not a hardcoded
  {en, fr}: S2's keep-rule already restricts membership, and a non-production
  snapshot must still produce a sane map.
* **Counters, not DuckDB.** The map is a one-shot streaming group-by over a
  bipartite stream: two passes over the snapshot with plain ``Counter``
  aggregates keep memory O(distinct concepts), never O(rows), so a
  few-million-row snapshot streams fine. DuckDB earns its place at S15, where
  mixtures are interactive SQL over the ``flags_*`` columns; a single
  aggregate does not need a query engine.
* **No flags, no drops.** S13 is a pure reporter: every row streams through to
  the passthrough snapshot unchanged except ``n_covered_by`` (and flagged rows
  still count toward coverage -- mixture SQL applies the flag filters at S15),
  so ``flag_rates`` is honestly empty in the manifest.
* **No THRESHOLDS consumption.** Neither the gap predicate nor the degree
  proxy reads a knob; the one free parameter, the target-list depth ``top_n``,
  is a stage parameter (default 200) recorded in the manifest config -- the
  manifest's ``thresholds`` block stays empty, on the answers.py precedent of
  recording exactly what a stage consumed.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger
from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow

log = get_logger(__name__)

INPUT_STAGE = "12_difficulty"
"""Default input snapshot; any snapshot with filled ``cuis`` works via ``input_stage``."""

OUTPUT_STAGE = "13_coverage"

MAP_FILENAME = "coverage_map.json"
TARGETS_FILENAME = "generation_targets.csv"

CSV_COLUMNS: tuple[str, ...] = ("concept_id", "total_n", "lang_gaps", "suggested_pool")

DEFAULT_TOP_N = 200
"""Target-list depth. A stage parameter, not a THRESHOLDS knob: it bounds the
CSV the S14 planner reads, it never gates a row."""

NO_BAND = "none"
"""Band key for rows S12 has not banded (``difficulty_band`` None). JSON/CSV
keys are strings, so the unbadged case needs an explicit name to stay countable."""

NO_COVERAGE = 0
"""Row-level ``n_covered_by`` for rows without cuis: nothing covers them."""


# --------------------------------------------------------------------------
# Aggregates -- Counters only, so the pass never holds row ids per concept.
# --------------------------------------------------------------------------


@dataclass
class ConceptStats:
    """Per-concept coverage accumulated while the snapshot streams.

    Sub-counters, not per-language id sets: the gap predicate needs only
    counts, and counts are what keeps a few-million-row pass flat in memory.
    """

    n: int = 0
    sources: Counter[str] = field(default_factory=Counter)
    langs: Counter[str] = field(default_factory=Counter)
    bands: Counter[str] = field(default_factory=Counter)


@dataclass
class CorpusTally:
    """Whole-corpus counters accumulated alongside the per-concept ones."""

    total_items: int = 0
    items_with_cuis: int = 0
    items_per_source: Counter[str] = field(default_factory=Counter)
    with_cuis_per_source: Counter[str] = field(default_factory=Counter)
    items_per_band: Counter[str] = field(default_factory=Counter)
    with_cuis_per_band: Counter[str] = field(default_factory=Counter)
    langs: Counter[str] = field(default_factory=Counter)
    """Language -> row count; the universe against which gaps are measured."""


def _band_key(item: CorpusItem) -> str:
    """Band label for tallying; unbanded rows count under NO_BAND, never vanish."""
    return item.difficulty_band if item.difficulty_band is not None else NO_BAND


def collect(items: Iterator[CorpusItem]) -> tuple[dict[str, ConceptStats], CorpusTally]:
    """Pass 1: one streaming group-by -> per-concept stats + corpus tallies."""
    concepts: dict[str, ConceptStats] = {}
    tally = CorpusTally()
    for item in items:
        tally.total_items += 1
        tally.langs[item.lang] += 1
        tally.items_per_source[item.source] += 1
        band = _band_key(item)
        tally.items_per_band[band] += 1
        if not item.cuis:
            continue
        tally.items_with_cuis += 1
        tally.with_cuis_per_source[item.source] += 1
        tally.with_cuis_per_band[band] += 1
        for cui in item.cuis:
            stats = concepts.get(cui)
            if stats is None:
                stats = ConceptStats()
                concepts[cui] = stats
            stats.n += 1
            stats.sources[item.source] += 1
            stats.langs[item.lang] += 1
            stats.bands[band] += 1
    return concepts, tally


# --------------------------------------------------------------------------
# coverage_map.json
# --------------------------------------------------------------------------


def concept_payload(concepts: Mapping[str, ConceptStats]) -> list[dict[str, Any]]:
    """Concept records in generation-priority order: degree desc, id asc.

    The id tiebreak makes the payload a pure function of the row multiset, not
    of the input snapshot's parquet part layout.
    """
    ranked = sorted(concepts.items(), key=lambda kv: (-kv[1].n, kv[0]))
    return [
        {
            "id": cui,
            "n_covered_by": stats.n,
            "sources": dict(sorted(stats.sources.items())),
            "langs": dict(sorted(stats.langs.items())),
            "bands": dict(sorted(stats.bands.items())),
        }
        for cui, stats in ranked
    ]


def build_coverage_map(concepts: Mapping[str, ConceptStats], tally: CorpusTally) -> dict[str, Any]:
    """The full coverage_map.json payload, exactly the contracted five keys."""
    by_source = {
        src: {
            "items": tally.items_per_source[src],
            "items_with_cuis": tally.with_cuis_per_source[src],
            "frac": (
                tally.with_cuis_per_source[src] / tally.items_per_source[src]
                if tally.items_per_source[src]
                else 0.0
            ),
        }
        for src in sorted(tally.items_per_source)
    }
    by_band = {
        band: {
            "items": tally.items_per_band[band],
            "items_with_cuis": tally.with_cuis_per_band[band],
        }
        for band in sorted(tally.items_per_band)
    }
    return {
        "total_items": tally.total_items,
        "items_with_cuis": tally.items_with_cuis,
        "concepts": concept_payload(concepts),
        "coverage_by_source": by_source,
        "by_band": by_band,
    }


# --------------------------------------------------------------------------
# generation_targets.csv
# --------------------------------------------------------------------------


def generation_targets(
    concepts: Mapping[str, ConceptStats],
    corpus_langs: Iterable[str],
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, str | int]]:
    """Top-``top_n`` concepts by degree that at least one corpus language misses.

    Rows keep the degree ranking (highest-degree gap first): the CSV is a
    generation budget, spent where degree says demand is highest. ``lang_gaps``
    is the sorted gap languages joined with ';' -- one csv cell, trivially
    splittable. ``suggested_pool`` is the band distribution of the items that
    DO cover the concept (a JSON object): the recipe S14's synthetic rows
    should mimic to land in the same training lanes.
    """
    langs = sorted(set(corpus_langs))
    ranked = sorted(concepts.items(), key=lambda kv: (-kv[1].n, kv[0]))[:top_n]
    rows: list[dict[str, str | int]] = []
    for cui, stats in ranked:
        gaps = [lang for lang in langs if stats.langs[lang] == 0]
        if not gaps:
            continue
        rows.append(
            {
                "concept_id": cui,
                "total_n": stats.n,
                "lang_gaps": ";".join(gaps),
                "suggested_pool": json.dumps(dict(sorted(stats.bands.items()))),
            }
        )
    return rows


def write_generation_targets(rows: list[dict[str, str | int]], path: Path) -> None:
    """The contracted CSV: header always, then one row per target."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([row[column] for column in CSV_COLUMNS])


# --------------------------------------------------------------------------
# Passthrough -- n_covered_by filled per row.
# --------------------------------------------------------------------------


def item_covered_by(item: CorpusItem, degrees: Mapping[str, int]) -> int:
    """Degree of the row's most-covered concept (self included); 0 without cuis.

    Max is the row-level reading of ``n_covered_by``: a row is at least as
    replaceable as its most-covered concept, which is the conservative number
    a mixture planner should see. A sum would double-count one hot concept the
    row mentions twice; a min would let a single rare concept mask four hot
    ones.
    """
    if not item.cuis:
        return NO_COVERAGE
    return max(degrees.get(cui, NO_COVERAGE) for cui in item.cuis)


def _passthrough(items: Iterator[CorpusItem], degrees: Mapping[str, int]) -> Iterator[CorpusItem]:
    """Stream rows through with ``n_covered_by`` filled; untouched rows pass as-is.

    Copying only changed rows keeps the output byte-stable when the stage
    re-runs over an already-filled snapshot.
    """
    for item in items:
        covered = item_covered_by(item, degrees)
        if item.n_covered_by != covered:
            yield item.model_copy(update={"n_covered_by": covered})
        else:
            yield item


# --------------------------------------------------------------------------
# Stage entry.
# --------------------------------------------------------------------------


def stage_entry(
    run_id: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    input_stage: str = INPUT_STAGE,
    top_n: int = DEFAULT_TOP_N,
) -> StageManifest:
    """S13: ``<input snapshot>`` -> ``13_coverage`` (map + targets CSV + passthrough).

    Any snapshot with ``cuis`` filled can stand in for 12_difficulty via
    ``input_stage`` (the orchestrator points S13 at 08_concept when S12 has
    not run); ``input_dir``/``output_dir`` exist for tests. A MISSING snapshot
    is a StageError (the S2 convention): reporting zero coverage over a
    directory nobody wrote would read as a measurement. An EMPTY snapshot is
    the opposite case and is reported honestly as an empty map -- a reporter
    has nothing to refuse.
    """
    started = utcnow()
    # run_dir, not stage_dir: stage_dir would mkdir the input snapshot and a
    # missing-input run would silently degrade into the empty-corpus report.
    inp = input_dir if input_dir is not None else store.run_dir(run_id) / input_stage
    if not inp.is_dir():
        raise StageError(f"S13: input snapshot {inp} does not exist -- run {input_stage} first")
    out = output_dir if output_dir is not None else store.stage_dir(run_id, OUTPUT_STAGE)
    out.mkdir(parents=True, exist_ok=True)  # explicit dirs skip stage_dir's mkdir
    # Resume contract (runner docstring): a re-run replaces its own snapshot only.
    for stale in out.glob("part-*.parquet"):
        stale.unlink()

    manifest = StageManifest(
        run_id=run_id,
        stage=OUTPUT_STAGE,
        started_at=started,
        config={
            "input_dir": str(inp),
            "output_dir": str(out),
            "input_stage": input_stage,
            "top_n": top_n,
            "gap_definition": "zero rows of a corpus-observed lang cover the concept",
            "degree_proxy": "n_covered_by = rows whose cuis contain the concept",
        },
        # No THRESHOLDS knob is consumed: the gap predicate is exact absence and
        # top_n is a stage parameter -- recorded in config rather than pretending
        # a threshold was read (answers.py precedent for honest partial blocks).
        thresholds={},
    )

    concepts, tally = collect(store.iter_items(inp))

    map_path = out / MAP_FILENAME
    map_path.write_text(json.dumps(build_coverage_map(concepts, tally), indent=2), encoding="utf-8")

    targets = generation_targets(concepts, tally.langs, top_n)
    targets_path = out / TARGETS_FILENAME
    write_generation_targets(targets, targets_path)

    degrees = {cui: stats.n for cui, stats in concepts.items()}
    store.write_items(_passthrough(store.iter_items(inp), degrees), out)

    manifest.rows_in = store.count_items(inp)
    manifest.rows_out = store.count_items(out)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.notes = {
        "total_items": tally.total_items,
        "items_with_cuis": tally.items_with_cuis,
        "items_without_cuis": tally.total_items - tally.items_with_cuis,
        "distinct_concepts": len(concepts),
        "corpus_langs": sorted(tally.langs),
        "generation_targets": len(targets),
        "coverage_map": str(map_path),
        "generation_targets_csv": str(targets_path),
        "n_covered_by_item_definition": (
            "max concept degree over the row's cuis (self included); 0 without cuis"
        ),
        "mirror_note": (
            "coverage_map.json mirrors via mirror_light's '*.json' pattern; "
            "generation_targets.csv has no '*.csv' pattern there yet"
        ),
    }
    assert manifest.rows_in == manifest.rows_out, "flags-not-deletes: S13 never drops"
    log.info(
        "S13 %s: %d rows, %d concepts, %d generation targets",
        run_id,
        tally.total_items,
        len(concepts),
        len(targets),
    )
    return manifest


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_TOP_N",
    "INPUT_STAGE",
    "MAP_FILENAME",
    "NO_BAND",
    "OUTPUT_STAGE",
    "TARGETS_FILENAME",
    "ConceptStats",
    "CorpusTally",
    "build_coverage_map",
    "collect",
    "concept_payload",
    "generation_targets",
    "item_covered_by",
    "stage_entry",
    "write_generation_targets",
]
