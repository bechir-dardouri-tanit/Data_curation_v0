"""S13 coverage-map tests: exact aggregate values on a synthetic 2-language
corpus with a known concept distribution, gap detection and top-N truncation,
band aggregation, the empty-corpus edge, and the CSV contract (header + row
shape) -- all through the real parquet store."""

from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any

import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, DifficultyBand, StageError
from medrl.curation.stages import coverage


def _item(
    item_id: str,
    *,
    source: str = "src_a",
    lang: str = "en",
    band: DifficultyBand | None = "rl",
    cuis: list[str] | None = None,
) -> CorpusItem:
    """A minimal row; the stage reads only source/lang/band/cuis (plus flags)."""
    return CorpusItem(
        id=item_id,
        source=source,
        lang=lang,
        messages=[{"role": "user", "content": f"question for {item_id}"}],
        difficulty_band=band,
        cuis=cuis or [],
    )


def _run_stage(
    tmp_path: Any,
    items: list[CorpusItem],
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Write the input snapshot, run S13, return (manifest, input dir, output dir)."""
    inp = tmp_path / "12_difficulty"
    out = tmp_path / "13_coverage"
    inp.mkdir(parents=True, exist_ok=True)  # re-runs reuse the snapshot dir
    if items:
        store.write_items(items, inp)
    manifest = coverage.stage_entry("test-run", input_dir=inp, output_dir=out, **kwargs)
    return manifest, inp, out


def _read_map(out: Any) -> dict[str, Any]:
    return json.loads((out / coverage.MAP_FILENAME).read_text())


def _read_targets(out: Any) -> tuple[list[str], list[list[str]]]:
    with (out / coverage.TARGETS_FILENAME).open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


# --------------------------------------------------------------------------
# Exact map values on a known 2-language / 2-source corpus.
# --------------------------------------------------------------------------


def _known_corpus() -> list[CorpusItem]:
    """Concept distribution, by construction:

    C1 <- a:1(en/rl), a:2(en/rl), b:1(fr/sft1)      degree 3
    C2 <- a:1(en/rl), b:3(fr/downsample)            degree 2
    C3 <- a:3(en/None band)                         degree 1
    C4 <- b:3(fr/downsample)                        degree 1
    b:2 carries no cuis.
    """
    return [
        _item("a:1", source="src_a", lang="en", band="rl", cuis=["C1", "C2"]),
        _item("a:2", source="src_a", lang="en", band="rl", cuis=["C1"]),
        _item("a:3", source="src_a", lang="en", band=None, cuis=["C3"]),
        _item("b:1", source="src_b", lang="fr", band="sft1", cuis=["C1"]),
        _item("b:2", source="src_b", lang="fr", band="hold", cuis=[]),
        _item("b:3", source="src_b", lang="fr", band="downsample", cuis=["C2", "C4"]),
    ]


def test_exact_map_values(tmp_path):
    manifest, _, out = _run_stage(tmp_path, _known_corpus())
    assert manifest.rows_in == manifest.rows_out == 6

    m = _read_map(out)
    assert m["total_items"] == 6
    assert m["items_with_cuis"] == 5

    assert [c["id"] for c in m["concepts"]] == ["C1", "C2", "C3", "C4"]  # degree desc
    c1, c2, c3, c4 = m["concepts"]
    assert c1["n_covered_by"] == 3
    assert c1["sources"] == {"src_a": 2, "src_b": 1}
    assert c1["langs"] == {"en": 2, "fr": 1}
    assert c1["bands"] == {"rl": 2, "sft1": 1}
    assert c2["n_covered_by"] == 2
    assert c2["bands"] == {"downsample": 1, "rl": 1}
    assert c3 == {
        "id": "C3",
        "n_covered_by": 1,
        "sources": {"src_a": 1},
        "langs": {"en": 1},
        "bands": {"none": 1},
    }
    assert c4["langs"] == {"fr": 1}

    assert m["coverage_by_source"] == {
        "src_a": {"items": 3, "items_with_cuis": 3, "frac": 1.0},
        "src_b": {"items": 3, "items_with_cuis": 2, "frac": pytest.approx(2 / 3)},
    }
    assert m["by_band"] == {
        "downsample": {"items": 1, "items_with_cuis": 1},
        "hold": {"items": 1, "items_with_cuis": 0},  # the cuis-less row
        "none": {"items": 1, "items_with_cuis": 1},
        "rl": {"items": 2, "items_with_cuis": 2},
        "sft1": {"items": 1, "items_with_cuis": 1},
    }


def test_passthrough_fills_n_covered_by(tmp_path):
    _, _, out = _run_stage(tmp_path, _known_corpus())
    rows = {it.id: it for it in store.iter_items(out)}
    assert set(rows) == {"a:1", "a:2", "a:3", "b:1", "b:2", "b:3"}
    # max concept degree over the row's cuis; 0 without cuis
    assert rows["a:1"].n_covered_by == 3  # max(C1=3, C2=2)
    assert rows["a:2"].n_covered_by == 3
    assert rows["a:3"].n_covered_by == 1
    assert rows["b:1"].n_covered_by == 3
    assert rows["b:2"].n_covered_by == 0  # no cuis
    assert rows["b:3"].n_covered_by == 2  # max(C2=2, C4=1)


def test_passthrough_preserves_everything_else(tmp_path):
    items = _known_corpus()
    items[0].flags = items[0].flags.model_copy(update={"f_dup_minhash": True})
    items[0].dup_of = "a:0"
    _, inp, out = _run_stage(tmp_path, items)
    before = {it.id: it for it in store.iter_items(inp)}
    after = {it.id: it for it in store.iter_items(out)}
    assert set(before) == set(after)
    for iid, item in before.items():
        expected = item.model_dump() | {"n_covered_by": after[iid].n_covered_by}
        assert after[iid].model_dump() == expected
    assert after["a:1"].flags.f_dup_minhash is True
    assert after["a:1"].dup_of == "a:0"


# --------------------------------------------------------------------------
# Gap detection + the CSV contract.
# --------------------------------------------------------------------------


def _gap_corpus() -> list[CorpusItem]:
    """C_hot: 4 en rows, no fr row -> the top gap target.
    C_bilingual: covered in both langs -> never a target.
    C_warm: 3 en rows, no fr row -> a gap target below C_hot."""
    items = [_item(f"en:{i}", source="s", lang="en", band="rl", cuis=["C_hot"]) for i in range(4)]
    items += [_item("en:b", source="s", lang="en", band="rl", cuis=["C_bilingual", "C_warm"])]
    items += [
        _item(f"warm:{i}", source="s", lang="en", band="rl", cuis=["C_warm"]) for i in range(2)
    ]
    items += [_item("fr:1", source="s", lang="fr", band="sft1", cuis=["C_bilingual"])]
    return items


def test_gap_detection_concept_present_in_en_only(tmp_path):
    _, _, out = _run_stage(tmp_path, _gap_corpus())
    header, rows = _read_targets(out)
    assert header == list(coverage.CSV_COLUMNS)
    assert [r[0] for r in rows] == ["C_hot", "C_warm"]  # degree order kept
    hot = dict(zip(header, rows[0], strict=True))
    assert hot["total_n"] == "4"
    assert hot["lang_gaps"] == "fr"  # absent from every fr row, present in en
    pool = json.loads(hot["suggested_pool"])
    assert pool == {"rl": 4}  # the band distribution of the covering items


def test_bilingual_concept_is_never_a_target(tmp_path):
    concepts, tally = coverage.collect(iter(_gap_corpus()))
    targets = coverage.generation_targets(concepts, tally.langs)
    assert "C_bilingual" not in {r["concept_id"] for r in targets}


def test_top_n_truncates_candidates(tmp_path):
    _, _, out = _run_stage(tmp_path, _gap_corpus(), top_n=1)
    _, rows = _read_targets(out)
    assert [r[0] for r in rows] == ["C_hot"]  # C_warm (degree 3) fell outside top-1


def test_multiple_lang_gaps_sorted_joined(tmp_path):
    items = [
        _item("en:1", lang="en", cuis=["C_only_en"]),
        _item("fr:1", lang="fr", cuis=["C_shared"]),
        _item("de:1", lang="de", cuis=["C_shared", "C_other"]),
    ]
    _, _, out = _run_stage(tmp_path, items)
    _, rows = _read_targets(out)
    by_id = {r[0]: r for r in rows}
    assert by_id["C_only_en"][2] == "de;fr"  # sorted, ';'-joined
    # C_shared is covered by the fr and de rows only -> the en column is the gap
    assert by_id["C_shared"][2] == "en"
    assert by_id["C_other"][2] == "en;fr"


def test_csv_row_shape_and_pool_json(tmp_path):
    _, _, out = _run_stage(tmp_path, _gap_corpus())
    header, rows = _read_targets(out)
    for row in rows:
        assert len(row) == len(header)
        assert row[1].isdigit()
        pool = json.loads(row[3])
        assert isinstance(pool, dict)
        assert all(isinstance(v, int) for v in pool.values())


# --------------------------------------------------------------------------
# Band aggregation (unit level, no IO).
# --------------------------------------------------------------------------


def test_band_aggregation_counts_unbanded_rows():
    items = [
        _item("a:1", band="rl", cuis=["C1"]),
        _item("a:2", band=None, cuis=["C1"]),
        _item("a:3", band="hold"),
    ]
    concepts, tally = coverage.collect(iter(items))
    assert concepts["C1"].bands == Counter({"rl": 1, "none": 1})
    assert tally.items_per_band == Counter({"rl": 1, "none": 1, "hold": 1})
    assert tally.with_cuis_per_band["hold"] == 0


# --------------------------------------------------------------------------
# Edge cases + manifest contract.
# --------------------------------------------------------------------------


def test_empty_corpus_produces_empty_map_and_header_only_csv(tmp_path):
    manifest, _, out = _run_stage(tmp_path, [])
    assert manifest.rows_in == manifest.rows_out == 0
    m = _read_map(out)
    assert m == {
        "total_items": 0,
        "items_with_cuis": 0,
        "concepts": [],
        "coverage_by_source": {},
        "by_band": {},
    }
    header, rows = _read_targets(out)
    assert header == list(coverage.CSV_COLUMNS)
    assert rows == []
    assert list(out.glob("part-*.parquet")) == []


def test_missing_input_snapshot_fails_loudly(tmp_path):
    with pytest.raises(StageError, match="12_difficulty"):
        coverage.stage_entry(
            "x", input_dir=tmp_path / "12_difficulty", output_dir=tmp_path / "13_coverage"
        )


def test_missing_input_is_not_silently_created(tmp_path):
    # store.stage_dir mkdirs its argument; S13 must not use it for the input,
    # or a never-run upstream stage would silently produce an empty report.
    with pytest.raises(StageError, match="does not exist"):
        coverage.stage_entry("never-run", output_dir=tmp_path / "13_coverage")


def test_rerun_replaces_its_own_snapshot(tmp_path):
    inp = tmp_path / "12_difficulty"
    out = tmp_path / "13_coverage"
    inp.mkdir(parents=True)
    store.write_items(_known_corpus(), inp)
    coverage.stage_entry("test-run", input_dir=inp, output_dir=out)
    coverage.stage_entry("test-run", input_dir=inp, output_dir=out)
    # write_items appends across calls; only the stale-part cleanup stops a
    # re-run from doubling the snapshot.
    assert store.count_items(out) == len(_known_corpus())


def test_manifest_contract(tmp_path):
    manifest, _, out = _run_stage(tmp_path, _known_corpus())
    assert manifest.stage == coverage.OUTPUT_STAGE
    assert manifest.config["top_n"] == coverage.DEFAULT_TOP_N
    assert manifest.thresholds == {}  # no THRESHOLDS knob consumed; see module docstring
    assert manifest.flag_rates == {}  # a reporter sets no flags
    assert manifest.input_sha256 and manifest.output_sha256
    notes = manifest.notes
    assert notes["distinct_concepts"] == 4
    # C3 (en only) and C4 (fr only) are single-lang concepts inside the top-200
    # candidates, so they are targets; C1/C2 cover both langs and are not.
    assert notes["generation_targets"] == 2
    assert notes["corpus_langs"] == ["en", "fr"]
    assert notes["n_covered_by_item_definition"].startswith("max concept degree")
    assert (out / coverage.MAP_FILENAME).exists()
    assert (out / coverage.TARGETS_FILENAME).exists()
