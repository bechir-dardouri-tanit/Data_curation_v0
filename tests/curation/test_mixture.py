"""S15 mixture-assembly tests: the tiny 8-row fixture through run_mixture
(task contract), the machinery edges (expansion, spec, artifacts, failures),
and every shipped recipe in configs/curation/mixtures executed for real."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, Flags, StageError
from medrl.curation.stages.mixture import run_mixture

RECIPE_DIR = Path(__file__).resolve().parents[2] / "configs" / "curation" / "mixtures"

# --------------------------------------------------------------------------
# The survivor predicate in test SQLs: the same 17-flag fail-closed predicate
# the shipped recipes define as a macro, kept in one helper here.
# --------------------------------------------------------------------------


def _survivor_macro_sql() -> str:
    return """
CREATE OR REPLACE MACRO survivor(
    empty, length, truncated, repetition, lang, encoding, refusal,
    dup_exact, dup_minhash, dup_semantic, dup_concept,
    contam_ngram, contam_semantic, contam_concept,
    answer_wrong, answer_right_reasoning_contradicts, kg_contradicted
) AS (
    NOT empty AND NOT length AND NOT truncated AND NOT repetition AND NOT lang
    AND NOT encoding AND NOT refusal
    AND NOT dup_exact AND NOT dup_minhash AND NOT dup_semantic AND NOT dup_concept
    AND NOT contam_ngram AND NOT contam_semantic AND NOT contam_concept
    AND NOT answer_wrong AND NOT answer_right_reasoning_contradicts
    AND NOT kg_contradicted
);
"""


def _surv() -> str:
    return """survivor(
            flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang,
            flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash,
            flags_dup_semantic, flags_dup_concept, flags_contam_ngram,
            flags_contam_semantic, flags_contam_concept, flags_answer_wrong,
            flags_answer_right_reasoning_contradicts, flags_kg_contradicted)"""


# A phase-1-style weighted sample over three sources: weights 50/30/20,
# anchored on the smallest surviving source (beta or gamma both have 1
# survivor; the name tiebreak picks beta). Targets: alpha round(.5/.3)=2,
# beta 1, gamma round(.2/.3)=1.
SQL_TINY_WEIGHTED = (
    "SELECT setseed(0.42);\n"
    + _survivor_macro_sql()
    + """
WITH pool(source, weight) AS (VALUES
    ('alpha', CAST(0.50 AS DOUBLE)), ('beta', CAST(0.30 AS DOUBLE)),
    ('gamma', CAST(0.20 AS DOUBLE))),
surv AS (SELECT id, source FROM corpus
         WHERE source IN (SELECT source FROM pool) AND """
    + _surv()
    + """),
counts AS (SELECT source, count(*)::BIGINT AS n FROM surv GROUP BY source),
anchor AS (SELECT c.n AS n_anchor, w.weight AS w_anchor
           FROM counts c JOIN pool w USING (source)
           ORDER BY c.n ASC, c.source ASC LIMIT 1),
targets AS (SELECT w.source,
                   CAST(round(w.weight * a.n_anchor / a.w_anchor) AS BIGINT) AS target
            FROM pool w CROSS JOIN anchor a),
ranked AS (SELECT id, source, row_number() OVER (PARTITION BY source ORDER BY random()) AS rn
           FROM surv),
seq AS (SELECT unnest(generate_series(
           1, CAST(coalesce((SELECT max(target) FROM targets), 0) AS BIGINT))) AS i),
draws AS (SELECT t.source, s.i, c.n FROM targets t
          JOIN counts c USING (source) JOIN seq s ON s.i <= t.target WHERE c.n > 0),
picked AS (SELECT d.source, r.id, d.i AS instance
           FROM draws d JOIN ranked r
             ON r.source = d.source AND r.rn = ((d.i - 1) % d.n) + 1)
SELECT id, source, instance FROM picked ORDER BY random();
"""
)

# Forces replacement expansion: 6 draws cycled over alpha's 3 survivors.
SQL_EXPANSION = (
    "SELECT setseed(0.42);\n"
    + _survivor_macro_sql()
    + """
WITH surv AS (SELECT id, source FROM corpus WHERE source = 'alpha' AND """
    + _surv()
    + """),
ranked AS (SELECT id, source, row_number() OVER (ORDER BY random()) AS rn FROM surv),
seq AS (SELECT unnest(generate_series(1, 6)) AS i)
SELECT r.id, r.source, s.i AS instance
FROM seq s JOIN ranked r ON r.rn = ((s.i - 1) % (SELECT count(*) FROM surv)) + 1
ORDER BY s.i;
"""
)


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------


def _item(item_id: str, source: str, **kw: Any) -> CorpusItem:
    kw.setdefault("messages", [{"role": "user", "content": f"clinical question {item_id}"}])
    return CorpusItem(id=item_id, source=source, **kw)


def _tiny_corpus(tmp_path: Path) -> Path:
    """8 items across 3 sources, mixed flags -> 5 clean survivors.

    alpha: 4 rows (alpha:3 flagged dup), beta: 2 (beta:0 flagged contam),
    gamma: 2 (gamma:1 flagged refusal).
    """
    inp = tmp_path / "09_answers"
    inp.mkdir(parents=True)
    items = [
        _item("alpha:0", "alpha"),
        _item("alpha:1", "alpha"),
        _item("alpha:2", "alpha"),
        _item("alpha:3", "alpha", flags=Flags(f_dup_exact=True)),
        _item("beta:0", "beta", flags=Flags(f_contam_ngram=True)),
        _item("beta:1", "beta"),
        _item("gamma:0", "gamma"),
        _item("gamma:1", "gamma", flags=Flags(f_refusal=True)),
    ]
    store.write_items(items, inp)
    return inp


def _recipe_corpus(tmp_path: Path) -> Path:
    """Rows under the real source ids (incl. Pool R/P and band values) so the
    shipped recipes execute against meaningful data."""
    inp = tmp_path / "12_difficulty"
    inp.mkdir(parents=True)
    items: list[CorpusItem] = []
    for i in range(6):
        kw: dict[str, Any] = {}
        if i == 1:
            kw["difficulty_band"] = "rl"
        if i == 0:
            kw["meta"] = {"pair": "grp-1"}
        if i == 5:
            kw["flags"] = Flags(f_dup_exact=True)
        items.append(_item(f"ii_medical_reasoning_sft:r{i}", "ii_medical_reasoning_sft", **kw))
    for i in range(4):
        items.append(_item(f"finemed_sft:f{i}", "finemed_sft"))
    for i in range(3):
        items.append(_item(f"chatdoctor_healthcaremagic:c{i}", "chatdoctor_healthcaremagic"))
    items.append(_item("generalthought_biology:g0", "generalthought_biology"))
    items.append(
        _item("generalthought_biology:g1", "generalthought_biology", flags=Flags(f_empty=True))
    )
    for i in range(3):
        items.append(
            _item(f"medical_r1_distill:d{i}", "medical_r1_distill", difficulty_band="rl")
        )
    for i in range(2):
        items.append(_item(f"m23k_tokenized:m{i}", "m23k_tokenized", difficulty_band="sft1"))
    for i in range(2):
        items.append(_item(f"medreason:e{i}", "medreason", difficulty_band="rl"))
    items.append(_item("huatuo_o1_reasoning:h0", "huatuo_o1_reasoning"))
    for i in range(2):
        items.append(_item(f"finemed_dpo:p{i}", "finemed_dpo"))
    items.append(_item("ii_medical_rl:l0", "ii_medical_rl"))
    items.append(_item("chatdoctor_rl:k0", "chatdoctor_rl"))
    store.write_items(items, inp)
    return inp


def _write_recipe(tmp_path: Path, name: str, sql: str) -> Path:
    d = tmp_path / "mixtures"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.sql"
    p.write_text(sql, encoding="utf-8")
    return d


def _run(
    tmp_path: Path, sql_dir: Path, name: str, inp: Path, **kw: Any
) -> tuple[Any, Any, Any, Path]:
    kw.setdefault("input_dir", inp)
    kw.setdefault("output_dir", tmp_path / f"out_{name}")
    kw.setdefault("experiments_dir", tmp_path / "experiments")
    m = run_mixture("run-t", sql_dir, name, **kw)
    ids_path = Path(kw["experiments_dir"]) / f"15_mixture_{name}.ids.jsonl"
    return m, kw["output_dir"], kw["experiments_dir"], ids_path


def _jsonl_ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# The task-contract test: tiny corpus, test phase SQL weighted by source.
# --------------------------------------------------------------------------


def test_weighted_survivors_selected_and_flagged_never_appear(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    sql_dir = _write_recipe(tmp_path, "phase_t", SQL_TINY_WEIGHTED)
    m, out, _exp, ids_path = _run(tmp_path, sql_dir, "phase_t", inp)

    assert m.stage == "15_mixture_phase_t"
    assert m.rows_in == 8
    assert m.rows_out == 4
    assert m.input_sha256 and m.output_sha256

    selected = _jsonl_ids(ids_path)
    lines = [json.loads(line) for line in ids_path.read_text().splitlines()]
    assert Counter(rec["source"] for rec in lines) == {"alpha": 2, "beta": 1, "gamma": 1}
    for flagged in ("alpha:3", "beta:0", "gamma:1"):
        assert flagged not in selected
    assert set(selected) <= {"alpha:0", "alpha:1", "alpha:2", "beta:1", "gamma:0"}

    # the snapshot is the unique-id set; the JSONL carries the instances
    snap_ids = {it.id for it in store.iter_items(out)}
    assert snap_ids == set(selected)
    assert [rec["position"] for rec in lines] == list(range(len(lines)))
    assert all("instance" in rec and rec["source"] for rec in lines)


def test_manifest_notes_carry_recheck_hook_and_tallies(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    sql_dir = _write_recipe(tmp_path, "phase_t", SQL_TINY_WEIGHTED)
    m, _out, _exp, _ids = _run(tmp_path, sql_dir, "phase_t", inp)

    assert m.notes["recheck_required"] == [
        "04_decontam_ngram",
        "06_decontam_sem",
        "08_concept",
    ]
    assert m.notes["instances_selected"] == 4
    assert m.notes["unique_rows"] == 4
    assert m.notes["rows_out_is_unique_rows"] is True
    assert m.notes["sampling_with_replacement"] is False
    assert m.notes["per_source_input"] == {"alpha": 4, "beta": 2, "gamma": 2}
    assert m.notes["per_source_survivors"] == {"alpha": 3, "beta": 1, "gamma": 1}
    assert m.notes["per_source_instances"] == {"alpha": 2, "beta": 1, "gamma": 1}
    assert m.flag_rates == {}  # S15 sets no flags
    assert m.thresholds == {}  # mixture constants live in the recipe, not thresholds


def test_replacement_expansion_records_instances_and_unique_set(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    sql_dir = _write_recipe(tmp_path, "phase_x", SQL_EXPANSION)
    m, out, _exp, ids_path = _run(tmp_path, sql_dir, "phase_x", inp)

    ids = _jsonl_ids(ids_path)
    assert len(ids) == 6
    assert Counter(ids) == {"alpha:0": 2, "alpha:1": 2, "alpha:2": 2}
    assert m.notes["sampling_with_replacement"] is True
    assert m.notes["instances_selected"] == 6
    assert m.notes["unique_rows"] == 3
    assert store.count_items(out) == 3  # one physical row per id


# --------------------------------------------------------------------------
# Artifacts.
# --------------------------------------------------------------------------


def test_spec_sql_is_exact_executed_sql_with_counts_header(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    sql_dir = _write_recipe(tmp_path, "phase_t", SQL_TINY_WEIGHTED)
    m, _out, exp, _ids = _run(tmp_path, sql_dir, "phase_t", inp)

    spec = (exp / "mixture_spec.sql").read_text()
    assert SQL_TINY_WEIGHTED in spec  # the exact executed SQL, verbatim
    assert m.config["sql_text"] == SQL_TINY_WEIGHTED
    assert "input / survivors / selected instances / unique selected" in spec
    assert "input_sha256=" + m.input_sha256 in spec
    for src in ("alpha", "beta", "gamma"):
        assert src in spec
    # alpha: 4 in / 3 survivors / 2 instances / 2 unique
    assert f"--   {'alpha':<32} {4:>8} / {3:>8} / {2:>8} / {2:>8}" in spec


def test_recipe_without_survivor_macro_omits_survivor_counts(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    plain = "SELECT id, source FROM corpus WHERE NOT flags_dup_exact ORDER BY id;"
    sql_dir = _write_recipe(tmp_path, "plain", plain)
    m, out, exp, ids_path = _run(tmp_path, sql_dir, "plain", inp)

    assert m.notes["per_source_survivors"] == {}
    assert "survivor counts unavailable" in (exp / "mixture_spec.sql").read_text()
    assert set(_jsonl_ids(ids_path)) == {it.id for it in store.iter_items(out)}
    assert "alpha:3" not in _jsonl_ids(ids_path)


# --------------------------------------------------------------------------
# Failure paths: fail loud, never seal a wrong assembly.
# --------------------------------------------------------------------------


def test_missing_recipe_raises(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="does not exist"):
        run_mixture(
            "run-t", tmp_path, "nope", input_dir=_tiny_corpus(tmp_path),
            output_dir=tmp_path / "o", experiments_dir=tmp_path / "e",
        )


def test_out_name_is_never_a_path(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="plain phase name"):
        run_mixture(
            "run-t", tmp_path, "../evil", input_dir=_tiny_corpus(tmp_path),
            output_dir=tmp_path / "o", experiments_dir=tmp_path / "e",
        )


def test_empty_selection_refused(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    sql_dir = _write_recipe(
        tmp_path, "empty", "SELECT id, source FROM corpus WHERE flags_empty AND flags_dup_exact;"
    )
    with pytest.raises(StageError, match="0 rows"):
        _run(tmp_path, sql_dir, "empty", inp)


def test_final_select_without_id_column_refused(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    sql_dir = _write_recipe(tmp_path, "noid", "SELECT source FROM corpus WHERE flags_dup_exact;")
    with pytest.raises(StageError, match="'id'"):
        _run(tmp_path, sql_dir, "noid", inp)


def test_missing_input_snapshot_raises(tmp_path: Path) -> None:
    sql_dir = _write_recipe(tmp_path, "phase_t", SQL_TINY_WEIGHTED)
    with pytest.raises(StageError, match="does not exist"):
        run_mixture(
            "run-t", sql_dir, "phase_t", input_dir=tmp_path / "missing",
            output_dir=tmp_path / "o", experiments_dir=tmp_path / "e",
        )


def test_input_without_part_files_raises(tmp_path: Path) -> None:
    inp = tmp_path / "09_answers"
    inp.mkdir()
    sql_dir = _write_recipe(tmp_path, "phase_t", SQL_TINY_WEIGHTED)
    with pytest.raises(StageError, match=r"no part-.*parquet"):
        _run(tmp_path, sql_dir, "phase_t", inp)


def test_output_must_differ_from_input(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    sql_dir = _write_recipe(tmp_path, "phase_t", SQL_TINY_WEIGHTED)
    with pytest.raises(StageError, match="must differ"):
        run_mixture(
            "run-t", sql_dir, "phase_t", input_dir=inp, output_dir=inp,
            experiments_dir=tmp_path / "e",
        )


def test_rerun_replaces_snapshot_instead_of_appending(tmp_path: Path) -> None:
    inp = _tiny_corpus(tmp_path)
    out = tmp_path / "out"
    exp = tmp_path / "e"
    sql_dir = _write_recipe(tmp_path, "all", "SELECT id, source FROM corpus WHERE NOT flags_dup_exact ORDER BY id;")
    run_mixture("run-t", sql_dir, "all", input_dir=inp, output_dir=out, experiments_dir=exp)
    assert store.count_items(out) == 7
    narrower = _write_recipe(
        tmp_path, "alpha_only", "SELECT id, source FROM corpus WHERE source = 'alpha' AND NOT flags_dup_exact ORDER BY id;"
    )
    m2 = run_mixture(
        "run-t", narrower, "alpha_only", input_dir=inp, output_dir=out, experiments_dir=exp
    )
    assert store.count_items(out) == 3  # replaced, not 7 + 3
    assert m2.rows_out == 3


def test_duckdb_below_session_seed_era_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(duckdb, "__version__", "0.9.2")
    sql_dir = _write_recipe(tmp_path, "phase_t", SQL_TINY_WEIGHTED)
    with pytest.raises(StageError, match="duckdb>="):
        _run(tmp_path, sql_dir, "phase_t", _tiny_corpus(tmp_path))


# --------------------------------------------------------------------------
# Default destinations: scratch snapshot + experiments-side light records.
# --------------------------------------------------------------------------


def test_default_paths_resolve_newest_snapshot_and_experiments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "SCRATCH_ROOT", tmp_path / "scratch")
    monkeypatch.setattr(store, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    older = store.stage_dir("run-x", "02_structural")
    older.mkdir(parents=True, exist_ok=True)  # empty: must be skipped
    inp = store.stage_dir("run-x", "09_answers")
    items = [
        _item("alpha:0", "alpha"),
        _item("alpha:1", "alpha"),
        _item("beta:0", "beta", flags=Flags(f_dup_minhash=True)),
        _item("beta:1", "beta"),
    ]
    store.write_items(items, inp)

    sql_dir = _write_recipe(
        tmp_path, "phase_t", "SELECT id, source FROM corpus WHERE NOT flags_dup_minhash ORDER BY id;"
    )
    m = run_mixture("run-x", sql_dir, "phase_t")

    assert m.config["input_stage"] == "09_answers"
    assert m.rows_in == 4 and m.rows_out == 3
    assert (tmp_path / "experiments" / "run-x" / "mixture_spec.sql").is_file()
    assert (
        tmp_path / "experiments" / "run-x" / "15_mixture_phase_t.ids.jsonl"
    ).is_file()
    snap = tmp_path / "scratch" / "run-x" / "15_mixture_phase_t"
    assert {it.id for it in store.iter_items(snap)} == {"alpha:0", "alpha:1", "beta:1"}


# --------------------------------------------------------------------------
# The shipped recipes, executed for real against a meaningful fixture.
# --------------------------------------------------------------------------


def test_phase1_recipe_anchor_rule(tmp_path: Path) -> None:
    inp = _recipe_corpus(tmp_path)
    m, _out, _exp, ids_path = _run(tmp_path, RECIPE_DIR, "phase1", inp)

    # anchor = generalthought_biology (1 survivor, w=0.05):
    # targets ii=round(.55/.05)=11, fm=5, cd=3, gb=1 -> 20 instances
    assert m.notes["per_source_instances"] == {
        "ii_medical_reasoning_sft": 11,
        "finemed_sft": 5,
        "chatdoctor_healthcaremagic": 3,
        "generalthought_biology": 1,
    }
    assert m.notes["sampling_with_replacement"] is True
    assert m.rows_out == 13  # 5 + 4 + 3 + 1 unique survivors
    ids = _jsonl_ids(ids_path)
    assert len(ids) == 20
    assert "ii_medical_reasoning_sft:r5" not in ids
    assert "generalthought_biology:g1" not in ids
    # the anchor contributes all of its survivors, undiluted
    assert Counter(ids)["generalthought_biology:g0"] == 1


def test_phase2_recipe_band_rl_gate(tmp_path: Path) -> None:
    inp = _recipe_corpus(tmp_path)
    m, out, _exp, ids_path = _run(tmp_path, RECIPE_DIR, "phase2", inp)

    # R band-rl survivors: 3 r1 + 2 medreason = 5 (m23k=sft1, huo=NULL excluded);
    # B budget round(5*0.25)=1 -> the single band-rl Pool-B row (ii:r1)
    assert set(_jsonl_ids(ids_path)) == {
        "medical_r1_distill:d0",
        "medical_r1_distill:d1",
        "medical_r1_distill:d2",
        "medreason:e0",
        "medreason:e1",
        "ii_medical_reasoning_sft:r1",
    }
    assert all(it.difficulty_band == "rl" for it in store.iter_items(out))
    assert m.notes["unique_rows"] == 6


def test_phase3_recipe_60_40_blend(tmp_path: Path) -> None:
    inp = _recipe_corpus(tmp_path)
    m, _out, _exp, ids_path = _run(tmp_path, RECIPE_DIR, "phase3", inp)

    # R survivors = 8 (no band filter) -> B budget round(8*1.5)=12 split by
    # weights: ii 7, fm 3, cd 2, gb 1 (rounding may shift a row; realized
    # counts are what the manifest publishes). Total = 8 R + 13 B instances.
    assert m.notes["per_source_instances"] == {
        "medical_r1_distill": 3,
        "m23k_tokenized": 2,
        "medreason": 2,
        "huatuo_o1_reasoning": 1,
        "ii_medical_reasoning_sft": 7,
        "finemed_sft": 3,
        "chatdoctor_healthcaremagic": 2,
        "generalthought_biology": 1,
    }
    ids = _jsonl_ids(ids_path)
    assert len(ids) == 21
    assert m.notes["unique_rows"] == 19  # B unique 11 + R 8
    assert "ii_medical_reasoning_sft:r5" not in ids
    assert "generalthought_biology:g1" not in ids
    r_ids = [i for i in ids if i.split(":")[0] in
             {"medical_r1_distill", "m23k_tokenized", "medreason", "huatuo_o1_reasoning"}]
    assert len(r_ids) == 8  # the 40% side is all R survivors


def test_phase4_recipe_pairs(tmp_path: Path) -> None:
    inp = _recipe_corpus(tmp_path)
    m, _out, _exp, ids_path = _run(tmp_path, RECIPE_DIR, "phase4", inp)

    assert set(_jsonl_ids(ids_path)) == {
        "finemed_dpo:p0",
        "finemed_dpo:p1",
        "ii_medical_rl:l0",
        "chatdoctor_rl:k0",
        "ii_medical_reasoning_sft:r0",  # pulled in by the meta.pair marker
    }
    assert m.notes["sampling_with_replacement"] is False
    assert m.notes["unique_rows"] == 5


def test_phase5_recipe_fixed_counts_and_seed(tmp_path: Path) -> None:
    inp = _recipe_corpus(tmp_path)
    m1, _out1, _exp1, ids1 = _run(tmp_path, RECIPE_DIR, "phase5", inp, output_dir=tmp_path / "o5a")
    _m2, _out2, _exp2, ids2 = _run(tmp_path, RECIPE_DIR, "phase5", inp, output_dir=tmp_path / "o5b")

    # LIMIT caps only: the fixture is far below 150k/50k, so everything clean
    # is taken -- ii 5 survivors + R 8 survivors
    assert m1.notes["per_source_unique"] == {
        "ii_medical_reasoning_sft": 5,
        "medical_r1_distill": 3,
        "m23k_tokenized": 2,
        "medreason": 2,
        "huatuo_o1_reasoning": 1,
    }
    assert m1.notes["unique_rows"] == 13
    assert "ii_medical_reasoning_sft:r5" not in _jsonl_ids(ids1)
    # setseed(0.42) pins the draw for a given duckdb build + thread count
    assert _jsonl_ids(ids1) == _jsonl_ids(ids2)


# --------------------------------------------------------------------------
# Determinism at parallel-scan scale (the tiny fixture above can never fan out;
# the 13-row determinism guarantee did not transfer to real corpora).
# --------------------------------------------------------------------------


def test_execute_spec_seeded_draw_is_deterministic_at_scale(tmp_path: Path) -> None:
    """Regression: setseed() reproducibility is per-connection AND per-scan.
    With default threads, the same seeded recipe over the same 2M-row parquet
    produced a different selection on every fresh connection (6/6 distinct
    measured on duckdb 1.5.5) -- the phase2/phase3 training sets silently
    changed between runs. _execute_spec pins threads=1; at a row count that
    actually exercises scan fan-out, runs must be identical."""
    from medrl.curation.stages import mixture as mixture_mod

    p = tmp_path / "big.parquet"
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT 'id' || i AS id, 'src' AS source FROM range(2000000) tbl(i)) "
        f"TO '{p}' (FORMAT PARQUET)"
    )
    con.close()

    script = "SELECT setseed(0.42);\nSELECT id FROM corpus ORDER BY random() LIMIT 10;"
    draws = {
        tuple(row["id"] for row in mixture_mod._execute_spec([p], script)[1])
        for _ in range(3)
    }
    assert len(draws) == 1, f"seeded draw differed across runs: {len(draws)} distinct results"


def test_runner_mixture_default_saves_every_phase_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the runner's default 15_mixture ran phases 1-5 but returned
    only the last manifest to _wrap -- 4 of 5 mixture outputs had no audit
    record (no config/row counts/hashes), and recipe drift in them was
    undetectable after the fact."""
    from medrl.curation import runner as runner_mod

    inp = _recipe_corpus(tmp_path)
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(store, "EXPERIMENTS_ROOT", experiments)
    monkeypatch.setattr(store, "SCRATCH_ROOT", tmp_path / "scratch")

    manifest = runner_mod.run_mixture_default("run-t", sql_dir=RECIPE_DIR, input_dir=inp)

    # the returned manifest is phase5's, still unsealed (that is _wrap's job)
    assert manifest.stage == "15_mixture_phase5"
    assert manifest.finished_at is None
    # phases 1-4 each have their own sealed audit record
    for phase in ("phase1", "phase2", "phase3", "phase4"):
        p = experiments / "run-t" / f"15_mixture_{phase}.manifest.json"
        assert p.is_file(), f"missing audit record for {phase}"
        blob = json.loads(p.read_text())
        assert blob["rows_out"] > 0
        assert blob["finished_at"] is not None
    # phase5's file lands when _wrap seals + saves the returned manifest
    assert not (experiments / "run-t" / "15_mixture_phase5.manifest.json").exists()
