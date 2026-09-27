"""Tests for S8 concept dedup + cross-lingual decontamination (stages/concept.py).

Each case runs through the full store round-trip via ``stage_entry`` with
tmp_path dirs and injected synthetic concept fingerprints (S7 linking is
licence-gated and unbuilt, so the eval side is exactly the explicit parameter
the stage contract names). The five contract behaviours from the plan get
dedicated cases: cross-lingual pair dup-flagged, same-CUI-set different-answer
NOT dup'd, the inclusive 0.85 Jaccard boundary, contam requiring BOTH the
question-Jaccard bound AND the answer_cui match, and determinism (lower id
canonical).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, Flags, StageError, StageManifest
from medrl.curation.stages import concept
from medrl.curation.thresholds import THRESHOLDS

# A translation pair by construction: no shared word, so S3's 13-grams, S4's
# n-grams and S6's cosine all see two unrelated rows. Only the linked concept
# identity agrees.
EN_Q = "How should doctors manage dangerously elevated blood sugar?"
FR_Q = "Comment faut-il prendre en charge une glycemie tres elevee ?"


def _item(
    id_: str,
    *,
    cuis: list[str] | None = None,
    answer_cui: str | None = None,
    lang: str = "en",
    question: str = "Question?",
    source: str | None = None,
    **kw: Any,
) -> CorpusItem:
    return CorpusItem(
        id=id_,
        source=source if source is not None else id_.split(":", 1)[0],
        lang=lang,
        cuis=list(cuis or []),
        answer_cui=answer_cui,
        messages=[{"role": "user", "content": question}],
        **kw,
    )


def _run(
    tmp_path: Path,
    items: list[CorpusItem],
    eval_records: Any = None,
    **kw: Any,
) -> tuple[StageManifest, dict[str, CorpusItem]]:
    inp = tmp_path / "06_decontam_sem"
    inp.mkdir(parents=True, exist_ok=True)
    store.write_items(items, inp)
    manifest = concept.stage_entry(
        "test-run",
        eval_items=eval_records,
        input_dir=inp,
        output_dir=tmp_path / "08_concept",
        **kw,
    )
    rows = {it.id: it for it in store.iter_items(tmp_path / "08_concept")}
    return manifest, rows


def _fingerprint(
    directory: Path,
) -> dict[str, tuple[tuple[str, ...], str | None, str | None]]:
    """The state an idempotent re-run must preserve exactly."""
    return {
        it.id: (it.flags.set_names(), it.dup_of, it.contam_benchmark)
        for it in store.iter_items(directory)
    }


# ----------------------------------------------------------------------------------
# Units: the Jaccard primitive and the dup identity
# ----------------------------------------------------------------------------------


def test_concept_jaccard_basics() -> None:
    assert concept.concept_jaccard({"A", "B"}, {"A", "B"}) == 1.0
    assert concept.concept_jaccard({"A", "B"}, {"C", "D"}) == 0.0
    assert concept.concept_jaccard(set(), {"A"}) == 0.0, "0/0 must never read as a match"
    assert concept.concept_jaccard({"A"}, set()) == 0.0
    assert concept.concept_jaccard({"A", "B", "C"}, {"B", "C", "D"}) == pytest.approx(0.5)


def test_dup_key_normalizes_order_and_multiplicity() -> None:
    a = _item("s:1", cuis=["C003", "C001", "C003", "C002"])
    assert concept.dup_key(a) == (("C001", "C002", "C003"), None)
    assert concept.dup_key(_item("s:2")) is None, "unlinked rows carry no identity"


# ----------------------------------------------------------------------------------
# (1) The cross-lingual pair: the catch no earlier stage can make
# ----------------------------------------------------------------------------------


def test_cross_lingual_translation_pair_is_dup_flagged(tmp_path: Path) -> None:
    en = _item("src_a:0002", cuis=["C001", "C002", "C003"], answer_cui="C009", question=EN_Q)
    fr = _item(
        "src_a:0001",
        cuis=["C001", "C002", "C003"],
        answer_cui="C009",
        lang="fr",
        question=FR_Q,
    )
    en_words = {w.strip("?'!,.") for w in EN_Q.lower().split()}
    fr_words = {w.strip("?'!,.") for w in FR_Q.lower().split()}
    assert en_words.isdisjoint(fr_words), "fixture must be a true pair: zero shared surface tokens"
    manifest, rows = _run(tmp_path, [en, fr])

    assert manifest.rows_in == manifest.rows_out == 2, "flags-not-deletes"
    # src_a:0001 (the FR row) holds the lower id, so it is canonical and clean;
    # the EN row is the loser even though the pair was planted EN-first.
    assert rows["src_a:0002"].flags.f_dup_concept is True
    assert rows["src_a:0002"].dup_of == "src_a:0001"
    assert rows["src_a:0001"].flags.f_dup_concept is False, "the canonical row stays clean"
    assert rows["src_a:0001"].dup_of is None
    assert rows["src_a:0001"].lang == "fr" and rows["src_a:0002"].lang == "en"
    assert manifest.notes["concept_dup_groups"] == 1
    assert manifest.notes["concept_dup_rows"] == 1


def test_same_cui_set_different_answer_is_not_dup(tmp_path: Path) -> None:
    a = _item("src_a:1", cuis=["C001", "C002"], answer_cui="C009")
    b = _item("src_a:2", cuis=["C001", "C002"], answer_cui="C010")
    manifest, rows = _run(tmp_path, [a, b])

    assert all(
        not rows[i].flags.f_dup_concept and rows[i].dup_of is None for i in ("src_a:1", "src_a:2")
    )
    assert manifest.notes["concept_dup_groups"] == 0
    assert manifest.flag_rates["f_dup_concept"]["_all"] == 0.0


def test_same_cui_set_unlinked_answers_still_dup(tmp_path: Path) -> None:
    """answer_cui None on both sides is identical under the contract: for Pool B
    rows (no gold answer to link) the exact question-set requirement IS the whole
    identity, and their translation pairs are this stage's primary target. A
    false merge stays auditable and reversible at the mixture-query level."""
    a = _item("src_a:1", cuis=["C001", "C002"], question=EN_Q)
    b = _item("src_a:2", cuis=["C001", "C002"], lang="fr", question=FR_Q)
    _, rows = _run(tmp_path, [a, b])

    assert rows["src_a:2"].flags.f_dup_concept is True
    assert rows["src_a:2"].dup_of == "src_a:1"


def test_dedup_canonical_is_lower_id_regardless_of_input_order(tmp_path: Path) -> None:
    items = [
        _item("src_z:9", cuis=["C001", "C002"], answer_cui="C009"),
        _item("src_m:5", cuis=["C002", "C001"], answer_cui="C009"),
        _item("src_a:1", cuis=["C001", "C001", "C002"], answer_cui="C009"),
    ]
    # Unit level: the pass is a pure function of the row multiset, so even a
    # shuffled sequence (never produced by the store) cannot decide the winner.
    assert concept.dup_pass(items) == {"src_m:5": "src_a:1", "src_z:9": "src_a:1"}

    manifest, rows = _run(tmp_path, items)
    assert rows["src_a:1"].dup_of is None and not rows["src_a:1"].flags.f_dup_concept
    assert rows["src_m:5"].dup_of == "src_a:1"
    assert rows["src_z:9"].dup_of == "src_a:1"
    assert manifest.notes["concept_dup_groups"] == 1


def test_rows_without_cuis_are_never_joined(tmp_path: Path) -> None:
    a = _item("src_a:1")
    b = _item("src_a:2")  # identical emptiness is not identity
    manifest, rows = _run(tmp_path, [a, b])

    assert all(not rows[i].flags.any() for i in ("src_a:1", "src_a:2"))
    assert manifest.notes["rows_without_cuis"] == 2
    assert manifest.notes["concept_dup_groups"] == 0


# ----------------------------------------------------------------------------------
# (2) Contamination: Jaccard >= concept_jaccard AND matching answer_cui, both required
# ----------------------------------------------------------------------------------

_SHARED_17 = frozenset(f"C{i:03d}" for i in range(17))
_EVAL_RECORD = ("usmle", _SHARED_17, "C100")
"""17 shared concepts: a corpus item with exactly these plus 3 own concepts
sits at 17/20 == 0.85, the threshold, exactly (both are the same double)."""


def test_contam_boundary_is_inclusive(tmp_path: Path) -> None:
    at_threshold = [*_SHARED_17, "C900", "C901", "C902"]  # 17/20 == 0.85
    below_threshold = [*_SHARED_17, "C900", "C901", "C902", "C903"]  # 17/21 < 0.85
    assert concept.concept_jaccard(set(at_threshold), set(_SHARED_17)) == THRESHOLDS.concept_jaccard
    assert (
        concept.concept_jaccard(set(below_threshold), set(_SHARED_17)) < THRESHOLDS.concept_jaccard
    )

    items = [
        _item("src:1", cuis=at_threshold, answer_cui="C100"),
        _item("src:2", cuis=below_threshold, answer_cui="C100"),
    ]
    manifest, rows = _run(tmp_path, items, [_EVAL_RECORD])

    assert rows["src:1"].flags.f_contam_concept is True, "J == threshold must flag"
    assert rows["src:1"].contam_benchmark == "usmle"
    assert rows["src:2"].flags.f_contam_concept is False, "just below must not flag"
    assert rows["src:2"].contam_benchmark is None
    assert manifest.notes["per_benchmark_hits"] == {"usmle": 1}


def test_contam_requires_both_jaccard_and_answer_cui(tmp_path: Path) -> None:
    items = [
        # hit: same concepts AND same answer concept
        _item("src:hit", cuis=[*_SHARED_17, "X1", "X2", "X3"], answer_cui="C100"),
        # same question concepts, different asserted answer: not leakage evidence
        _item("src:ans_diff", cuis=list(_SHARED_17), answer_cui="C200"),
        # same answer concept, disjoint question concepts: any row with this gold
        # answer would flag -- useless
        _item("src:ques_diff", cuis=["Z1", "Z2", "Z3"], answer_cui="C100"),
        # unlinked answer on both sides: a match on "both unknown" is no evidence
        _item("src:ans_none", cuis=list(_SHARED_17), answer_cui=None),
    ]
    manifest, rows = _run(tmp_path, items, [_EVAL_RECORD])

    assert rows["src:hit"].flags.f_contam_concept is True
    assert rows["src:hit"].contam_benchmark == "usmle"
    for clean in ("src:ans_diff", "src:ques_diff", "src:ans_none"):
        assert rows[clean].flags.f_contam_concept is False, clean
        assert rows[clean].contam_benchmark is None, clean
    assert manifest.notes["per_benchmark_hits"] == {"usmle": 1}
    assert manifest.notes["contam_rows"] == 1


def test_contam_attribution_strongest_jaccard_then_lexicographic(tmp_path: Path) -> None:
    item = _item("src:1", cuis=[*_SHARED_17, "X1", "X2", "X3"], answer_cui="C100")
    records = [
        ("zzz_bank", _SHARED_17, "C100"),  # J = 17/20 = 0.85, qualifying but weakest
        ("medqa", _SHARED_17 | {"X1", "X2"}, "C100"),  # J = 0.95, tied...
        ("medmcqa", _SHARED_17 | {"X1", "X2"}, "C100"),  # ...loses the tie-break
        ("no_ans", _SHARED_17 | {"X1", "X2"}, None),  # can never qualify, not indexed
        ("diff_ans", _SHARED_17 | {"X1", "X2"}, "C999"),  # answer mismatch, never qualifies
    ]
    manifest, rows = _run(tmp_path, [item], records)

    assert rows["src:1"].contam_benchmark == "medmcqa"
    # Every benchmark with a qualifying record for the row counts once (S4's
    # convention), independent of which one the tie-break named.
    assert manifest.notes["per_benchmark_hits"] == {"medmcqa": 1, "medqa": 1, "zzz_bank": 1}


def test_eval_records_accept_plain_tuple_form(tmp_path: Path) -> None:
    """The documented record shape (benchmark, cuis, answer_cui) coerces to the
    same decisions as the dataclass -- the future S4-style index speaks tuples."""

    def fresh_item() -> CorpusItem:
        return _item("src:1", cuis=list(_SHARED_17), answer_cui="C100")

    as_dataclass = _run(
        tmp_path / "variant1",
        [fresh_item()],
        [concept.ConceptEvalItem(benchmark="usmle", cuis=_SHARED_17, answer_cui="C100")],
    )
    as_tuple = _run(tmp_path / "variant2", [fresh_item()], [("usmle", list(_SHARED_17), "C100")])
    assert as_dataclass[0].notes == as_tuple[0].notes
    assert as_dataclass[1]["src:1"].flags.set_names() == as_tuple[1]["src:1"].flags.set_names()
    assert as_dataclass[1]["src:1"].contam_benchmark == "usmle"


# ----------------------------------------------------------------------------------
# (3) Stage plumbing: skip semantics, flag preservation, idempotency, manifest
# ----------------------------------------------------------------------------------


def test_no_eval_records_skips_contam_but_still_dedups(tmp_path: Path) -> None:
    en = _item("src_a:1", cuis=["C001", "C002"], answer_cui="C009", question=EN_Q)
    fr = _item("src_a:2", cuis=["C001", "C002"], answer_cui="C009", lang="fr", question=FR_Q)
    manifest, rows = _run(tmp_path, [en, fr], None)

    assert rows["src_a:2"].flags.f_dup_concept is True, "dedup needs no eval data"
    assert all(not rows[i].flags.f_contam_concept for i in ("src_a:1", "src_a:2"))
    assert manifest.notes["contam_status"] == "skipped"
    assert "contam_skip_reason" in manifest.notes
    assert manifest.notes["contam_rows"] == 0
    assert manifest.flag_rates["f_contam_concept"]["_all"] == 0.0


def test_prior_stage_flags_and_attribution_are_preserved(tmp_path: Path) -> None:
    """S3/S4 already spoke about this row: S8 adds its flags, never resets
    earlier ones, and never steals S4's contam_benchmark attribution."""
    prior = _item(
        "src_a:2",
        cuis=list(_SHARED_17),
        answer_cui="C100",
        flags=Flags(f_dup_minhash=True, f_contam_ngram=True),
        contam_benchmark="medqa",
    )
    canonical = _item("src_a:1", cuis=list(_SHARED_17), answer_cui="C100")
    _, rows = _run(tmp_path, [prior, canonical], [_EVAL_RECORD])

    got = rows["src_a:2"]
    assert got.flags.f_dup_minhash and got.flags.f_contam_ngram
    assert got.flags.f_dup_concept and got.flags.f_contam_concept
    assert got.dup_of == "src_a:1"
    assert got.contam_benchmark == "medqa", "first attribution wins across stages"


def test_idempotent_over_own_output(tmp_path: Path) -> None:
    inp = tmp_path / "06_decontam_sem"
    inp.mkdir()
    store.write_items(
        [
            _item("src_a:1", cuis=["C001", "C002"], answer_cui="C009", question=EN_Q),
            _item("src_a:2", cuis=["C001", "C002"], answer_cui="C009", lang="fr", question=FR_Q),
            _item("src_a:3", cuis=list(_SHARED_17), answer_cui="C100"),
        ],
        inp,
    )
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    concept.stage_entry("r", eval_items=[_EVAL_RECORD], input_dir=inp, output_dir=out1)
    manifest2 = concept.stage_entry("r", eval_items=[_EVAL_RECORD], input_dir=out1, output_dir=out2)

    assert _fingerprint(out1) == _fingerprint(out2), "no new flags, same dup_of/benchmark"
    assert manifest2.notes["concept_dup_groups"] == 1
    assert manifest2.notes["contam_rows"] == 1, "rates count flags carried, not newly set"
    assert manifest2.input_sha256 == manifest2.output_sha256, "re-run changed no row"


def test_rerun_replaces_own_snapshot_instead_of_appending(tmp_path: Path) -> None:
    inp = tmp_path / "06_decontam_sem"
    inp.mkdir()
    store.write_items(
        [_item(f"src_a:{i}", cuis=["C001", "C002"], answer_cui="C009") for i in range(3)],
        inp,
    )
    out = tmp_path / "08_concept"
    m1 = concept.stage_entry("r", input_dir=inp, output_dir=out)
    m2 = concept.stage_entry("r", input_dir=inp, output_dir=out)

    assert m1.rows_out == m2.rows_out == 3, "write_items appends: stale parts must go first"
    assert m2.output_sha256 == m1.output_sha256


def test_missing_input_snapshot_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="does not exist"):
        concept.stage_entry("r", input_dir=tmp_path / "nope", output_dir=tmp_path / "out")


def test_empty_input_snapshot_fails_loudly(tmp_path: Path) -> None:
    inp = tmp_path / "06_decontam_sem"
    inp.mkdir()
    with pytest.raises(StageError, match="empty"):
        concept.stage_entry("r", input_dir=inp, output_dir=tmp_path / "out")


def test_manifest_contract(tmp_path: Path) -> None:
    en = _item("src_a:1", cuis=["C001", "C002"], answer_cui="C009", question=EN_Q)
    fr = _item("src_a:2", cuis=["C001", "C002"], answer_cui="C009", lang="fr", question=FR_Q)
    other = _item("src_b:1", cuis=["C777"], answer_cui="C888", source="src_b")
    manifest, _rows = _run(tmp_path, [en, fr, other], [_EVAL_RECORD], cui_namespace="sctid")

    assert manifest.stage == "08_concept"
    assert manifest.run_id == "test-run"
    assert manifest.rows_in == manifest.rows_out == 3
    assert manifest.input_sha256 and manifest.output_sha256
    assert manifest.thresholds["concept_jaccard"] == THRESHOLDS.concept_jaccard
    # Backbone-agnostic ids: the namespace is provenance, recorded verbatim.
    assert manifest.config["cui_namespace"] == "sctid"
    assert manifest.config["n_eval_items"] == 1
    assert manifest.config["dup_identity"] == concept.DUP_IDENTITY
    assert manifest.config["contam_rule"] == concept.CONTAM_RULE

    rates = manifest.flag_rates["f_dup_concept"]
    assert rates["src_a"] == pytest.approx(1 / 2)
    assert rates["src_b"] == 0.0, "every source present, zero included"
    assert rates["_all"] == pytest.approx(1 / 3)
    assert manifest.flag_rates["f_contam_concept"]["src_b"] == 0.0
    assert manifest.flag_rates["f_contam_concept"]["_all"] == 0.0
