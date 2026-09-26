"""Tests for the S3 lexical dedup stage (stages/dedup_lex.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError
from medrl.curation.stages import dedup_lex
from medrl.curation.thresholds import THRESHOLDS
from medrl.data.dedup import ngram_overlap


def _mk(
    id_: str,
    *,
    q: str,
    a: str,
    thinking: str | None = None,
    licence: str = "unknown",
) -> CorpusItem:
    return CorpusItem(
        id=id_,
        source=id_.split(":", 1)[0],
        licence=licence,
        messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        thinking=thinking,
    )


def _words(n: int, offset: int = 0) -> str:
    """Synthetic token stream with controllable 5/13-gram overlap between variants."""
    return " ".join(f"w{i + offset}" for i in range(n))


def _planted_items() -> list[CorpusItem]:
    """One exact pair, one minhash-only paraphrase pair (0.83 <= J < 0.9), one clean row."""
    base = _words(60, 0)
    variant = base.replace("w30", "w3")  # same length: the keep-rule must tie to lane 3
    return [
        _mk("s:base", q="Prompt about aspirin dosing.", a=base),
        _mk("s:variant", q="Prompt about aspirin dosing.", a=variant),
        _mk("s:copy1", q="Exact question?", a="An exact answer."),
        _mk("s:copy2", q="Exact question?", a="An exact answer."),
        _mk("s:other", q="Unrelated prompt?", a=_words(60, 500)),
    ]


def _fingerprint(directory: Path) -> dict[str, tuple[tuple[str, ...], str | None]]:
    """(set flag names, dup_of) per id -- the state an idempotent re-run must preserve."""
    return {it.id: (it.flags.set_names(), it.dup_of) for it in store.iter_items(directory)}


# --------------------------------------------------------------------------
# Keep-rule units (each lane, in rank order)
# --------------------------------------------------------------------------


def test_licence_tier_buckets() -> None:
    assert dedup_lex.licence_tier("Apache-2.0") == 0
    assert dedup_lex.licence_tier("mit") == 0
    assert dedup_lex.licence_tier("CC BY 4.0") == 1
    assert dedup_lex.licence_tier("cc0-1.0") == 1
    assert dedup_lex.licence_tier("unknown") == 2
    assert dedup_lex.licence_tier("") == 2
    assert dedup_lex.licence_tier("proprietary") == 3
    assert dedup_lex.licence_tier("noncommercial-custom") == 3


def test_keep_rule_prefers_permissive_licence() -> None:
    q, a = "Same question?", "Same answer."
    group = [
        _mk("s:prop", q=q, a=a, licence="proprietary"),
        _mk("s:mit", q=q, a=a, licence="mit"),
        _mk("s:cc", q=q, a=a, licence="cc-by-4.0"),
    ]
    winner, lane = dedup_lex.apply_keep_rule(group)
    assert (winner.id, lane) == ("s:mit", "licence")


def test_keep_rule_then_prefers_longer_thinking() -> None:
    q, a = "Same question?", "Same answer."
    group = [
        _mk("s:aaa", q=q, a=a, licence="cc-by-4.0", thinking="short"),
        _mk("s:zzz", q=q, a=a, licence="cc-by-4.0", thinking="a much longer reasoning trace"),
    ]
    winner, lane = dedup_lex.apply_keep_rule(group)
    assert (winner.id, lane) == ("s:zzz", "thinking")


def test_keep_rule_falls_back_to_assistant_length_when_no_thinking() -> None:
    q = "Same question?"
    group = [
        _mk("s:aaa", q=q, a="short answer", licence="unknown"),
        _mk("s:zzz", q=q, a="a considerably longer assistant answer", licence="unknown"),
    ]
    winner, lane = dedup_lex.apply_keep_rule(group)
    assert (winner.id, lane) == ("s:zzz", "thinking")


def test_keep_rule_source_id_is_the_final_total_order() -> None:
    q, a = "Same question?", "Same answer."
    group = [_mk("s:bbb", q=q, a=a), _mk("s:aaa", q=q, a=a), _mk("s:ccc", q=q, a=a)]
    winner, lane = dedup_lex.apply_keep_rule(group)
    assert (winner.id, lane) == ("s:aaa", "source_id")


def test_keep_rule_singleton_undecided() -> None:
    winner, lane = dedup_lex.apply_keep_rule([_mk("s:one", q="q", a="a")])
    assert (winner.id, lane) == ("s:one", None)


# --------------------------------------------------------------------------
# Per-pass units on planted pairs
# --------------------------------------------------------------------------


def test_exact_pass_flags_copy_but_not_whitespace_free_distinct_rows() -> None:
    a = _mk("s:1", q="What treats X?", a="Aspirin treats X.")
    b = _mk("s:2", q="what   treats\nX?", a="ASPIRIN treats X.")  # equal after normalization
    c = _mk("s:3", q="Different row", a="Not a duplicate at all")
    dup_of, lanes = dedup_lex.exact_pass([a, b, c])
    assert dup_of == {"s:2": "s:1"}
    assert lanes == {"source_id": 1}


def test_ngram_pass_catches_word_level_near_copy_that_exact_misses() -> None:
    base = _words(300, 0)
    variant = base.replace("w150", "x150")  # one word swapped, same length
    a = _mk("s:a", q="q", a=base)
    b = _mk("s:b", q="q", a=variant)

    sh_a = dedup_lex.word_shingles(dedup_lex.dedup_text(a), THRESHOLDS.ngram_dup_n)
    sh_b = dedup_lex.word_shingles(dedup_lex.dedup_text(b), THRESHOLDS.ngram_dup_n)
    jac = ngram_overlap(sh_a, sh_b)
    assert THRESHOLDS.ngram_dup_jaccard <= jac < 1.0, "fixture must sit in the n-gram band"

    assert dedup_lex.exact_pass([a, b]) == ({}, {})
    dup_of, lanes = dedup_lex.ngram_pass([a, b])
    assert dup_of == {"s:b": "s:a"}
    assert lanes == {"source_id": 1}


def test_minhash_pass_catches_paraphrase_below_the_ngram_band() -> None:
    base = _words(60, 0)
    variant = base.replace("w30", "w3")
    a = _mk("s:a", q="Prompt about aspirin dosing.", a=base)
    b = _mk("s:b", q="Prompt about aspirin dosing.", a=variant)

    sh_a = dedup_lex.word_shingles(dedup_lex.dedup_text(a), THRESHOLDS.minhash_shingle_n)
    sh_b = dedup_lex.word_shingles(dedup_lex.dedup_text(b), THRESHOLDS.minhash_shingle_n)
    jac = ngram_overlap(sh_a, sh_b)
    assert THRESHOLDS.minhash_jaccard <= jac < THRESHOLDS.ngram_dup_jaccard, (
        "fixture must be caught by MinHash, not by the 13-gram pass"
    )

    assert dedup_lex.exact_pass([a, b]) == ({}, {})
    assert dedup_lex.ngram_pass([a, b]) == ({}, {})
    dup_of, lanes = dedup_lex.minhash_pass([a, b])
    assert dup_of == {"s:b": "s:a"}
    assert lanes == {"thinking": 1}  # base's assistant payload is one char longer


# --------------------------------------------------------------------------
# Stage entry: flags-not-deletes, distinct flag kinds, rates, notes
# --------------------------------------------------------------------------


def test_stage_flags_exact_and_minhash_distinctly(tmp_path: Path) -> None:
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    store.write_items(_planted_items(), inp)

    m = dedup_lex.stage_entry("r", input_dir=inp, output_dir=out)
    got = {it.id: it for it in store.iter_items(out)}

    assert m.rows_in == m.rows_out == 5  # flags-not-deletes
    assert m.stage == "03_dedup"
    assert m.input_sha256 and m.output_sha256

    assert got["s:copy2"].flags.f_dup_exact and got["s:copy2"].dup_of == "s:copy1"
    assert not got["s:copy2"].flags.f_dup_minhash
    assert not got["s:copy1"].flags.any() and got["s:copy1"].dup_of is None

    assert got["s:variant"].flags.f_dup_minhash and got["s:variant"].dup_of == "s:base"
    assert not got["s:variant"].flags.f_dup_exact
    assert not got["s:base"].flags.any()

    assert not got["s:other"].flags.any()

    assert m.notes["exact_groups"] == 1
    assert m.notes["ngram_groups"] == 0
    assert m.notes["minhash_groups"] == 1
    assert m.notes["kept_by_rule"] == {"licence": 0, "thinking": 1, "source_id": 1}

    rates_exact = m.flag_rates["f_dup_exact"]
    assert rates_exact["s"] == pytest.approx(1 / 5)
    assert rates_exact["_all"] == pytest.approx(1 / 5)
    assert m.flag_rates["f_dup_minhash"]["s"] == pytest.approx(1 / 5)


def test_flag_rates_are_per_source_with_corpus_wide_all(tmp_path: Path) -> None:
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    store.write_items(
        [
            _mk("s1:1", q="q1", a="a1"),
            _mk("s1:2", q="q1", a="a1"),  # exact loser in s1
            _mk("s1:3", q="q3", a="a3"),
            _mk("s2:1", q="q4", a="a4"),  # clean row from another source
        ],
        inp,
    )
    m = dedup_lex.stage_entry("r", input_dir=inp, output_dir=out)

    rates = m.flag_rates["f_dup_exact"]
    assert rates["s1"] == pytest.approx(1 / 3)
    assert rates["s2"] == 0.0  # every source present, zero included
    assert rates["_all"] == pytest.approx(1 / 4)
    assert m.flag_rates["f_dup_minhash"]["_all"] == 0.0


def test_distinct_items_pass_through_untouched(tmp_path: Path) -> None:
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    items = [
        _mk("s:a", q="Question one?", a=_words(60, 0)),
        _mk("s:b", q="Question two?", a=_words(60, 1000)),
    ]
    store.write_items(items, inp)

    m = dedup_lex.stage_entry("r", input_dir=inp, output_dir=out)
    assert all(not it.flags.any() and it.dup_of is None for it in store.iter_items(out))
    assert m.notes == {
        "exact_groups": 0,
        "ngram_groups": 0,
        "minhash_groups": 0,
        "kept_by_rule": {"licence": 0, "thinking": 0, "source_id": 0},
    }
    assert m.input_sha256 == m.output_sha256  # nothing changed, not even the bytes


def test_idempotent_over_already_flagged_input(tmp_path: Path) -> None:
    inp, out1, out2 = tmp_path / "in", tmp_path / "out1", tmp_path / "out2"
    inp.mkdir()
    store.write_items(_planted_items(), inp)

    dedup_lex.stage_entry("r", input_dir=inp, output_dir=out1)
    m2 = dedup_lex.stage_entry("r", input_dir=out1, output_dir=out2)

    assert _fingerprint(out1) == _fingerprint(out2)  # no new dup flags, same dup_of
    assert m2.rows_in == m2.rows_out == 5
    assert m2.notes["exact_groups"] == 1
    assert m2.notes["minhash_groups"] == 1


def test_missing_input_snapshot_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(StageError):
        dedup_lex.stage_entry("r", input_dir=tmp_path / "nope", output_dir=tmp_path / "out")
