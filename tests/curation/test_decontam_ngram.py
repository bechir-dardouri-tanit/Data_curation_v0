"""Tests for S4 n-gram decontamination (src/medrl/curation/stages/decontam_ngram.py).

Covers the four contract cases: planted contamination flagged with the right
benchmark, paraphrased/unrelated text clean, per-benchmark attribution (with the
tie-break rule), and inclusive threshold-boundary behaviour -- each through the
full store round-trip via ``stage_entry`` with an injected synthetic
``BenchmarkIndex`` and tmp_path dirs (no eval-loader downloads, no /scratch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, Flags, StageError
from medrl.curation.stages import decontam_ngram
from medrl.curation.thresholds import THRESHOLDS
from medrl.data.decontam import BenchmarkIndex, BenchmarkItem
from medrl.eval import loaders as eval_loaders
from medrl.eval.items import EvalItem, VerifySpec
from medrl.eval.tasks.spec import TaskSpec, VerifyStyle

_MEDQA_QUESTION = (
    "A 67-year-old man presents with crushing substernal chest pain radiating to his left arm "
    "and jaw, associated with diaphoresis and shortness of breath. ECG shows ST elevation in "
    "leads II, III, and aVF. What is the most likely diagnosis?"
)
_MEDMCQA_QUESTION = (
    "Which antihypertensive drug class is contraindicated in bilateral renal artery stenosis "
    "because it can precipitate acute kidney injury?"
)
_SHARED_QUESTION = (
    "A 54-year-old woman with rheumatoid arthritis develops a swollen, painful right calf. "
    "What is the most appropriate next diagnostic step?"
)


def _make_index(entries: list[tuple[str, str, str]]) -> BenchmarkIndex:
    """A BenchmarkIndex built exactly per its real constructor, from synthetic items."""
    index = BenchmarkIndex(ngram_n=THRESHOLDS.contam_ngram_n)
    for benchmark, item_id, text in entries:
        index.add(BenchmarkItem(benchmark=benchmark, item_id=item_id, text=text))
    return index


def _corpus_item(item_id: str, question: str, *, source: str = "src_a", **kw: Any) -> CorpusItem:
    return CorpusItem(
        id=item_id,
        source=source,
        messages=[{"role": "user", "content": question}],
        answer_type="none",
        **kw,
    )


def _run_stage(tmp_path: Path, items: list[CorpusItem], index: BenchmarkIndex) -> Any:
    inp = tmp_path / "03_dedup"
    inp.mkdir(parents=True, exist_ok=True)
    store.write_items(items, inp)
    return decontam_ngram.stage_entry(
        "test-run",
        benchmarks=["medqa", "medmcqa"],
        index=index,
        input_dir=inp,
        output_dir=tmp_path / "04_decontam_ngram",
    )


def _out_items(tmp_path: Path) -> dict[str, CorpusItem]:
    out = tmp_path / "04_decontam_ngram"
    return {it.id: it for it in store.iter_items(out)}


# ----------------------------------------------------------------------------------
# (1) planted contamination: a corpus question embedding a benchmark question is
# flagged, with the benchmark name recorded, and everything survives the parquet
# round-trip under flags-not-deletes.
# ----------------------------------------------------------------------------------


def test_planted_question_is_flagged_and_recorded(tmp_path) -> None:
    index = _make_index([("medqa", "q1", _MEDQA_QUESTION)])
    planted = _corpus_item("src_a:1", "Clinical case: " + _MEDQA_QUESTION + " Thank you.")
    clean = _corpus_item("src_a:2", "What should a healthy adult eat before a colonoscopy?")
    # S3 already flagged this row as an exact dup; S4 must ADD its flag, not reset.
    prior_dup = _corpus_item("src_a:3", _MEDQA_QUESTION, flags=Flags(f_dup_exact=True))

    manifest = _run_stage(tmp_path, [planted, clean, prior_dup], index)
    rows = _out_items(tmp_path)

    assert manifest.rows_in == manifest.rows_out == 3
    assert rows["src_a:1"].flags.f_contam_ngram is True
    assert rows["src_a:1"].contam_benchmark == "medqa"
    assert rows["src_a:2"].flags.f_contam_ngram is False
    assert rows["src_a:2"].contam_benchmark is None
    assert rows["src_a:3"].flags.f_dup_exact is True, "S3's flag must survive S4"
    assert rows["src_a:3"].flags.f_contam_ngram is True

    assert manifest.flag_rates["f_contam_ngram"]["src_a"] == pytest.approx(2 / 3)
    assert manifest.flag_rates["f_contam_ngram"]["_all"] == pytest.approx(2 / 3)
    assert manifest.notes["per_benchmark_hits"] == {"medqa": 2}
    assert manifest.thresholds["contam_ngram_n"] == THRESHOLDS.contam_ngram_n
    assert manifest.thresholds["contam_ngram_threshold"] == THRESHOLDS.contam_ngram_threshold
    assert manifest.input_sha256 and manifest.output_sha256
    canary = manifest.notes["canary"]
    assert canary["n_positive_detected"] == canary["n_positive"] == 1


# ----------------------------------------------------------------------------------
# (2) negatives: a genuine rewrite and an unrelated question stay clean.
# ----------------------------------------------------------------------------------


def test_paraphrased_and_unrelated_questions_stay_clean(tmp_path) -> None:
    index = _make_index([("medqa", "q1", _MEDQA_QUESTION)])
    paraphrase = (
        "A male in his late sixties complains of intense central thoracic pain spreading towards "
        "the left upper limb, together with sweating and dyspnoea; the ECG demonstrates "
        "ST-segment elevation in the inferior leads. Which condition best explains this?"
    )
    unrelated = "What is the recommended daily iron intake for a healthy woman of childbearing age?"

    manifest = _run_stage(
        tmp_path,
        [_corpus_item("src_a:1", paraphrase), _corpus_item("src_a:2", unrelated)],
        index,
    )
    rows = _out_items(tmp_path)

    for item_id in ("src_a:1", "src_a:2"):
        assert rows[item_id].flags.f_contam_ngram is False
        assert rows[item_id].contam_benchmark is None
    assert manifest.notes["per_benchmark_hits"] == {}
    assert manifest.flag_rates["f_contam_ngram"]["_all"] == 0.0


# ----------------------------------------------------------------------------------
# (3) per-benchmark attribution: each hit names its benchmark; a question indexed
# under two benchmarks ties at overlap 1.0 and is attributed deterministically
# (lexicographically smallest "benchmark::item_id" -- the index's own hit order is
# set-iteration order and not stable across processes).
# ----------------------------------------------------------------------------------


def test_per_benchmark_attribution_and_deterministic_tie_break(tmp_path) -> None:
    index = _make_index(
        [
            ("medmcqa", "q1", _MEDMCQA_QUESTION),
            ("medqa", "q1", _MEDQA_QUESTION),
            ("medmcqa", "shared", _SHARED_QUESTION),
            ("medqa", "shared", _SHARED_QUESTION),
        ]
    )

    manifest = _run_stage(
        tmp_path,
        [
            _corpus_item("src_a:1", "Board review: " + _MEDMCQA_QUESTION),
            _corpus_item("src_a:2", _SHARED_QUESTION),
        ],
        index,
    )
    rows = _out_items(tmp_path)

    assert rows["src_a:1"].contam_benchmark == "medmcqa"
    assert rows["src_a:2"].contam_benchmark == "medmcqa"  # stable tie-break, not set order
    assert manifest.notes["per_benchmark_hits"] == {"medmcqa": 2, "medqa": 1}


# ----------------------------------------------------------------------------------
# (4) threshold boundary: overlap is Jaccard over n-gram hashes and the data-side
# comparison is inclusive (overlap >= threshold). The construction makes the
# arithmetic exact: a benchmark text of 32 distinct characters has exactly 20
# distinct 13-grams; a verbatim prefix of p characters contributes p-n+1 of them,
# all inside the benchmark set, so overlap is (p-n+1)/20.
# ----------------------------------------------------------------------------------

_UNION_WINDOWS = 20  # 0.80 * 20 = 16 -- integer counts that hit the boundary exactly
_BENCH_TEXT = "abcdefghijklmnopqrstuvwxyz012345"  # 32 distinct normalised chars -> 20 windows


def test_threshold_boundary_is_inclusive(tmp_path) -> None:
    n = THRESHOLDS.contam_ngram_n
    assert n == 13 and THRESHOLDS.contam_ngram_threshold == 0.80  # arithmetic below assumes these
    index = _make_index([("medqa", "b1", _BENCH_TEXT)])

    at_threshold = _BENCH_TEXT[: n + (_UNION_WINDOWS * 4 // 5) - 1]  # 28 chars -> 16/20 == 0.80
    below_threshold = _BENCH_TEXT[: n + (_UNION_WINDOWS * 4 // 5) - 2]  # 27 chars -> 15/20 == 0.75

    hit, bench, overlaps = decontam_ngram.match_question(index, at_threshold)
    assert hit and bench == "medqa"
    assert overlaps == {"medqa": pytest.approx(0.80)}
    hit_below, bench_below, _ = decontam_ngram.match_question(index, below_threshold)
    assert not hit_below and bench_below is None

    manifest = _run_stage(
        tmp_path,
        [_corpus_item("src_a:1", at_threshold), _corpus_item("src_a:2", below_threshold)],
        index,
    )
    rows = _out_items(tmp_path)
    assert rows["src_a:1"].flags.f_contam_ngram is True, "overlap == threshold must flag"
    assert rows["src_a:2"].flags.f_contam_ngram is False, "overlap just below must not flag"
    assert manifest.notes["per_benchmark_hits"] == {"medqa": 1}


def test_short_question_matches_only_verbatim(tmp_path) -> None:
    """Below n-gram length the shared primitive hashes the whole text: only a
    verbatim copy can collide. Inherited medrl.data.dedup behaviour, pinned here."""
    index = _make_index([("medqa", "tiny", "Rx?")])

    manifest = _run_stage(
        tmp_path,
        [_corpus_item("src_a:1", "Rx?"), _corpus_item("src_a:2", "Rx? Please explain the choice.")],
        index,
    )
    rows = _out_items(tmp_path)

    assert rows["src_a:1"].flags.f_contam_ngram is True
    assert rows["src_a:1"].contam_benchmark == "medqa"
    assert rows["src_a:2"].flags.f_contam_ngram is False
    assert manifest.notes["per_benchmark_hits"] == {"medqa": 1}


# ----------------------------------------------------------------------------------
# Index builder: user-role text only, names validated against the eval registry,
# loader faked -- no downloads.
# ----------------------------------------------------------------------------------


def test_build_question_index_indexes_user_role_content_only(monkeypatch) -> None:
    answer_text = "ST elevation myocardial infarction in leads II, III, and aVF."
    system_text = "You are a medical exam grader. Answer with a single letter."
    captured: list[TaskSpec] = []

    def fake_load_items(spec: TaskSpec) -> eval_loaders.LoadResult:
        captured.append(spec)
        return eval_loaders.LoadResult(
            items=[
                EvalItem(
                    benchmark="medqa",
                    item_id="q1",
                    messages=(
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": _MEDQA_QUESTION},
                        {"role": "assistant", "content": answer_text},
                    ),
                    verify=VerifySpec(style=VerifyStyle.LETTER, letters="ABCDE", gold_letter="A"),
                )
            ]
        )

    monkeypatch.setattr(eval_loaders, "load_items", fake_load_items)
    index = decontam_ngram.build_question_index(["medqa"])

    assert captured and captured[0].name == "medqa"  # spec resolved from the eval registry
    assert len(index.items) == 1
    assert index.items["medqa::q1"].text == _MEDQA_QUESTION  # system/assistant text excluded

    detected, bench, _ = decontam_ngram.match_question(index, answer_text)
    assert not detected and bench is None, "gold-answer leakage must not be indexed"
    detected, bench, _ = decontam_ngram.match_question(index, _MEDQA_QUESTION)
    assert detected and bench == "medqa"


def test_build_question_index_rejects_unknown_benchmark() -> None:
    with pytest.raises(StageError, match="not in the eval registry"):
        decontam_ngram.build_question_index(["definitely_not_a_benchmark"])


# ----------------------------------------------------------------------------------
# Canary: positives (verbatim self-matches) must all be detected; negative-control
# detection counts are recorded as the over-aggressiveness measure.
# ----------------------------------------------------------------------------------


def test_canary_records_detection_evidence() -> None:
    index = _make_index(
        [
            ("medqa", "q1", _MEDQA_QUESTION),
            ("medmcqa", "q1", _MEDMCQA_QUESTION),
            ("medqa", "q2", _SHARED_QUESTION),
        ]
    )

    canary = decontam_ngram.run_canary(index, n_items=2, seed=0)

    assert canary["status"] == "ok"
    assert canary["n_positive"] == 2
    assert canary["n_positive_detected"] == 2, "verbatim benchmark text must be caught"
    assert canary["n_negative_controls"] == 2 * len(canary["modifications"])
    assert canary["modifications"] == ["paraphrase", "shuffle", "negate", "perturb"]
    assert canary["n_negative_detected"] == sum(
        canary["negative_detected_by_modification"].values()
    )


def test_canary_skips_empty_index() -> None:
    canary = decontam_ngram.run_canary(BenchmarkIndex(ngram_n=THRESHOLDS.contam_ngram_n))
    assert canary["status"] == "skipped"


# ----------------------------------------------------------------------------------
# re-run idempotence + fail-loud input guards (S2/S8 conventions; S4 was the
# deviating stage: write_items APPENDS, so a re-run used to double the snapshot).
# ----------------------------------------------------------------------------------


def test_rerun_replaces_own_snapshot_instead_of_appending(tmp_path: Path) -> None:
    items = [_corpus_item(f"src_a:{i}", _SHARED_QUESTION) for i in range(4)]
    inp = tmp_path / "03_dedup"
    inp.mkdir(parents=True, exist_ok=True)
    store.write_items(items, inp)
    index = _make_index([("medqa", "m1", _MEDQA_QUESTION)])
    out = tmp_path / "04_decontam_ngram"

    m1 = decontam_ngram.stage_entry(
        "test-run", benchmarks=["medqa"], index=index, input_dir=inp, output_dir=out
    )
    m2 = decontam_ngram.stage_entry(
        "test-run", benchmarks=["medqa"], index=index, input_dir=inp, output_dir=out
    )

    assert m1.rows_out == m2.rows_out == 4, "stale parts must go before the re-write"
    assert m2.output_sha256 == m1.output_sha256


def test_missing_input_snapshot_fails_loudly(tmp_path: Path) -> None:
    index = _make_index([("medqa", "m1", _MEDQA_QUESTION)])
    with pytest.raises(StageError, match="does not exist"):
        decontam_ngram.stage_entry(
            "test-run", benchmarks=["medqa"], index=index,
            input_dir=tmp_path / "nope", output_dir=tmp_path / "out",
        )


def test_empty_input_snapshot_fails_loudly(tmp_path: Path) -> None:
    inp = tmp_path / "03_dedup"
    inp.mkdir(parents=True)
    index = _make_index([("medqa", "m1", _MEDQA_QUESTION)])
    with pytest.raises(StageError, match="empty"):
        decontam_ngram.stage_entry(
            "test-run", benchmarks=["medqa"], index=index,
            input_dir=inp, output_dir=tmp_path / "out",
        )
