"""S9 answer-verification tests — pure decision paths, no IO needed."""

from __future__ import annotations

from medrl.curation.schema import CorpusItem
from medrl.curation.stages.answers import verify_mcqa, verify_numeric


def _mcqa(thinking: str, final: str, answer: str = "B", options=None) -> CorpusItem:
    return CorpusItem(
        id="t:1",
        source="t",
        answer_type="mcqa",
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": final}],
        thinking=thinking,
        answer=answer,
        meta={"options": options} if options else {},
    )


def test_mcqa_correct_letter_passes():
    item = _mcqa("The screening test is X.", "Answer: B")
    assert verify_mcqa(item) == {
        "f_answer_wrong": False,
        "f_answer_right_reasoning_contradicts": False,
    }


def test_mcqa_wrong_letter_flagged():
    item = _mcqa("It is clearly A... no wait, let me be careful.", "Answer: C", answer="B")
    assert verify_mcqa(item)["f_answer_wrong"] is True


def test_mcqa_right_but_contradicts_flagged():
    item = _mcqa(
        "Initially I thought A. Actually, the correct mechanism is the other one, so B.",
        "Answer: B",
    )
    flags = verify_mcqa(item)
    assert flags["f_answer_wrong"] is False
    assert flags["f_answer_right_reasoning_contradicts"] is True


def test_mcqa_no_extractable_answer_not_flagged():
    item = _mcqa("rambling", "no contract line here")
    assert verify_mcqa(item)["f_answer_wrong"] is False


def test_mcqa_gold_from_option_text():
    item = _mcqa(
        "reasoning", "Answer: A", answer="Metformin", options=["Metformin", "Insulin", "Glipizide"]
    )
    assert verify_mcqa(item)["f_answer_wrong"] is False


def test_numeric_within_tolerance():
    item = CorpusItem(
        id="t:2",
        source="t",
        answer_type="numeric",
        answer="1200",
        messages=[{"role": "assistant", "content": "...the dose is 1195 mg daily."}],
    )
    assert verify_numeric(item) is False  # within 0.5% -> not wrong


def test_numeric_outside_tolerance_flagged():
    item = CorpusItem(
        id="t:3",
        source="t",
        answer_type="numeric",
        answer="1200",
        messages=[{"role": "assistant", "content": "...the dose is 950 mg daily."}],
    )
    assert verify_numeric(item) is True


def test_numeric_window_from_meta():
    item = CorpusItem(
        id="t:4",
        source="t",
        answer_type="numeric",
        answer="1180",
        meta={"lower": 1150, "upper": 1250},
        messages=[{"role": "assistant", "content": "value: 1200"}],
    )
    assert verify_numeric(item) is False


def test_numeric_unparseable_gold_not_flagged():
    item = CorpusItem(
        id="t:5",
        source="t",
        answer_type="numeric",
        answer="see table",
        messages=[{"role": "assistant", "content": "value: 100"}],
    )
    assert verify_numeric(item) is False
