"""Rubric scoring, reliability layers, and judge calibration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from medrl.eval.scorers.judge import (
    Criterion,
    CriterionVerdict,
    DeterministicJudge,
    JudgeError,
    LLMJudge,
    RubricScore,
)
from medrl.eval.scorers.rubric import (
    aggregate,
    aggregate_micro,
    calibration_report,
    grade_with_robustness,
    score_conversation,
)

CONV = [
    {"role": "user", "content": "I have chest pain, what should I do?"},
    {"role": "assistant", "content": "Seek immediate care; call emergency services now."},
]
CRITERIA = [
    Criterion(id="c1", text="+seek immediate care", weight=2.0),
    Criterion(id="c2", text="+call emergency services", weight=1.0),
    Criterion(id="c3", text="-recommend ignoring the pain", weight=1.0),
]


class _FakeClient:
    """Minimal openai-compatible client double recording every call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        outer = self

        def create(**kwargs: Any) -> Any:
            outer.calls.append(kwargs)
            reply = outer._replies.pop(0)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_deterministic_end_to_end_fraction() -> None:
    score = score_conversation(CONV, CRITERIA, DeterministicJudge())
    assert score.n_met == 3 and score.n_total == 3
    assert score.met_fraction == 1.0
    assert score.points == 100.0


def test_weighting_changes_the_fraction() -> None:
    criteria = [Criterion("a", "+seek immediate care", 3.0), Criterion("b", "+not present", 1.0)]
    score = score_conversation(CONV, criteria, DeterministicJudge())
    assert score.met_fraction == 0.75  # 3 of 4 weight units met
    assert score.weighted is True


def test_majority_vote_flips_minority() -> None:
    class _Voting(DeterministicJudge):
        """Met on pass 1, unmet on passes 2-3: majority is False."""

        def __init__(self) -> None:
            self.n = 0

        def grade(self, messages, criteria):  # type: ignore[override]
            self.n += 1
            return [
                CriterionVerdict(id=c.id, met=(self.n == 1 and c.id == "c1"), weight=c.weight)
                for c in criteria
            ]

    score = grade_with_robustness(CONV, CRITERIA[:1], _Voting(), n_consistency=3, position_swap=False)
    assert score.n_met == 0


def test_position_swap_detects_order_sensitive_judge() -> None:
    class _OrderSensitive:
        """Says yes to whatever appears first in the list; a pathological but real bias."""

        def grade(self, messages, criteria):
            first = criteria[0].id
            return [CriterionVerdict(id=c.id, met=(c.id == first), weight=c.weight) for c in criteria]

    score = grade_with_robustness(CONV, CRITERIA, _OrderSensitive(), n_consistency=1, position_swap=True)
    # c1 is met only in the original order, c3 only in the reversed one, c2 in neither:
    # two flips, and the conservative AND rule pays nothing.
    assert score.flip_count == 2
    assert score.n_met == 0


def test_position_swap_agrees_with_stable_judge() -> None:
    score = grade_with_robustness(CONV, CRITERIA, DeterministicJudge(), n_consistency=2, position_swap=True)
    assert score.flip_count == 0
    assert score.n_met == 3


def test_macro_vs_micro_aggregation() -> None:
    one = RubricScore(met_fraction=1.0, verdicts=(), n_met=2, n_total=2)  # 2 light criteria
    half = RubricScore(met_fraction=0.5, verdicts=(), n_met=5, n_total=10)  # 10 heavy criteria
    assert aggregate([one, half]) == 75.0
    # micro over the same verdicts would weight the 10-criterion conversation more; with
    # empty verdict tuples micro is degenerate, so drive it through real ones instead.
    heavy = [
        CriterionVerdict(id=f"h{i}", met=(i < 5), weight=5.0) for i in range(10)
    ]
    light = [CriterionVerdict(id="l0", met=True, weight=1.0), CriterionVerdict(id="l1", met=True, weight=1.0)]
    s1 = RubricScore(met_fraction=1.0, verdicts=tuple(light), n_met=2, n_total=2)
    s2 = RubricScore(met_fraction=0.5, verdicts=tuple(heavy), n_met=5, n_total=10)
    assert aggregate([s1, s2]) == 75.0
    assert aggregate_micro([s1, s2]) == pytest.approx(100.0 * (2.0 + 25.0) / (2.0 + 50.0))


def test_kappa_extremes() -> None:
    ids = [f"c{i}" for i in range(10)]
    a = [CriterionVerdict(id=i, met=True, weight=1.0) for i in ids]
    b = [CriterionVerdict(id=i, met=True, weight=1.0) for i in ids]
    assert calibration_report(a, b)["cohens_kappa"] == 1.0

    # Anti-correlated alternating judgments: expected agreement 0.5, observed 0 -> kappa -1.
    a_alt = [CriterionVerdict(id=i, met=(n % 2 == 0), weight=1.0) for n, i in enumerate(ids)]
    b_alt = [CriterionVerdict(id=i, met=(n % 2 == 1), weight=1.0) for n, i in enumerate(ids)]
    assert calibration_report(a_alt, b_alt)["cohens_kappa"] == pytest.approx(-1.0)

    # Prevalence paradox: a constant judge against its inverse disagrees on everything,
    # yet kappa is exactly 0 -- chance agreement is also 0. Raw agreement (0 here) is the
    # number to check instead, and the reason kappa alone cannot audit a judge.
    b_inv = [CriterionVerdict(id=i, met=False, weight=1.0) for i in ids]
    report = calibration_report(a, b_inv)
    assert report["cohens_kappa"] == 0.0
    assert report["agreement"] == 0.0

    # Partial agreement lands strictly between the extremes: flip one of ten verdicts on
    # a 50/50 base and kappa must be in (0, 1) -- not pinned to either end.
    a_half = [CriterionVerdict(id=i, met=(n < 5), weight=1.0) for n, i in enumerate(ids)]
    b_near = [CriterionVerdict(id=i, met=(n < 4), weight=1.0) for n, i in enumerate(ids)]
    kappa = calibration_report(a_half, b_near)["cohens_kappa"]
    assert 0.0 < kappa < 1.0


def test_llm_judge_parses_fenced_json() -> None:
    client = _FakeClient(['```json\n{"met": ["c1", "c2"]}\n```'])
    verdicts = LLMJudge(model="j", client=client).grade(CONV, CRITERIA)
    assert {v.id: v.met for v in verdicts} == {"c1": True, "c2": True, "c3": False}


def test_llm_judge_recovers_with_one_retry() -> None:
    client = _FakeClient(["sorry, here are the results you asked for", '{"met": ["c1"]}'])
    verdicts = LLMJudge(model="j", client=client).grade(CONV, CRITERIA)
    assert client.calls[1]["messages"][0]["content"].endswith("JSON only.")
    assert {v.id: v.met for v in verdicts}["c1"] is True


def test_llm_judge_raises_after_retry() -> None:
    client = _FakeClient(["garbage", "still garbage"])
    with pytest.raises(JudgeError, match="unparseable"):
        LLMJudge(model="j", client=client).grade(CONV, CRITERIA)


def test_llm_judge_bounds_generation_and_passes_extra_body() -> None:
    client = _FakeClient(['{"met": ["c1"]}'])
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    LLMJudge(model="j", client=client, extra_body=extra).grade(CONV, CRITERIA)
    assert client.calls[0]["max_tokens"] == 512
    assert client.calls[0]["extra_body"] == extra


def test_llm_judge_omits_extra_body_when_unset() -> None:
    client = _FakeClient(['{"met": ["c1"]}'])
    LLMJudge(model="j", client=client).grade(CONV, CRITERIA)
    assert "extra_body" not in client.calls[0]
    # The bound is not optional: an unbounded thinking judge never emits the JSON.
    assert client.calls[0]["max_tokens"] == 512


def test_llm_judge_ignores_invented_ids() -> None:
    client = _FakeClient(['{"met": ["c1", "hallucinated-9"]}'])
    verdicts = LLMJudge(model="j", client=client).grade(CONV, CRITERIA)
    assert all(v.id != "hallucinated-9" for v in verdicts)
    assert sum(v.met for v in verdicts) == 1
