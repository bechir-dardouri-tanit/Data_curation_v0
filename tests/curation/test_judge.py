"""S11 judge tests — pure prompt/parse/score paths plus gateway-injected end-to-end runs.

The end-to-end tests inject a fake chat function by subclassing Gateway (chat returns
scripted replies, no HTTP), which drives the real run_generation resume machinery —
the same seam production uses, minus the vLLM server.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, Flags, StageError
from medrl.curation.serving import Gateway, ServerHandle
from medrl.curation.stages.judge import (
    _RETRY_NOTE,
    _SYSTEM,
    AXIS_TO_Q,
    DEFAULT_AXES_FILE,
    RETRY_KEY_SUFFIX,
    VERDICTS_FILENAME,
    axis_verdict,
    build_axis_prompt,
    load_axes,
    parse_verdict,
    stage_entry,
)
from medrl.eval.scorers.judge import Criterion

CRITERIA = (
    Criterion(id="c-heavy", text="heavy criterion", weight=3.0),
    Criterion(id="c-light", text="light criterion", weight=1.0),
)

TEST_AXES_YAML = """\
axes:
  coherence:
    criteria:
      - {id: coh-heavy, text: heavy coherence criterion, weight: 3.0}
      - {id: coh-light, text: light coherence criterion, weight: 1.0}
  clinical:
    criteria:
      - {id: clin-only, text: the clinical criterion, weight: 2.0}
  formatting:
    criteria:
      - {id: fmt-a, text: formatting criterion a, weight: 1.0}
      - {id: fmt-b, text: formatting criterion b, weight: 1.0}
"""


class FakeGateway(Gateway):
    """Gateway whose chat is the injected fake: no HTTP, replies scripted per prompt."""

    def __init__(self, reply: Callable[[list[dict[str, str]]], str]) -> None:
        super().__init__(ServerHandle(model="fake-judge", port=1), max_concurrency=8)
        self.reply = reply
        self.user_prompts: list[str] = []

    async def chat(
        self, client: Any, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        self.user_prompts.append(messages[-1]["content"])
        return {"content": self.reply(messages), "reasoning": None, "usage": {}}


def _item(id_: str, source: str = "s1", flags: Flags | None = None, **q: int) -> CorpusItem:
    updates: dict[str, Any] = {"flags": flags} if flags else {}
    updates.update(q)
    return CorpusItem(
        id=id_,
        source=source,
        messages=[
            {"role": "user", "content": f"question for {id_}"},
            {"role": "assistant", "content": f"response for {id_}"},
        ],
        answer_type="free_text",
        **updates,
    )


def _axes_file(tmp_path: Path) -> Path:
    p = tmp_path / "axes.yaml"
    p.write_text(TEST_AXES_YAML)
    return p


def _run(
    tmp_path: Path, reply: Callable[[list[dict[str, str]]], str], items: list[CorpusItem], **kw: Any
) -> tuple[Any, Path, FakeGateway]:
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir(exist_ok=True)
    store.write_items(items, inp)
    gw = FakeGateway(reply)
    manifest = stage_entry(
        "test-run",
        input_dir=inp,
        output_dir=out,
        axes_path=_axes_file(tmp_path),
        gateway=gw,
        **kw,
    )
    return manifest, out, gw


def _scripted_reply(messages: list[dict[str, str]]) -> str:
    """Good row 'a' meets everything; row 'b' misses each axis in a different way:
    coherence 1/4 (below), clinical 0/2 (below), formatting 1/2 (exactly at 0.5)."""
    user = messages[-1]["content"]
    who = "a" if "s1:a" in user else "b"
    if "coh-heavy" in user:
        met = ["coh-heavy", "coh-light"] if who == "a" else ["coh-light"]
    elif "clin-only" in user:
        met = ["clin-only"] if who == "a" else []
    else:
        met = ["fmt-a", "fmt-b"] if who == "a" else ["fmt-a"]
    return json.dumps({"met": met})


# --------------------------------------------------------------------------
# parse_verdict -- the met-ids protocol
# --------------------------------------------------------------------------


def test_parse_verdict_met_ids() -> None:
    assert parse_verdict('{"met": ["c-heavy", "c-light"]}', CRITERIA) == {"c-heavy", "c-light"}


def test_parse_verdict_tolerates_decoration() -> None:
    fenced = 'Sure!\n```json\n{"met": ["c-light"]}\n```\nHope that helps.'
    assert parse_verdict(fenced, CRITERIA) == {"c-light"}


def test_parse_verdict_drops_unknown_ids() -> None:
    assert parse_verdict('{"met": ["c-heavy", "invented-id"]}', CRITERIA) == {"c-heavy"}


def test_parse_verdict_malformed_is_conservative_zero() -> None:
    junk = ["", "no json at all", '{"met": "c-heavy"}', '{"met": [null]}', '{"nope": 1}']
    for text in junk:
        assert parse_verdict(text, CRITERIA) == set(), text


# --------------------------------------------------------------------------
# weighted threshold behaviour
# --------------------------------------------------------------------------


def test_axis_verdict_weighted_threshold() -> None:
    assert axis_verdict({"c-heavy"}, CRITERIA, 0.5) == 1  # 3/4
    assert axis_verdict({"c-light"}, CRITERIA, 0.5) == 0  # 1/4
    assert axis_verdict({"c-heavy", "c-light"}, CRITERIA, 0.5) == 1  # 4/4
    assert axis_verdict(set(), CRITERIA, 0.5) == 0  # 0/4


def test_axis_verdict_threshold_is_inclusive() -> None:
    even = (Criterion(id="a", text="a", weight=1.0), Criterion(id="b", text="b", weight=1.0))
    assert axis_verdict({"a"}, even, 0.5) == 1  # exactly 0.5 passes


def test_build_axis_prompt_carries_all_surfaces() -> None:
    prompt = build_axis_prompt(_item("s1:p"), CRITERIA)
    assert "question for s1:p" in prompt
    assert "response for s1:p" in prompt
    assert "c-heavy" in prompt and "c-light" in prompt
    assert '"met"' in prompt  # the reply contract is stated in the prompt itself


# --------------------------------------------------------------------------
# axes file
# --------------------------------------------------------------------------


def test_default_axes_file_valid() -> None:
    axes = load_axes(DEFAULT_AXES_FILE)
    assert set(axes) == set(AXIS_TO_Q)
    for criteria in axes.values():
        ids = [c.id for c in criteria]
        assert ids and len(set(ids)) == len(ids)
        assert all(c.weight > 0 for c in criteria)
        heaviest, *rest = sorted(criteria, key=lambda c: -c.weight)
        # A critical failure must fail its axis alone: the remaining criteria can
        # never reach the pass threshold. This is the yaml's weighting invariant.
        assert heaviest.weight > sum(c.weight for c in rest)


def test_load_axes_rejects_unknown_axis(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("axes:\n  vibes:\n    criteria:\n      - {id: v1, text: x, weight: 1.0}\n")
    with pytest.raises(StageError, match="axes must be exactly"):
        load_axes(bad)


# --------------------------------------------------------------------------
# end-to-end through the injected gateway
# --------------------------------------------------------------------------


def test_stage_entry_judges_survivors_only(tmp_path: Path) -> None:
    items = [_item("s1:a"), _item("s1:b"), _item("s1:c", flags=Flags(f_dup_exact=True))]
    manifest, out, gw = _run(tmp_path, _scripted_reply, items)

    assert manifest.stage == "11_judge"
    assert manifest.rows_in == manifest.rows_out == 3
    assert manifest.notes["judged"] == 2
    assert manifest.notes["skipped"] == 0
    assert manifest.notes["flagged_passthrough"] == 1
    assert all("s1:c" not in p for p in gw.user_prompts)  # flagged rows never billed

    rows = {it.id: it for it in store.iter_items(out)}
    assert (rows["s1:a"].q_coherence, rows["s1:a"].q_clinical, rows["s1:a"].q_format) == (1, 1, 1)
    assert (rows["s1:b"].q_coherence, rows["s1:b"].q_clinical, rows["s1:b"].q_format) == (0, 0, 1)
    assert (rows["s1:c"].q_coherence, rows["s1:c"].q_clinical, rows["s1:c"].q_format) == (
        None,
        None,
        None,
    )

    rates = manifest.notes["axis_pass_rates"]
    assert rates["q_coherence"]["_all"] == pytest.approx(0.5)
    assert rates["q_clinical"]["_all"] == pytest.approx(0.5)
    assert rates["q_format"]["_all"] == pytest.approx(1.0)


def test_resume_skips_fully_judged_rows(tmp_path: Path) -> None:
    manifest1, out, _ = _run(tmp_path, _scripted_reply, [_item("s1:a"), _item("s1:b")])
    assert manifest1.notes["judged"] == 2

    silent = FakeGateway(lambda messages: "should never be called")
    manifest2 = stage_entry(
        "test-run",
        input_dir=tmp_path / "in",
        output_dir=out,
        axes_path=_axes_file(tmp_path),
        gateway=silent,
    )
    assert manifest2.notes["judged"] == 0
    assert manifest2.notes["skipped"] == 2
    assert manifest2.notes["gen"]["ran"] == 0
    assert silent.user_prompts == []

    rows = {it.id: it for it in store.iter_items(out)}
    assert rows["s1:a"].q_format == 1
    assert rows["s1:b"].q_coherence == 0


def test_resume_is_per_axis_for_partially_judged_rows(tmp_path: Path) -> None:
    preset = _item("s1:a", q_coherence=1)
    manifest, out, gw = _run(tmp_path, _scripted_reply, [preset])

    assert manifest.notes["judged"] == 1
    assert len(gw.user_prompts) == 2  # coherence already done: two axes judged, not three
    rows = {it.id: it for it in store.iter_items(out)}
    assert rows["s1:a"].q_coherence == 1  # preset value preserved...
    assert rows["s1:a"].q_clinical == 1  # ...missing axes filled
    assert rows["s1:a"].q_format == 1


def test_malformed_reply_retried_once_then_used(tmp_path: Path) -> None:
    def reply(messages: list[dict[str, str]]) -> str:
        if messages[0]["content"] == _SYSTEM + _RETRY_NOTE:
            if "coh-heavy" in messages[-1]["content"]:
                return json.dumps({"met": ["coh-heavy", "coh-light"]})
            if "clin-only" in messages[-1]["content"]:
                return json.dumps({"met": ["clin-only"]})
            return json.dumps({"met": ["fmt-a", "fmt-b"]})
        return "the judge rambles, no json here <<>>"

    manifest, out, _ = _run(tmp_path, reply, [_item("s1:a")])
    rows = {it.id: it for it in store.iter_items(out)}
    assert (rows["s1:a"].q_coherence, rows["s1:a"].q_clinical, rows["s1:a"].q_format) == (1, 1, 1)
    assert manifest.notes["retry_jobs"] == 3
    assert manifest.notes["gen"]["ran"] == 6  # three primaries + three retries
    lines = (out / VERDICTS_FILENAME).read_text().splitlines()
    assert sum(RETRY_KEY_SUFFIX in line for line in lines) == 3


def test_unparseable_after_retry_is_conservative_zero(tmp_path: Path) -> None:
    manifest, out, _ = _run(tmp_path, lambda messages: "still not json", [_item("s1:a")])
    rows = {it.id: it for it in store.iter_items(out)}
    assert (rows["s1:a"].q_coherence, rows["s1:a"].q_clinical, rows["s1:a"].q_format) == (0, 0, 0)
    assert manifest.notes["unparseable_after_retry"] == {
        "coherence": 1,
        "clinical": 1,
        "formatting": 1,
    }


def test_limit_slices_input(tmp_path: Path) -> None:
    items = [_item(f"s1:{letter}") for letter in "abcd"]
    manifest, _out, _gw = _run(tmp_path, _scripted_reply, items, limit=2)
    assert manifest.rows_in == manifest.rows_out == 2


# --------------------------------------------------------------------------
# fail-loud contracts
# --------------------------------------------------------------------------


def test_missing_input_snapshot_raises(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="input snapshot"):
        stage_entry(
            "test-run",
            input_dir=tmp_path / "nope",
            output_dir=tmp_path / "out",
            axes_path=_axes_file(tmp_path),
        )


def test_missing_axes_file_raises(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    store.write_items([_item("s1:a")], inp)
    with pytest.raises(StageError, match="axes file"):
        stage_entry(
            "test-run",
            input_dir=inp,
            output_dir=tmp_path / "out",
            axes_path=tmp_path / "nope.yaml",
        )


def test_rubric_change_invalidates_cached_verdicts_and_prior_q(tmp_path: Path) -> None:
    """Regression: cache keys were only `<id>::<axis>` and nothing compared the
    cached rubric to the current one -- a re-run after editing judge_axes.yaml
    or swapping the judge silently reused verdicts graded under the old rubric
    (and fully-judged rows were never re-asked at all via the prior-q harvest)."""
    inp = tmp_path / "in"
    inp.mkdir()
    store.write_items([_item("s1:a"), _item("s1:b")], inp)

    yaml_v1 = tmp_path / "axes_v1.yaml"
    yaml_v1.write_text(TEST_AXES_YAML)
    gw1 = FakeGateway(
        lambda messages: '{"met": ["coh-heavy", "coh-light", "clin-only", "fmt-a", "fmt-b"]}'
    )
    stage_entry("r", input_dir=inp, output_dir=tmp_path / "out", axes_path=yaml_v1, gateway=gw1)
    n_v1_prompts = len(gw1.user_prompts)
    assert n_v1_prompts == 6  # 2 rows x 3 axes

    yaml_v2 = tmp_path / "axes_v2.yaml"
    yaml_v2.write_text(TEST_AXES_YAML.replace("weight: 3.0", "weight: 5.0"))
    gw2 = FakeGateway(lambda messages: '{"met": []}')
    manifest2 = stage_entry(
        "r", input_dir=inp, output_dir=tmp_path / "out", axes_path=yaml_v2, gateway=gw2
    )

    # every row re-asked under the new rubric, not answered from cache
    assert len(gw2.user_prompts) == n_v1_prompts
    assert manifest2.notes["gen"]["resumed"] == 0
    assert manifest2.notes["axis_pass_rates"]["q_coherence"]["_all"] == 0.0
