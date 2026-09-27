"""S12 difficulty tests: pure band logic to the boundary, the think-retry and
adaptive top-up rounds against a mocked gateway, and the streaming stage entry
over the real parquet store (small passk knobs so one test covers every round)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from medrl.core.hashing import hash_text
from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError
from medrl.curation.serving import Gateway, ServerHandle
from medrl.curation.stages import difficulty
from medrl.curation.thresholds import THRESHOLDS

_OPTS5 = ["Metformin", "Insulin", "Glipizide", "Amoxicillin", "Epinephrine"]

_MAIN_TOKENS = THRESHOLDS.passk_max_tokens
_RETRY_TOKENS = THRESHOLDS.passk_max_tokens + THRESHOLDS.passk_think_budget // 2


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


def _mcqa(item_id: str, q: str, answer: str, options: list[str] | None = _OPTS5) -> CorpusItem:
    return CorpusItem(
        id=item_id,
        source="unit",
        answer_type="mcqa",
        messages=[
            {"role": "user", "content": q},
            {"role": "assistant", "content": "SOURCE_ASSISTANT_TEXT"},
        ],
        answer=answer,
        meta={"options": options} if options is not None else {},
    )


def _numeric(item_id: str, q: str, answer: str, **meta: Any) -> CorpusItem:
    return CorpusItem(
        id=item_id,
        source="unit",
        answer_type="numeric",
        messages=[{"role": "user", "content": q}],
        answer=answer,
        meta=dict(meta),
    )


@pytest.fixture
def small_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """pass@2 with +2 top-up so the end-to-end test exercises every round cheaply.

    Only the sampling knobs move; the band edges stay at the shipped values, so
    route_band semantics are identical to production.
    """
    monkeypatch.setattr(
        difficulty,
        "THRESHOLDS",
        THRESHOLDS.model_copy(
            update={"passk_samples": 2, "passk_topup_samples": 2, "passk_topup_k": (1, 2)}
        ),
    )


def _gateway() -> Gateway:
    return Gateway(ServerHandle(model="test-model", port=1), max_retries=1, max_concurrency=4)


def _handler(request: httpx.Request) -> httpx.Response:
    """The mocked labelling model, routed by question marker + request shape.

    - q:a  correct iff the repeat's seed is even (repeats of one item have
      consecutive seeds, so this is a deterministic per-repeat pattern);
    - q:b  always correct, gold given as option text;
    - q:c  think-incomplete at every budget -> both repeats end up missing;
    - q:d  repeat 0 (even seed) incomplete even after retry, repeat 1 wrong;
    - q:h  incomplete at the main budget, completes correctly on the retry;
    - q:g  numeric, wrong value (gold 120).
    """

    def reply(content: str, reasoning: str | None = None) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content, "reasoning_content": reasoning}}],
                "usage": {},
            },
        )

    body = json.loads(request.content)
    seed = int(body.get("seed") or 0)
    retry = int(body.get("max_tokens") or 0) > _MAIN_TOKENS
    marker = body["messages"][-1]["content"].splitlines()[0].strip()

    if marker == "q:c":
        return reply("", "<think>partial") if not retry else reply("", "<think>still open")
    if marker == "q:d":
        if retry:
            return reply("", "<think>still open")
        return reply("", "<think>partial") if seed % 2 == 0 else reply("Answer: A")
    if marker == "q:h":
        return reply("", "<think>partial") if not retry else reply("Answer: B")
    if marker == "q:g":
        return reply("Computed value: 999.")
    if marker == "q:a":
        return reply("Answer: B" if seed % 2 == 0 else "Answer: A")
    return reply("Answer: B")


def _factory() -> Any:
    def make() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    return make


def _write_input(tmp_path: Any, items: list[CorpusItem]) -> Any:
    inp = tmp_path / "11_judge"
    inp.mkdir(parents=True)
    store.write_items(items, inp)
    return inp


# --------------------------------------------------------------------------
# route_band.
# --------------------------------------------------------------------------


def test_route_band_boundaries():
    """The exact convention: inclusive-lower/exclusive-upper, ends inclusive."""
    assert difficulty.route_band(0.0) == "hold"
    assert difficulty.route_band(0.099) == "hold"
    assert difficulty.route_band(0.1) == "rl"  # band_rl_low, inclusive
    assert difficulty.route_band(0.125) == "rl"  # 1/8
    assert difficulty.route_band(0.4) == "rl"  # band_rl_high is inside rl
    assert difficulty.route_band(0.4375) == "rl"  # 7/16: post-top-up gap stays rl
    assert difficulty.route_band(0.5) == "sft1"  # band_sft1_low, inclusive
    assert difficulty.route_band(0.875) == "sft1"  # 7/8
    assert difficulty.route_band(0.9375) == "sft1"  # 15/16: upper gap stays sft1
    assert difficulty.route_band(0.999) == "sft1"
    assert difficulty.route_band(1.0) == "downsample"  # the one exact end


@pytest.mark.parametrize("bad", [-0.1, 1.01, 2.0, float("nan")])
def test_route_band_rejects_out_of_range(bad: float):
    with pytest.raises(ValueError):
        difficulty.route_band(bad)


def test_route_band_reads_patched_thresholds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        difficulty,
        "THRESHOLDS",
        THRESHOLDS.model_copy(update={"band_rl_low": 0.2, "band_sft1_low": 0.6}),
    )
    assert difficulty.route_band(0.1) == "hold"
    assert difficulty.route_band(0.2) == "rl"
    assert difficulty.route_band(0.5) == "rl"
    assert difficulty.route_band(0.6) == "sft1"


# --------------------------------------------------------------------------
# band_stable.
# --------------------------------------------------------------------------


def test_band_stable_shipped_knobs_always_unstable():
    """With the shipped knobs a top-up window is 8/16 = 0.5 wide, wider than the
    largest band gap (0.4), so every base pass-count could still move band. The
    passk_topup_k tuple, not stability, is what limits who pays for top-up."""
    for k in range(THRESHOLDS.passk_samples + 1):
        assert difficulty.band_stable(k, THRESHOLDS.passk_samples) is False


def test_band_stable_true_when_window_fits_a_band(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        difficulty, "THRESHOLDS", THRESHOLDS.model_copy(update={"passk_topup_samples": 2})
    )
    assert difficulty.band_stable(7, 8)  # window [0.7, 0.9]: all sft1
    assert not difficulty.band_stable(8, 8)  # [0.8, 1.0]: sft1 vs downsample
    assert not difficulty.band_stable(3, 8)  # [0.3, 0.5]: rl vs sft1
    assert not difficulty.band_stable(0, 8)  # [0.0, 0.2]: hold vs rl


@pytest.mark.parametrize("args", [(9, 8), (0, 0)])
def test_band_stable_rejects_impossible_counts(args: tuple[int, int]):
    with pytest.raises(ValueError):
        difficulty.band_stable(*args)


# --------------------------------------------------------------------------
# aggregate.
# --------------------------------------------------------------------------


def test_aggregate_full_pass():
    rate, note = difficulty.aggregate([True] * 8)
    assert rate == 1.0 and note == ""
    rate, note = difficulty.aggregate([True, False, True, False, True, False, True, False])
    assert rate == 0.5 and note == ""


def test_aggregate_missing_repeats_leave_the_denominator():
    rate, note = difficulty.aggregate([True, None, False, None, True, True, False, False])
    assert rate == 0.5  # 3 of 6 valid
    assert note == "missing_repeats=2/8"


def test_aggregate_too_many_missing_is_unrateable():
    rate, note = difficulty.aggregate([True, None, None, None, True, True, True, True])
    assert rate is None
    assert "3 of 8" in note and "unrateable" in note


def test_aggregate_all_missing_but_under_the_guard():
    rate, note = difficulty.aggregate([None, None])
    assert rate is None and "no valid repeats" in note
    rate, note = difficulty.aggregate([])
    assert rate is None and "no valid repeats" in note


# --------------------------------------------------------------------------
# sample_seed / think_incomplete.
# --------------------------------------------------------------------------


def test_sample_seed_deterministic_blocked_by_item():
    base = 1234
    for repeat in range(16):
        expected = (
            base + (int(hash_text("s:1")[:8], 16) % 10**6) * THRESHOLDS.passk_samples + repeat
        )
        assert difficulty.sample_seed("s:1", repeat, base) == expected
    seeds = [difficulty.sample_seed("s:1", r, base) for r in range(THRESHOLDS.passk_samples)]
    assert len(set(seeds)) == THRESHOLDS.passk_samples  # repeats never collide
    assert difficulty.sample_seed("s:1", 3, base) == difficulty.sample_seed("s:1", 3, base)
    assert difficulty.sample_seed("s:2", 3, base) != difficulty.sample_seed("s:1", 3, base)


def test_think_incomplete_per_surface():
    assert difficulty.think_incomplete("<think>partial", None)
    assert difficulty.think_incomplete(None, "<think>partial")  # inline-think servers
    assert difficulty.think_incomplete("<think>a</think>", "Answer: B") is False
    assert difficulty.think_incomplete("<think>a</think>", "<think>b")  # second surface open
    assert difficulty.think_incomplete(None, None) is False
    assert difficulty.think_incomplete("no tags", "Answer: B") is False


# --------------------------------------------------------------------------
# grading, prompts, eligibility.
# --------------------------------------------------------------------------


def test_grade_sample_mcqa():
    item = _mcqa("s:1", "q", "B")
    assert difficulty.grade_sample(item, "...so the answer.\nAnswer: B", None)
    assert not difficulty.grade_sample(item, "Answer: C", None)
    assert not difficulty.grade_sample(item, "rambling with no marker", None)
    assert difficulty.grade_sample(item, "", "long thought ... Answer: B")  # reasoning fallback


def test_grade_sample_mcqa_gold_from_option_text():
    item = _mcqa("s:1", "q", "Insulin")
    assert difficulty.grade_sample(item, "Answer: B", None)
    assert not difficulty.grade_sample(item, "Answer: A", None)


def test_grade_sample_numeric_tolerance_and_window():
    item = _numeric("s:2", "q", "1200")
    assert difficulty.grade_sample(item, "the dose is 1195 mg daily", None)  # within 0.5%
    assert not difficulty.grade_sample(item, "the dose is 950 mg daily", None)
    assert difficulty.grade_sample(item, "", "final value 1199.9")  # reasoning fallback
    windowed = _numeric("s:3", "q", "7.2", lower=6.5, upper=8.0)
    assert difficulty.grade_sample(windowed, "between 7 and 7.9 mmol/L", None)
    assert not difficulty.grade_sample(windowed, "exactly 9", None)
    assert not difficulty.grade_sample(_numeric("s:4", "q", "not a number"), "Answer: 5", None)


def test_sampling_messages_render_options_and_never_the_source_response():
    item = _mcqa("s:1", "What dose of adrenaline?", "B")
    msgs = difficulty.sampling_messages(item)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Answer: <LETTER>" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert user.startswith("What dose of adrenaline?")
    assert "A. Metformin" in user and "E. Epinephrine" in user
    assert "SOURCE_ASSISTANT_TEXT" not in user  # the source response would leak the gold
    num = difficulty.sampling_messages(_numeric("s:2", "Compute the CrCl", "120"))
    assert num[0]["content"].startswith(
        "You are a medical expert performing a clinical calculation"
    )
    assert num[1]["content"] == "Compute the CrCl"


def test_labelable_matrix():
    assert difficulty.labelable(_mcqa("s:1", "q", "B"))
    assert difficulty.labelable(_mcqa("s:1", "q", "Insulin"))
    assert difficulty.labelable(_numeric("s:2", "q", "120"))
    assert not difficulty.labelable(_mcqa("s:1", "q", "Beta"))  # unparseable gold
    assert not difficulty.labelable(_mcqa("s:1", "q", "B", options=None))  # no options to render
    assert not difficulty.labelable(_numeric("s:2", "q", "not a number"))
    assert not difficulty.labelable(
        CorpusItem(
            id="s:3",
            source="unit",
            answer_type="free_text",
            messages=[{"role": "user", "content": "q"}],
        )
    )
    assert not difficulty.labelable(
        CorpusItem(id="s:4", source="unit", answer_type="mcqa", meta={"options": _OPTS5})
    )  # no gold answer
    assert not difficulty.labelable(_mcqa("s:5", "", "B"))  # nothing to prompt with


# --------------------------------------------------------------------------
# Stage entry, end to end over the real store.
# --------------------------------------------------------------------------


def _all_items() -> list[CorpusItem]:
    return [
        _mcqa("s:a", "q:a", "B"),
        _mcqa("s:b", "q:b", "Insulin"),
        _mcqa("s:c", "q:c", "B"),
        _mcqa("s:d", "q:d", "B"),
        _numeric("s:g", "q:g", "120"),
        _mcqa("s:h", "q:h", "B"),
        CorpusItem(
            id="s:e",
            source="unit",
            answer_type="free_text",
            messages=[{"role": "user", "content": "q:e"}, {"role": "assistant", "content": "resp"}],
        ),
        _mcqa("s:f", "q:f", "Beta"),
    ]


def test_stage_entry_end_to_end(tmp_path: Any, small_k: None):
    inp = _write_input(tmp_path, _all_items())
    out = tmp_path / "12_difficulty"
    gen = tmp_path / "gen" / "generations.jsonl"

    manifest = difficulty.stage_entry(
        "run-x",
        input_dir=inp,
        output_dir=out,
        gateway=_gateway(),
        generation_out=gen,
        seed_base=0,
        client_factory=_factory(),
    )

    assert manifest.rows_in == manifest.rows_out == 8
    assert manifest.input_sha256 and manifest.output_sha256
    assert manifest.flag_rates == {}  # S12 sets no Flags fields, by design
    assert manifest.config["retry_max_tokens"] == _RETRY_TOKENS
    assert manifest.config["seed_formula"].startswith("seed_base + hash_text")

    rows = {it.id: it for it in store.iter_items(out)}
    assert len(rows) == 8
    assert rows["s:a"].difficulty == 0.5 and rows["s:a"].difficulty_band == "sft1"
    assert rows["s:b"].difficulty == 1.0 and rows["s:b"].difficulty_band == "downsample"
    assert rows["s:d"].difficulty == 0.0 and rows["s:d"].difficulty_band == "hold"
    assert rows["s:g"].difficulty == 0.0 and rows["s:g"].difficulty_band == "hold"
    assert rows["s:h"].difficulty == 1.0 and rows["s:h"].difficulty_band == "downsample"
    assert rows["s:c"].difficulty is None and rows["s:c"].difficulty_band is None
    for skip in ("s:e", "s:f"):
        assert rows[skip].difficulty is None and rows[skip].difficulty_band is None

    # notes: top-ups and missing repeats are on the rows, counts in the manifest
    assert rows["s:a"].meta["difficulty_note"] == "topup+2"
    assert rows["s:b"].meta["difficulty_note"] == "topup+2"
    assert rows["s:h"].meta["difficulty_note"] == "topup+2"
    assert rows["s:d"].meta["difficulty_note"].startswith("missing_repeats=1/2")
    assert "unrateable" in rows["s:c"].meta["difficulty_note"]
    assert "difficulty_note" not in rows["s:g"].meta  # clean 0.0 is just hold
    assert "difficulty_note" not in rows["s:e"].meta  # untouched passthrough

    assert manifest.notes["by_band"] == {"downsample": 2, "hold": 2, "sft1": 1}
    assert manifest.notes["labelled"] == 5
    assert manifest.notes["unrateable_too_few_valid_repeats"] == 1
    assert manifest.notes["not_labelled"] == {"free_text": 1, "no_gold_answer": 1}
    assert (
        manifest.notes["knob_notes"]
        and "passk_think_budget // 2" in manifest.notes["knob_notes"][0]
    )

    # generation store: every round's keys, and only the rows that needed them
    keys = {json.loads(line)["key"] for line in gen.read_text().splitlines()}
    assert len(keys) == 25  # 12 main + 5 think-retries + 6 top-up + 2 top-up retries
    assert "s:d::0::retry" in keys and "s:d::1::retry" not in keys
    assert "s:h::2" in keys and "s:h::0::retry" in keys
    assert "s:h::2::retry" in keys  # top-up samples get the same incomplete retry
    assert "s:e::0" not in keys and "s:f::0" not in keys
    rounds = manifest.notes["gen_rounds"]
    assert [r["ran"] for r in rounds] == [12, 5, 6, 2]  # main, think-retry, top-up, top-up retry


def test_stage_entry_resume_reproduces_labels(tmp_path: Any, small_k: None):
    inp = _write_input(tmp_path, _all_items())
    gen = tmp_path / "gen" / "generations.jsonl"
    first = difficulty.stage_entry(
        "run-x",
        input_dir=inp,
        output_dir=tmp_path / "12_difficulty",
        gateway=_gateway(),
        generation_out=gen,
        seed_base=0,
        client_factory=_factory(),
    )
    # A fresh output dir, same generation store: every job is a resume hit and
    # the recomputed labels are identical.
    second = difficulty.stage_entry(
        "run-x",
        input_dir=inp,
        output_dir=tmp_path / "12_difficulty_b",
        gateway=_gateway(),
        generation_out=gen,
        seed_base=0,
        client_factory=_factory(),
    )
    assert [r["resumed"] and not r["ran"] for r in second.notes["gen_rounds"]] == [
        True,
        True,
        True,
        True,
    ]
    assert second.notes["by_band"] == first.notes["by_band"]


def test_stage_entry_missing_input_fails_loudly(tmp_path: Any):
    with pytest.raises(StageError, match="11_judge"):
        difficulty.stage_entry(
            "x", input_dir=tmp_path / "11_judge", output_dir=tmp_path / "12_difficulty"
        )


def test_stage_entry_requires_gateway_for_labelable_rows(tmp_path: Any):
    inp = _write_input(tmp_path, [_mcqa("s:a", "q:a", "B")])
    with pytest.raises(StageError, match="gateway"):
        difficulty.stage_entry("x", input_dir=inp, output_dir=tmp_path / "12_difficulty")


def test_stage_entry_unlabelable_only_needs_no_gateway(tmp_path: Any):
    inp = _write_input(
        tmp_path,
        [
            CorpusItem(
                id="s:e",
                source="unit",
                answer_type="free_text",
                messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}],
            )
        ],
    )
    manifest = difficulty.stage_entry("x", input_dir=inp, output_dir=tmp_path / "12_difficulty")
    assert manifest.rows_out == 1
    assert manifest.notes["gen_rounds"] == []
    assert manifest.notes["labelled"] == 0


def test_stage_entry_rerun_replaces_own_snapshot_but_keeps_resume(
    tmp_path: Any, small_k: None
):
    """Regression: stage_entry only mkdir'd its output; write_items APPENDS with
    continuing part numbers, so the documented re-run doubled the snapshot and
    then died on rows_out != rows_in (an AssertionError the runner does not
    catch). generations.jsonl must survive the reset -- it is the resume cache."""
    inp = _write_input(tmp_path, _all_items())
    out = tmp_path / "12_difficulty"
    gen = tmp_path / "gen" / "generations.jsonl"

    first = difficulty.stage_entry(
        "run-x", input_dir=inp, output_dir=out,
        gateway=_gateway(), generation_out=gen, seed_base=0, client_factory=_factory(),
    )
    gen_lines_after_first = len(gen.read_text().splitlines())

    second = difficulty.stage_entry(
        "run-x", input_dir=inp, output_dir=out,
        gateway=_gateway(), generation_out=gen, seed_base=0, client_factory=_factory(),
    )

    assert first.rows_out == second.rows_out == 8, "stale parts must go before the re-write"
    assert second.output_sha256 == first.output_sha256
    # resume cache survived and was reused: no new generation lines appended
    assert len(gen.read_text().splitlines()) == gen_lines_after_first
    assert second.notes["gen_rounds"][0]["ran"] == 0
