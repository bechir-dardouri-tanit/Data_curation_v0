"""Grading and generation-runtime semantics on CPU: store/resume, thinking control,
letter rescue, tolerance windows, and HealthBench signed-point folding."""

from __future__ import annotations

import pytest

from medrl.core.config import SamplingConfig, ThinkingConfig, ThinkingMode
from medrl.eval.generate import (
    CompletionStore,
    GenRecord,
    _extract_ok,
    _messages_with_thinking,
    _seed_for,
    generate_all,
)
from medrl.eval.grade import (
    RepeatOutcome,
    extraction_fail_rate,
    grade_benchmark,
    grade_letter,
    grade_number,
    think_completion_rate,
)
from medrl.eval.items import EvalItem, VerifySpec
from medrl.eval.scorers.judge import Criterion, CriterionVerdict, DeterministicJudge
from medrl.eval.scorers.rubric import _fold
from medrl.eval.tasks.spec import VerifyStyle
from tests.test_loaders import MEDQA_ROW, _map


def _letter_item() -> EvalItem:
    item, _ = _map("medqa", MEDQA_ROW)
    return item


def _record(item: EvalItem, content: str | None, *, repeat: int = 0, **kw: object) -> GenRecord:
    return GenRecord(
        benchmark=item.benchmark, item_id=item.item_id, repeat=repeat,
        content=content, reasoning=None, finish_reason="stop",
        prompt_tokens=10, completion_tokens=20, **kw,
    )


# ------------------------------------------------------------------ letter + rescue
def test_contract_answer_scores_correct() -> None:
    item = _letter_item()
    assert grade_letter(item, _record(item, "Reasoning...\nAnswer: B")).score == 1.0


def test_wrong_letter_scores_zero() -> None:
    item = _letter_item()
    assert grade_letter(item, _record(item, "Answer: C")).score == 0.0


def test_failed_extraction_counts_even_when_rescued() -> None:
    item = _letter_item()
    outcome = grade_letter(item, _record(item, "the answer is obviously B", retry_content="Answer: B"))
    assert outcome.score == 1.0
    assert outcome.rescued is True
    assert outcome.extraction_path == "contract"
    # The rescue protects the score, not the statistic: the drift still counts.
    rate = extraction_fail_rate([_ItemOut([outcome])])
    assert rate == 1.0


class _ItemOut:
    def __init__(self, repeats: list[RepeatOutcome]) -> None:
        self.repeats = repeats


def test_unparseable_retry_keeps_failed_path_and_zero() -> None:
    item = _letter_item()
    outcome = grade_letter(item, _record(item, "garbage", retry_content="still garbage"))
    assert outcome.score == 0.0
    assert outcome.rescued is False
    assert outcome.extraction_path == "failed"


def test_extraction_fail_rate_counts_failed_paths() -> None:
    item = _letter_item()
    ok = grade_letter(item, _record(item, "Answer: B"))
    bad = grade_letter(item, _record(item, "no marker at all"))
    rate = extraction_fail_rate([_ItemOut([ok, bad])])
    assert rate == pytest.approx(0.5)


# --------------------------------------------------------------------------- number
def _number_item(**verify_kw: object) -> EvalItem:
    return EvalItem(
        benchmark="medcalc", item_id="1",
        messages=({"role": "system", "content": "s"}, {"role": "user", "content": "q"}),
        verify=VerifySpec(style=VerifyStyle.NUMBER, **verify_kw),  # type: ignore[arg-type]
    )


def test_number_uses_dataset_window_when_present() -> None:
    item = _number_item(gold_number=25.2381, lower=23.97619, upper=26.50001)
    assert grade_number(item, _record(item, "CrCl = 24.0 mL/min")).score == 1.0
    assert grade_number(item, _record(item, "CrCl = 25.2 mL/min")).score == 1.0  # rtol would pass too
    assert grade_number(item, _record(item, "CrCl = 30.0 mL/min")).score == 0.0


def test_number_falls_back_to_rtol_without_window() -> None:
    item = _number_item(gold_number=100.0, rtol=0.005)
    assert grade_number(item, _record(item, "Answer: 100.2")).score == 1.0
    assert grade_number(item, _record(item, "Answer: 101")).score == 0.0


def test_number_no_number_found_scores_zero() -> None:
    item = _number_item(gold_number=100.0)
    assert grade_number(item, _record(item, "cannot determine")).score == 0.0


# ------------------------------------------------------------------ HealthBench fold
def test_negative_criteria_subtract_and_clamp_at_zero() -> None:
    verdicts = [
        CriterionVerdict(id="good", met=True, weight=10.0),
        CriterionVerdict(id="harm", met=True, weight=-10.0),
    ]
    assert _fold(verdicts).met_fraction == 0.0  # 10 - 10, and further harm cannot go below 0
    only_harm = [CriterionVerdict(id="harm", met=True, weight=-10.0)]
    assert _fold(only_harm).met_fraction == 0.0


def test_all_positive_rubrics_unchanged_by_signed_semantics() -> None:
    verdicts = [
        CriterionVerdict(id="a", met=True, weight=2.0),
        CriterionVerdict(id="b", met=False, weight=1.0),
    ]
    assert _fold(verdicts).met_fraction == pytest.approx(2 / 3)


def test_zero_weight_criterion_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        Criterion(id="z", text="vacuous", weight=0.0)


# ------------------------------------------------------------------ rubric pipeline
def test_grade_benchmark_rubric_with_deterministic_judge() -> None:
    item = EvalItem(
        benchmark="hb", item_id="x",
        messages=({"role": "system", "content": "s"},
                  {"role": "user", "content": "what should I do about chest pain?"}),
        verify=VerifySpec(
            style=VerifyStyle.RUBRIC,
            criteria=(Criterion(id="c1", text="+seek immediate care", weight=2.0),),
        ),
    )
    record = _record(item, "You should seek immediate care right away.")
    outcomes = grade_benchmark([item], [record], judge=DeterministicJudge())
    assert outcomes[0].scores == [1.0]
    assert think_completion_rate(outcomes) == 1.0


def test_grade_benchmark_contains_judge_failure_to_a_missing_repeat() -> None:
    """A sporadic unparseable verdict must cost one repeat, not the whole run."""
    from medrl.eval.scorers.judge import JudgeError

    class _FlakyJudge(DeterministicJudge):
        failures_left = 1

        def grade(self, messages, criteria):  # type: ignore[override]
            if _FlakyJudge.failures_left > 0:
                _FlakyJudge.failures_left -= 1
                raise JudgeError("judge output unparseable after retry: 'thinking...'")
            return super().grade(messages, criteria)

    item = EvalItem(
        benchmark="hb", item_id="x",
        messages=({"role": "system", "content": "s"},
                  {"role": "user", "content": "what should I do about chest pain?"}),
        verify=VerifySpec(
            style=VerifyStyle.RUBRIC,
            criteria=(Criterion(id="c1", text="+seek immediate care", weight=2.0),),
        ),
    )
    records = [_record(item, "You should seek immediate care.", repeat=0),
               _record(item, "You should seek immediate care.", repeat=1)]
    outcomes = grade_benchmark([item], records, judge=_FlakyJudge())
    # Repeat 0's verdict was lost to the instrument; repeat 1 survives intact.
    assert len(outcomes[0].repeats) == 1
    assert outcomes[0].repeats[0].score == 1.0
    assert outcomes[0].repeats[0].repeat == 1


def test_grade_benchmark_rubric_requires_judge() -> None:
    item = EvalItem(
        benchmark="hb", item_id="x",
        messages=({"role": "system", "content": "s"}, {"role": "user", "content": "q"}),
        verify=VerifySpec(style=VerifyStyle.RUBRIC,
                          criteria=(Criterion(id="c1", text="+x"),)),
    )
    from medrl.eval.grade import GradingError

    with pytest.raises(GradingError, match="judge"):
        grade_benchmark([item], [_record(item, "answer")])


# --------------------------------------------------------------------- generation
def test_store_roundtrip_and_resume(tmp_path) -> None:
    store = CompletionStore(tmp_path / "c.jsonl")
    item = _letter_item()
    store.write(_record(item, "Answer: B"))
    assert store.has("medqa", item.item_id, 0)
    assert not store.has("medqa", item.item_id, 1)

    reopened = CompletionStore(tmp_path / "c.jsonl")
    assert reopened.has("medqa", item.item_id, 0)
    records = CompletionStore.read_all(tmp_path / "c.jsonl")
    assert records[0].content == "Answer: B"


def test_store_skips_errored_and_torn_lines(tmp_path) -> None:
    # Resume must treat errored records as incomplete (they regenerate, so a
    # transient server failure cannot permanently zero an item) and survive a
    # torn final line (the signature of a killed run) without crashing.
    item = _letter_item()
    path = tmp_path / "c.jsonl"
    store = CompletionStore(path)
    store.write(_record(item, "Answer: B"))
    store.write(_record(item, None, error="APIError: boom", repeat=1))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"benchmark": "medqa", "item_i')  # torn write

    reopened = CompletionStore(path)
    assert reopened.has("medqa", item.item_id, 0)  # usable record: resume-complete
    assert not reopened.has("medqa", item.item_id, 1)  # errored: must regenerate


def test_thinking_knobs_travel_in_extra_body(tmp_path) -> None:
    # Regression (adversarial review, confirmed): the vLLM knobs splatted as
    # top-level kwargs raise TypeError against the SDK's closed create()
    # signature -- client-side, before any request, swallowed by the retry loop.
    item = _letter_item()
    client = _FakeClient()
    generate_all(
        client, "m", [item],
        SamplingConfig(n_repeats=1), ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0),
        CompletionStore(tmp_path / "c.jsonl"),
    )
    call = client.calls[0]
    assert call["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 40,  # vLLM extension params ride in the body too
    }
    assert "chat_template_kwargs" not in call


def test_sampling_config_travels_in_full() -> None:
    from medrl.eval.generate import _one

    item = _letter_item()
    client = _FakeClient()
    _one(client, "m", item, 0,
         SamplingConfig(n_repeats=1, top_k=37, presence_penalty=0.15),
         ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0))
    call = client.calls[0]
    assert call["presence_penalty"] == 0.15
    assert call["extra_body"]["top_k"] == 37
    assert "top_k" not in call  # vLLM extension, not an SDK parameter


def test_error_field_clears_on_success(tmp_path) -> None:
    """A retry that lands must not store its earlier failure string."""
    from medrl.eval.generate import _one

    class _OnceFlaky(_FakeClient):
        n = 0

        def create(self, **kw):
            _OnceFlaky.n += 1
            if _OnceFlaky.n == 1:
                raise RuntimeError("transient")
            return super().create(**kw)

    item = _letter_item()
    rec = _one(_OnceFlaky(), "m", item, 0,
               SamplingConfig(n_repeats=1), ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0))
    assert rec.content is not None and rec.error is None


def test_generate_all_aborts_when_server_dies(tmp_path) -> None:
    from medrl.eval.generate import GenerationAbortedError

    # The liveness probe is consulted every 64 writes; 65 items guarantee the
    # boundary is crossed while work remains.
    items = [_letter_item() for _ in range(65)]
    store = CompletionStore(tmp_path / "c.jsonl")
    with pytest.raises(GenerationAbortedError, match="died"):
        generate_all(
            _FakeClient(), "m", items,
            SamplingConfig(n_repeats=1), ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0),
            store, max_workers=1, alive=lambda: False,
        )


def test_generate_all_aborts_on_error_streak(tmp_path, monkeypatch) -> None:
    import medrl.eval.generate as gen
    from medrl.eval.generate import GenerationAbortedError

    # Shrink the streak and the backoff: the production values (256, [5, 30]s)
    # exist for a 55k-completion GPU run, not for a unit test.
    monkeypatch.setattr(gen, "_ABORT_ERROR_STREAK", 2)
    monkeypatch.setattr(gen, "_BACKOFF_S", (0.0, 0.0))

    class _Dead(_FakeClient):
        def create(self, **kw):
            raise RuntimeError("connection refused")

    items = [_letter_item() for _ in range(4)]
    store = CompletionStore(tmp_path / "c.jsonl")
    with pytest.raises(GenerationAbortedError, match="server process is alive but not serving"):
        generate_all(
            _Dead(), "m", items,
            SamplingConfig(n_repeats=1), ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0),
            store, max_workers=1, alive=lambda: True,
        )
    # Every attempted item errored and was persisted: the diagnosis is on disk.
    assert store.path.stat().st_size > 0


def test_seeds_are_process_and_order_independent() -> None:
    a = _seed_for("medqa", "q1", 0, 7)
    b = _seed_for("medqa", "q1", 0, 7)
    assert a == b
    assert a != _seed_for("medqa", "q1", 1, 7)
    assert a != _seed_for("medqa", "q2", 0, 7)


def test_thinking_off_sends_template_kwarg() -> None:
    item = _letter_item()
    thinking = ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0)
    messages, extra = _messages_with_thinking(item, thinking)
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert messages[-1]["role"] == "user"


def test_thinking_on_prefills_open_think_tag() -> None:
    item = _letter_item()
    thinking = ThinkingConfig(mode=ThinkingMode.ON, prefill_think=True)
    messages, extra = _messages_with_thinking(item, thinking)
    assert messages[-1] == {"role": "assistant", "content": "<think>\n"}
    assert extra["continue_final_message"] is True


def test_extract_ok_semantics() -> None:
    item = _letter_item()
    assert _extract_ok("Answer: B", item)
    assert not _extract_ok("no marker", item)
    assert not _extract_ok(None, item)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 7})()


class _FakeClient:
    """Returns 'Answer: B' for every request; records every call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def chat(self) -> _FakeClient:
        return self

    @property
    def completions(self) -> _FakeClient:
        return self

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        top_p: float | None = None,
        presence_penalty: float = 0.0,
        max_tokens: int,
        seed: int,
        timeout: float,
        extra_body: dict | None = None,
    ) -> _FakeResponse:
        # Closed signature on purpose: the real openai SDK has no **kwargs, so any
        # vLLM knob passed top-level (instead of via extra_body) raises TypeError
        # before a request is ever sent. This fake must fail the same way.
        self.calls.append(
            {
                "presence_penalty": presence_penalty,
                "model": model, "messages": messages, "temperature": temperature,
                "top_p": top_p, "max_tokens": max_tokens, "seed": seed,
                "timeout": timeout, "extra_body": extra_body,
            }
        )
        return _FakeResponse("Answer: B")


def test_generate_all_respects_resume(tmp_path) -> None:
    item = _letter_item()
    store = CompletionStore(tmp_path / "c.jsonl")
    store.write(_record(item, "Answer: B"))  # repeat 0 already on disk
    client = _FakeClient()
    done = generate_all(
        client, "m", [item],
        SamplingConfig(n_repeats=2), ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0),
        store,
    )
    assert done == {"medqa": 1}  # only repeat 1 generated
    assert len(client.calls) == 1
    assert CompletionStore.read_all(tmp_path / "c.jsonl").__len__() == 2


def test_generate_all_no_retry_when_extraction_ok(tmp_path) -> None:
    item = _letter_item()
    store = CompletionStore(tmp_path / "c.jsonl")
    client = _FakeClient()
    generate_all(
        client, "m", [item],
        SamplingConfig(n_repeats=1), ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0),
        store,
    )
    assert len(client.calls) == 1  # parsed fine -> no constrained re-ask
    rec = CompletionStore.read_all(tmp_path / "c.jsonl")[0]
    assert rec.retry_content is None and rec.error is None


def test_generate_all_fires_constrained_retry_on_bad_format(tmp_path) -> None:
    item = _letter_item()
    store = CompletionStore(tmp_path / "c.jsonl")

    class _BadThenGood(_FakeClient):
        def create(self, **kwargs: object) -> _FakeResponse:
            self.calls.append(kwargs)  # type: ignore[arg-type]
            return _FakeResponse("no marker" if len(self.calls) == 1 else "Answer: B")

    client = _BadThenGood()
    generate_all(
        client, "m", [item],
        SamplingConfig(n_repeats=1), ThinkingConfig(mode=ThinkingMode.OFF, think_budget=0),
        store,
    )
    assert len(client.calls) == 2  # primary + constrained re-ask
    rec = CompletionStore.read_all(tmp_path / "c.jsonl")[0]
    assert rec.retry_content == "Answer: B"
    outcome = grade_letter(item, rec)
    assert outcome.score == 1.0 and outcome.rescued is True


def test_prefill_sends_the_validated_flag_pair() -> None:
    # vLLM validates these as a pair: continuing a prefilled final message is only
    # legal with the generation-prompt header off (its default is on -> 400).
    item = _letter_item()
    messages, extra = _messages_with_thinking(
        item, ThinkingConfig(mode=ThinkingMode.ON, think_budget=64)
    )
    assert messages[-1] == {"role": "assistant", "content": "<think>\n"}
    assert extra == {"add_generation_prompt": False, "continue_final_message": True}


def test_read_all_collapses_regenerations(tmp_path) -> None:
    # Resume appends a regeneration after the errored attempt it replaces; grading
    # must see ONE record per key -- the usable one -- or the stale zero shifts
    # repeat columns and biases every statistic.
    item = _letter_item()
    path = tmp_path / "c.jsonl"
    store = CompletionStore(path)
    store.write(_record(item, None, error="APIError: boom"))
    store.write(_record(item, "Answer: B"))
    other = _record(item, "Answer: C", repeat=1, error="APIError: still down")
    store.write(other)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"benchmark": "medqa", "item_i')  # torn write from a kill

    records = CompletionStore.read_all(path)
    assert len(records) == 2
    by_repeat = {r.repeat: r for r in records}
    assert by_repeat[0].content == "Answer: B"
    assert by_repeat[1].error == "APIError: still down"  # no usable successor: visible


def test_fresh_store_discards_existing_completions(tmp_path) -> None:
    item = _letter_item()
    path = tmp_path / "c.jsonl"
    CompletionStore(path).write(_record(item, "Answer: B"))
    store = CompletionStore(path, fresh=True)
    assert not store.has("medqa", item.item_id, 0)
    assert CompletionStore.read_all(path) == []


def test_strict_incomplete_zeroes_rescued_truncation() -> None:
    # A reasoning chain that never closed earns no credit for the letter the
    # constrained re-ask recovered -- unless the config explicitly relaxes it.
    item = _letter_item()  # gold B
    record = _record(item, "", retry_content="Answer: B")
    strict = grade_letter(item, record, strict_incomplete=True)
    lax = grade_letter(item, record, strict_incomplete=False)
    assert strict.score == 0.0 and strict.rescued and not strict.think_completed
    assert lax.score == 1.0 and lax.rescued
