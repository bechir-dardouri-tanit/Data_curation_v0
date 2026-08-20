"""Grading: stored completions -> per-item scores.

This module is the single place where a completion becomes a number, and it is pure CPU:
the judge (if any) runs during the judge phase and is injected as a callable, so the
grading logic itself is testable with the deterministic reference judge.

Semantics per verify style:

- ``letter``: extract via the shared parser (CONTRACT -> LAST_LINE -> FAILED), falling
  back to the constrained re-ask's output when the primary parse failed. The *primary*
  parse decides the extraction-failure statistic; a rescued answer scores normally but
  the rescue is counted, so a rising rescue rate is visible long before it becomes
  score-relevant.
- ``number``: the dataset's own tolerance window when it ships one (MedCalc's
  Lower/Upper), otherwise relative tolerance on the gold.
- ``rubric``: HealthBench-style judge over signed-weight criteria, with the reliability
  layers from :mod:`medrl.eval.scorers.rubric`.
- ``format_rules``: IFEval semantics -- every instruction in the item must hold; the
  score is strict prompt-level instruction following, as published.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from medrl.core.logging import get_logger
from medrl.eval.extraction import ExtractionPath, extract_mcqa
from medrl.eval.generate import GenRecord
from medrl.eval.items import EvalItem
from medrl.eval.scorers.judge import Judge
from medrl.eval.scorers.rubric import grade_with_robustness
from medrl.eval.tasks.spec import VerifyStyle
from medrl.eval.verifiers import parse_quantity, verify_letter, verify_number

log = get_logger(__name__)


class GradingError(RuntimeError):
    """An item could not be graded; a harness bug, not a model result."""


@dataclass(frozen=True)
class RepeatOutcome:
    repeat: int
    score: float
    extraction_path: str | None  # letters only; None for non-MCQA styles
    rescued: bool  # primary parse failed, constrained re-ask recovered
    think_completed: bool


@dataclass
class ItemOutcome:
    benchmark: str
    item_id: str
    repeats: list[RepeatOutcome] = field(default_factory=list)

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.repeats]


def grade_letter(item: EvalItem, record: GenRecord, *, strict_incomplete: bool = True) -> RepeatOutcome:
    verify = item.verify
    result = extract_mcqa(record.content or "", verify.letters)
    rescued = False
    if result.path is ExtractionPath.FAILED and record.retry_content:
        # The re-ask is server-constrained to the contract regex, so a non-parsing
        # retry means the constraint was rejected, not that the model misbehaved;
        # count it as a plain extraction failure and let the run-level gate see it.
        retry = extract_mcqa(record.retry_content, verify.letters)
        if retry.path is ExtractionPath.CONTRACT:
            result = retry
            rescued = True
    think_completed = bool(record.content)
    correct = (
        result.path is not ExtractionPath.FAILED
        and verify_letter(result.value, verify.gold_letter or "", verify.letters)
    )
    if strict_incomplete and not think_completed:
        # The reasoning chain never closed (or the request failed): a constrained
        # re-ask may still recover a *letter*, but a truncated chain has not earned
        # credit for it. The rescue stays visible via `rescued` either way.
        correct = False
    return RepeatOutcome(
        repeat=record.repeat,
        score=1.0 if correct else 0.0,
        extraction_path=result.path.value,
        rescued=rescued,
        think_completed=think_completed,
    )


def grade_number(item: EvalItem, record: GenRecord) -> RepeatOutcome:
    verify = item.verify
    parsed = parse_quantity(record.content or "")
    value = parsed[0] if parsed else None
    if value is not None and verify.lower is not None and verify.upper is not None:
        ok = verify.lower <= value <= verify.upper
    elif value is not None and verify.gold_number is not None:
        ok = verify_number(value, verify.gold_number, rtol=verify.rtol, atol=verify.atol)
    else:
        ok = False
    return RepeatOutcome(
        repeat=record.repeat,
        score=1.0 if ok else 0.0,
        extraction_path=None,
        rescued=False,
        think_completed=bool(record.content),
    )


def grade_rubric(item: EvalItem, record: GenRecord, judge: Judge, *, n_consistency: int, position_swap: bool) -> RepeatOutcome:
    messages = [dict(m) for m in item.messages]
    messages.append({"role": "assistant", "content": record.content or ""})
    score = grade_with_robustness(
        messages, item.verify.criteria, judge,
        n_consistency=n_consistency, position_swap=position_swap,
    )
    return RepeatOutcome(
        repeat=record.repeat,
        score=score.met_fraction,
        extraction_path=None,
        rescued=False,
        think_completed=bool(record.content),
    )


def grade_format_rules(item: EvalItem, record: GenRecord) -> RepeatOutcome:
    ok = _check_ifeval(record.content or "", item.verify.instruction_ids, item.verify.instruction_kwargs)
    return RepeatOutcome(
        repeat=record.repeat,
        score=1.0 if ok else 0.0,
        extraction_path=None,
        rescued=False,
        think_completed=bool(record.content),
    )


_IFEVAL_REGISTRY: object | None = None
_IFEVAL_MISSING = (
    "no IFEval instruction checker is installed; Google's instruction_following_eval is "
    "not on PyPI -- install it explicitly, e.g. "
    "uv pip install 'instruction_following_eval "
    "@ git+https://github.com/josejg/instruction_following_eval' "
    "(or inspect-evals, which re-exports it), once that source is accepted"
)


def ifeval_available() -> bool:
    try:
        _load_ifeval_registry()
        return True
    except ImportError:
        return False


def _load_ifeval_registry() -> object:
    """Google's instruction checkers. Not published to PyPI under their own name; the
    pip-installable source the community (and inspect_evals) points at is
    josejg/instruction_following_eval on GitHub -- deliberately NOT an automatic
    dependency. Install it explicitly once the provenance is accepted."""
    try:
        import instruction_following_eval as ife  # type: ignore[import-not-found]
    except ImportError:
        import inspect_evals.ifeval as ife  # type: ignore[import-not-found]
    return ife.instructions_registry


def _check_ifeval(response: str, instruction_ids: tuple[str, ...], kwargs_list: tuple[str, ...]) -> bool:
    """IFEval strict prompt-level check: every instruction in the item must hold.

    Lazy import: the checker library is heavyweight and only this style needs it.
    """
    global _IFEVAL_REGISTRY
    if _IFEVAL_REGISTRY is None:
        try:
            _IFEVAL_REGISTRY = _load_ifeval_registry()
        except ImportError as exc:
            raise GradingError(_IFEVAL_MISSING) from exc
    registry = _IFEVAL_REGISTRY
    for iid, kwargs_json in zip(instruction_ids, kwargs_list, strict=False):
        kwargs = json.loads(kwargs_json) if kwargs_json else {}
        instruction = registry.INSTRUCTION_DICT[iid](iid, kwargs)  # type: ignore[attr-defined]
        instruction.build_description()
        if not instruction.check_following(response):
            return False
    return True


def grade_benchmark(
    items: list[EvalItem],
    records: list[GenRecord],
    *,
    judge: Judge | None = None,
    judge_n_consistency: int = 1,
    judge_position_swap: bool = True,
    strict_incomplete: bool = True,
    max_workers: int = 32,
) -> list[ItemOutcome]:
    """Grade all records for one benchmark; rubric items fan out across threads.

    ``records`` must already be collapsed to one per (item, repeat) --
    :meth:`CompletionStore.read_all` does that. ``strict_incomplete`` implements the
    ThinkingConfig field of the same name: a repeat whose reasoning never closed
    scores 0 even when the constrained re-ask recovered a well-formed letter.
    """
    by_item: dict[str, list[GenRecord]] = {}
    for record in records:
        by_item.setdefault(record.item_id, []).append(record)

    needs_judge = items and items[0].verify.style is VerifyStyle.RUBRIC
    if needs_judge and judge is None:
        raise GradingError("rubric items require a judge; none was configured")

    def _grade(item: EvalItem) -> ItemOutcome:
        outcome = ItemOutcome(benchmark=item.benchmark, item_id=item.item_id)
        for record in sorted(by_item.get(item.item_id, []), key=lambda r: r.repeat):
            if record.error and not record.content:
                log.warning("%s/%s repeat %d has error %s; scoring 0", item.benchmark, item.item_id, record.repeat, record.error[:80])
            if item.verify.style is VerifyStyle.LETTER:
                outcome.repeats.append(grade_letter(item, record, strict_incomplete=strict_incomplete))
            elif item.verify.style is VerifyStyle.NUMBER:
                outcome.repeats.append(grade_number(item, record))
            elif item.verify.style is VerifyStyle.RUBRIC:
                assert judge is not None  # grade_benchmark checked before dispatch
                outcome.repeats.append(
                    grade_rubric(
                        item, record, judge,
                        n_consistency=judge_n_consistency,
                        position_swap=judge_position_swap,
                    )
                )
            elif item.verify.style is VerifyStyle.FORMAT_RULES:
                outcome.repeats.append(grade_format_rules(item, record))
            else:  # pragma: no cover - VerifyStyle is closed
                raise GradingError(f"unhandled verify style {item.verify.style}")
        return outcome

    with ThreadPoolExecutor(max_workers=max_workers if needs_judge else 1) as pool:
        return list(pool.map(_grade, items))


def extraction_fail_rate(outcomes: list[ItemOutcome]) -> float:
    """Fraction of letter-style repeats whose *primary* parse failed (pre-rescue).

    Rescued repeats count as failures here on purpose: the rescue protects the score,
    not the statistic -- a model drifting out of the contract must show up in this
    number even while its score holds.
    """
    repeats = [r for o in outcomes for r in o.repeats if r.extraction_path is not None]
    if not repeats:
        return 0.0
    failed = sum(
        1 for r in repeats
        if r.extraction_path == ExtractionPath.FAILED.value or r.rescued
    )
    return failed / len(repeats)


def think_completion_rate(outcomes: list[ItemOutcome]) -> float:
    """Fraction of repeats that produced final content.

    With the reasoning parser active (it always is for the policy server), content is
    empty exactly when the model never closed ``</think>`` (or the request failed), so
    this is the truncation/failure rate. For non-thinking runs it degenerates to the
    non-empty-response rate.
    """
    repeats = [r for o in outcomes for r in o.repeats]
    if not repeats:
        return 0.0
    return sum(1 for r in repeats if r.think_completed) / len(repeats)
