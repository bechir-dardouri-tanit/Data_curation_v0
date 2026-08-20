"""Eval items: one benchmark row promoted to the shape the runtime consumes.

Pure data, no ML imports. An item is the messages to send plus everything grading needs;
loaders build these, the generator sends them, the grader consumes them. Keeping the type
here -- rather than inside the loaders -- lets generation and grading import it without
dragging dataset code in, and lets tests construct items directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from medrl.eval.scorers.judge import Criterion
from medrl.eval.tasks.spec import VerifyStyle


@dataclass(frozen=True)
class VerifySpec:
    """How to grade a completion for one item.

    One class, style-dispatched: the style picks which subset of fields is meaningful.
    MedCalc-style items carry the dataset's *own* tolerance window (``lower``/``upper``),
    which beats any guessed ``rtol`` -- the dataset authors already encoded acceptable
    rounding for each calculator.
    """

    style: VerifyStyle
    # letter verification
    letters: str = "ABCDE"
    gold_letter: str | None = None
    # number verification
    gold_number: float | None = None
    lower: float | None = None
    upper: float | None = None
    rtol: float = 0.005
    atol: float = 1e-8
    # rubric verification (HealthBench)
    criteria: tuple[Criterion, ...] = ()
    # instruction-following verification (IFEval)
    instruction_ids: tuple[str, ...] = ()
    instruction_kwargs: tuple[str, ...] = ()  # one JSON object string per instruction


@dataclass(frozen=True)
class EvalItem:
    """Everything the runtime needs about one benchmark item."""

    benchmark: str
    item_id: str
    messages: tuple[dict[str, str], ...]
    verify: VerifySpec
    # Grammar for the constrained re-ask when free-form extraction fails (e.g.
    # "Answer: [A-E]"). Rendered from the item's own alphabet, because option counts
    # vary *inside* datasets (MediQAl and MedXpertQA both have 4- and 5-option rows).
    retry_grammar: str | None = None


def item_key(benchmark: str, item_id: str) -> str:
    """Stable identity of an item across generation and grading phases."""
    return f"{benchmark}::{item_id}"
