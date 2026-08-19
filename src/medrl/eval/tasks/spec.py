"""Benchmark task specifications.

A task spec is *pure data*: how to build the prompt, how to extract the answer, how to
verify it, what the metric is. Everything here runs on a CPU box with no ML stack, so the
contract the model is graded against is testable -- and the same module feeds both the
Inspect-AI runtime and (via ``medrl.rl.rewards``) the RL reward, which is what keeps the
train-time and report-time target identical.

The Inspect-AI adapter lives in :mod:`medrl.eval.tasks.inspect_adapter` and is imported
only at run time.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator

from medrl.core.config import BenchmarkConfig
from medrl.core.registry import Registry


class PromptStyle(StrEnum):
    MCQA_LETTER = "mcqa_letter"
    """Numbered options, strict terminal ``Answer: <LETTER>`` contract."""

    OPEN_RUBRIC = "open_rubric"
    """Free-text answer graded against weighted binary criteria (HealthBench style)."""

    NUMERIC = "numeric"
    """Free-text answer verified as a number within tolerance (MedCalc style)."""


class VerifyStyle(StrEnum):
    LETTER = "letter"
    NUMBER = "number"
    RUBRIC = "rubric"
    FORMAT_RULES = "format_rules"


class TaskSpec(BenchmarkConfig):
    """One benchmark end to end, minus the model."""

    prompt_style: PromptStyle
    verify_style: VerifyStyle
    verify_params: dict[str, str] = Field(default_factory=dict)
    # The MCQA answer alphabet. Drives the option labels, the guided-decoding grammar
    # and the extraction/verification classes; a task whose dataset has 10 options
    # (MMLU-Pro) MUST declare A-J or items with gold F-J are forced-and-scored wrong.
    letters: str = "ABCDE"
    # Guided decoding grammar for the final answer; removes extraction failures by
    # construction for MCQA instead of regexing around them.
    guided_decoding: str | None = None
    pass_at_k: int = 1
    notes: str | None = None

    @field_validator("letters")
    @classmethod
    def _letters_is_contiguous_range(cls, value: str) -> str:
        upper = value.upper()
        if not upper.startswith("A") or ord(upper[-1]) - ord(upper[0]) + 1 != len(upper):
            raise ValueError(
                f"letters must be a contiguous uppercase range starting at A, got {value!r}"
            )
        return upper

    def with_overrides(self, **kwargs: object) -> Self:
        data = self.model_dump()
        data.update(kwargs)
        return self.__class__.model_validate(data)

    def as_benchmark_config(self) -> BenchmarkConfig:
        """The runtime view: prompt/verify styling is resolved through the registry,
        so the eval config carries only what a run needs."""
        fields = set(BenchmarkConfig.model_fields)
        return BenchmarkConfig.model_validate(
            {k: v for k, v in self.model_dump().items() if k in fields}
        )


TASKS: Registry[TaskSpec] = Registry("task")
register_task = TASKS.register
