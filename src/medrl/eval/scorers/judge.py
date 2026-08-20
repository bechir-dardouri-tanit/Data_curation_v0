"""LLM-judge abstraction for rubric grading.

The judge grades a conversation against a list of weighted binary criteria (HealthBench
semantics). Because a judge is itself an unreliable measurement instrument, reliability
engineering is layered on top in :mod:`medrl.eval.scorers.rubric` (majority vote,
position swap); this module only defines the interface, a deterministic reference
implementation for tests, and the LLM client with defensive parsing.

The same code path grades evaluation and RL rewards -- if the judge prompt drifted between
the two, the policy would be optimized against a different target than the one reported.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from medrl.core.logging import get_logger

log = get_logger(__name__)


class JudgeError(RuntimeError):
    """The judge produced unparseable output after a retry."""


@dataclass(frozen=True)
class Criterion:
    """One binary rubric criterion.

    ``weight`` is signed HealthBench semantics: positive criteria pay when met, negative
    criteria (harmful-response penalties) *subtract* when met. Zero is rejected -- a
    criterion that cannot change a score is a rubric bug, not a neutral entry.
    """

    id: str
    text: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight == 0:
            raise ValueError(f"criterion {self.id!r} weight must be nonzero")


@dataclass(frozen=True)
class CriterionVerdict:
    id: str
    met: bool
    weight: float
    rationale: str | None = None


@dataclass(frozen=True)
class RubricScore:
    """HealthBench-style score for one conversation."""

    met_fraction: float
    verdicts: tuple[CriterionVerdict, ...]
    n_met: int
    n_total: int
    weighted: bool = True
    flip_count: int = 0
    """Criteria whose verdict changed under criterion reordering; a judge-reliability probe."""

    @property
    def points(self) -> float:
        return self.met_fraction * 100.0


class Judge(Protocol):
    def grade(self, messages: list[dict[str, str]], criteria: Sequence[Criterion]) -> tuple[CriterionVerdict, ...]:
        ...


@dataclass
class DeterministicJudge:
    """Reference judge: no model, grades by convention on criterion text.

    A criterion whose text starts with ``+`` is met iff the remainder appears as a substring
    of the final assistant message; ``-`` requires it absent; anything else is met iff any
    word of its text appears. It exists so tests and CI exercise the full scoring path --
    including the reliability layers -- without a GPU, and so downstream code can never
    treat "no judge configured" as "all criteria met".
    """

    def grade(
        self, messages: list[dict[str, str]], criteria: Sequence[Criterion]
    ) -> tuple[CriterionVerdict, ...]:
        assistant = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), ""
        ).casefold()
        verdicts: list[CriterionVerdict] = []
        for criterion in criteria:
            text = criterion.text.strip()
            if text.startswith("+"):
                required = text[1:].strip().casefold()
                met = required in assistant
            elif text.startswith("-"):
                forbidden = text[1:].strip().casefold()
                met = forbidden not in assistant
            else:
                words = [w for w in re.findall(r"\w+", text.casefold()) if len(w) > 3]
                met = any(w in assistant for w in words) if words else False
            verdicts.append(
                CriterionVerdict(id=criterion.id, met=met, weight=criterion.weight)
            )
        return tuple(verdicts)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _parse_met_ids(raw: str) -> set[str] | None:
    """Extract the set of met criterion ids from a judge reply, tolerating decoration."""
    # Try the whole reply, then the first {...} block inside it; judges decorate JSON
    # with prose and fences far more often than they emit malformed JSON itself.
    attempts = [raw]
    match = _JSON_OBJECT.search(raw)
    if match:
        attempts.append(match.group(0))
    for attempt in attempts:
        with contextlib.suppress(json.JSONDecodeError):
            obj = json.loads(attempt)
            if isinstance(obj, dict) and isinstance(obj.get("met"), list):
                return {str(x) for x in obj["met"]}
    return None


_PROMPT = """You are grading a medical AI assistant's response against binary criteria.

Conversation:
{conversation}

Criteria (grade each one independently):
{criteria}

A criterion is met only if the response clearly satisfies it. Judge the response alone,
not what it could have said. Respond with JSON only, of the exact form:
{{"met": ["<id>", ...]}}
listing the ids of every met criterion. Omitted ids count as not met."""


@dataclass
class LLMJudge:
    """Judge backed by any OpenAI-compatible chat endpoint (vLLM serve, or an API)."""

    model: str
    client: Any = None
    temperature: float = 0.0
    max_concurrency: int = 32
    # The judge's reply is a tiny JSON object; an unbounded generation lets a
    # thinking-capable judge ruminate until truncation and never emit it. 512 is
    # orders of magnitude above the largest legitimate verdict.
    max_tokens: int = 512
    # vLLM-ism the runner sets for thinking-capable judges (the Qwen3.5 family
    # this project uses for both tiers): without it the model's reasoning is
    # emitted into content ahead of the JSON, and with no server-side reasoning
    # parser the contract breaks on every call.
    extra_body: dict[str, Any] | None = None
    system: str = "You are a precise rubric grader. Respond with JSON only."

    def _render(self, messages: list[dict[str, str]], criteria: Sequence[Criterion]) -> str:
        conversation = "\n".join(f"[{m['role']}]: {m['content']}" for m in messages)
        listed = "\n".join(f"{i+1}. id={c.id!r}: {c.text}" for i, c in enumerate(criteria))
        return _PROMPT.format(conversation=conversation, criteria=listed)

    def _complete(self, user_prompt: str, *, retry_note: str = "") -> str:
        if self.client is None:
            raise JudgeError("LLMJudge.client is not configured")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system + retry_note},
                {"role": "user", "content": user_prompt},
            ],
            **({"extra_body": self.extra_body} if self.extra_body else {}),
        )
        return str(response.choices[0].message.content or "")

    def grade(
        self, messages: list[dict[str, str]], criteria: Sequence[Criterion]
    ) -> tuple[CriterionVerdict, ...]:
        prompt = self._render(messages, criteria)
        raw = self._complete(prompt)
        met_ids = _parse_met_ids(raw)
        if met_ids is None:
            # One retry with an explicit nudge; judges recover almost always on the second try.
            raw = self._complete(prompt, retry_note=" Respond with JSON only.")
            met_ids = _parse_met_ids(raw)
        if met_ids is None:
            raise JudgeError(f"judge output unparseable after retry: {raw[:200]!r}")

        known = {c.id for c in criteria}
        unknown = met_ids - known
        if unknown:
            log.warning("judge invented %d unknown criterion ids; ignoring", len(unknown))
        met_ids &= known

        return tuple(
            CriterionVerdict(
                id=c.id,
                met=c.id in met_ids,
                weight=c.weight,
                rationale=None if c.id in met_ids else "not listed as met",
            )
            for c in criteria
        )
