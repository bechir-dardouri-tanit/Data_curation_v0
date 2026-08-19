"""Prompt builders -- the single place the output contract is phrased.

The contract string appears verbatim in the system prompt, in the guided-decoding grammar,
and in the parser's expectations. It is defined once here so the three cannot drift: a
model told to ``Answer: B`` and constrained to ``Answer: [A-E]`` and parsed for exactly
that marker has no room to be mis-scored on formatting.
"""

from __future__ import annotations

from collections.abc import Sequence

MCQA_SYSTEM = (
    "You are a medical expert. Reason carefully, then give your final answer as a single "
    "line in the exact form 'Answer: <LETTER>' where <LETTER> is one of the option letters. "
    "The response must end with that line."
)
"""The terminal-line contract; mirrors the guided-decoding grammar and the CONTRACT parser."""

OPEN_RUBRIC_SYSTEM = (
    "You are a medical expert answering a patient or clinician question. Be accurate, "
    "complete and appropriately cautious; state uncertainty and escalation guidance where "
    "the situation calls for it."
)

NUMERIC_SYSTEM = (
    "You are a medical expert performing a clinical calculation. Show the computation, "
    "then give the final value on its own last line."
)

MCQA_GRAMMAR = 'Answer: [A-E]'
"""Regex handed to vLLM guided decoding (xgrammar/outlines) for the answer terminal."""


def build_mcqa_user(
    question: str,
    options: Sequence[str | tuple[str, str]],
    *,
    letter_start: str = "A",
) -> str:
    """Render an MCQA item. Options are strings or ``(letter, text)`` pairs."""
    from string import ascii_uppercase

    lines = [question.strip(), ""]
    for i, option in enumerate(options):
        letter = option[0] if isinstance(option, tuple) else ascii_uppercase[i]
        text = option[1] if isinstance(option, tuple) else option
        lines.append(f"{letter}. {text}")
    lines.append("")
    lines.append("End with 'Answer: <LETTER>'.")
    return "\n".join(lines)


def build_open_user(question: str, *, history: Sequence[dict[str, str]] = ()) -> str:
    """Render an open-ended turn, with prior conversation turns if present."""
    parts = [f"[{m['role']}]: {m['content']}" for m in history]
    parts.append(f"[user]: {question.strip()}")
    return "\n\n".join(parts)
