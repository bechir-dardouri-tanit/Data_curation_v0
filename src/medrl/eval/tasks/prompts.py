"""Prompt builders -- the single place the output contract is phrased.

The contract string appears verbatim in the system prompt, in the guided-decoding grammar,
and in the parser's expectations. It is defined once here so the three cannot drift: a
model told to ``Answer: B`` and constrained to ``Answer: [A-E]`` (the task's alphabet,
A-J for MMLU-Pro) and parsed for exactly that marker has no room to be mis-scored on
formatting.
"""

from __future__ import annotations

from collections.abc import Sequence

MCQA_SYSTEM = (
    "You are a medical expert. Reason carefully, then give your final answer as a single "
    "line in the exact form 'Answer: <LETTER>' where <LETTER> is one of the option letters. "
    "The response must end with that line. "
    "Examples of correct format: 'Answer: B', 'Answer: C', 'Answer: (B)', 'Answer: [C]'. "
    "Do NOT write: 'The answer is B' or 'I choose option B' or 'Option B is correct'. "
    "Only use the exact format: Answer: <LETTER>."
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

def mcqa_grammar(letters: str = "ABCDE") -> str:
    """The guided-decoding grammar for an MCQA answer alphabet.

    Handed to vLLM guided decoding (xgrammar/outlines); ``letters`` is the task's
    alphabet -- ``ABCDE`` by default, ``ABCDEFGHIJ`` for MMLU-Pro-style 10-option
    tasks. Constraining to the wrong alphabet doesn't just risk extraction failure,
    it *forces* a wrong answer for items whose gold is outside the range.
    """
    return f"Answer: [{letters[0]}-{letters[-1]}]"


MCQA_GRAMMAR = mcqa_grammar()
"""The default-alphabet grammar (A-E); mirrors the CONTRACT parser's expectations."""


def build_mcqa_user(
    question: str,
    options: Sequence[str | tuple[str, str]],
    *,
    letter_start: str = "A",
) -> str:
    """Render an MCQA item. Options are strings or ``(letter, text)`` pairs.

    Auto-numbered options are lettered from ``letter_start`` (a single uppercase
    letter) onward -- datasets whose options are already labelled elsewhere in the
    item sometimes restart the alphabet mid-way.
    """
    start = letter_start.strip().upper()
    if len(start) != 1 or not "A" <= start <= "Z":
        raise ValueError(f"letter_start must be one uppercase letter, got {letter_start!r}")
    lines = [question.strip(), ""]
    for i, option in enumerate(options):
        letter = option[0] if isinstance(option, tuple) else chr(ord(start) + i)
        text = option[1] if isinstance(option, tuple) else option
        lines.append(f"{letter}. {text}")
    lines.append("")
    lines.append("Format requirements:")
    lines.append("- Your final answer must be on its own line as 'Answer: <LETTER>'")
    lines.append("- <LETTER> must be one of the option letters above")
    lines.append("- Examples: Answer: A, Answer: B, Answer: C")
    lines.append("- Do NOT use other formats like 'The answer is B' or 'I choose B'")
    lines.append("")
    lines.append("Your final answer:")
    return "\n".join(lines)


def build_open_user(question: str, *, history: Sequence[dict[str, str]] = ()) -> str:
    """Render an open-ended turn, with prior conversation turns if present."""
    parts = [f"[{m['role']}]: {m['content']}" for m in history]
    parts.append(f"[user]: {question.strip()}")
    return "\n\n".join(parts)
