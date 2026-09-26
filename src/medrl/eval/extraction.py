"""Answer extraction from raw model completions.

This module is the *only* place an answer is pulled out of a completion, and it is shared
verbatim between offline evaluation and the RL reward computation. Duplicating the parser
on the reward side is how a run ends up rewarding a format the eval never scored, so
everything here is pure, stdlib-only and total: no exception ever escapes on malformed
input (inside a rollout worker an exception kills the episode and silently skews the
reward distribution), and nothing is logged -- callers measure the failure rate themselves
via :func:`extraction_stats` and compare it against ``EvalConfig.max_extraction_fail_rate``
(a spike there is a harness bug, not a model result).

The design is a fixed priority ladder (contract, guided JSON, boxed, last line) rather
than the regex cascade it replaces. First hit wins *across* paths, because the paths are
ordered by how strongly the format was imposed on the model; within one path the LAST
occurrence wins, because models self-correct ("Answer: A ... wait ... Answer: C") and the
final commitment is the one to score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any

# Decoration a model may wrap around the answer letter ("**B**", '"B"', "(B)", "[B]").
# Kept as data (and public) so every letter-facing regex here and in
# :mod:`medrl.eval.verifiers.format_rules` shares one vocabulary by construction
# instead of by coincidence. Curly quotes and guillemets are escaped because ruff
# flags the raw glyphs as ambiguous look-alikes; an escape is cheaper than a lint
# exemption. Guillemets matter for real: French models write "Answer: \u00abB\u00bb".
_MARKDOWN_DECOR = "*`_~"
DECOR_OPEN = "\"'\u201c\u2018\u00ab(<[" + _MARKDOWN_DECOR
DECOR_CLOSE = _MARKDOWN_DECOR + "\"'\u201d\u2019\u00bb.)>]:,!"
_LETTER_DECOR = DECOR_OPEN + DECOR_CLOSE + "!{}"

# Case-insensitive Answer: marker to handle models that write "answer:", "ANSWER:", "Answer:"
# etc. Robustness over strictness - we want to extract correct answers even when format
# varies, as long as we can reliably identify the intent.
_ANSWER_RE = re.compile(
    r"(?i:Answer):\s*[" + re.escape(DECOR_OPEN) + r"]{0,4}\(?([A-Ea-e])\)?(?![A-Za-z0-9])"
)

# Prose answer patterns for conversational models that don't use "Answer:" markers
# Captures phrases like "the answer is B", "I choose option C", etc.
_PROSE_ANSWER_RE = re.compile(
    r"(?:the\s+)?answer\s+is\s+(?:option\s+)?([A-Ea-e])"
    r"|(?:i\s+)?(?:would\s+)?choose\s+(?:option\s+)?([A-Ea-e])"
    r"|select\s+(?:option\s+)?([A-Ea-e])"
    r"|correct\s+option\s+is\s+(?:option\s+)?([A-Ea-e])",
    re.IGNORECASE | re.MULTILINE
)
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")


def _validate_letters(letters: str) -> str:
    """Alphabets must be one contiguous uppercase range (``ABCDE``, ``ABCDEFGHIJ``).

    The letter-facing regexes are built as ``[first-last]`` classes, and a task
    misconfigured with a gappy alphabet would silently never match the gaps' answers.
    """
    upper = letters.upper()
    if not upper or not ("A" <= upper[0] <= "Z" and "A" <= upper[-1] <= "Z"):
        raise ValueError(f"letters must be uppercase A-Z, got {letters!r}")
    if ord(upper[-1]) - ord(upper[0]) + 1 != len(upper):
        raise ValueError(f"letters must be contiguous, got {letters!r}")
    return upper


@lru_cache(maxsize=16)
def _answer_re(letters: str) -> re.Pattern[str]:
    """The CONTRACT marker regex for a non-default alphabet (default uses _ANSWER_RE).

    Mirrors ``_ANSWER_RE``'s shape; the only difference is the letter class.
    Case-insensitive for robustness.
    """
    return re.compile(
        r"(?i:Answer):\s*[" + re.escape(DECOR_OPEN) + r"]{0,4}\(?(["
        + letters[0]
        + "-"
        + letters[-1]
        + letters[0].lower()
        + "-"
        + letters[-1].lower()
        + "])\)?(?![A-Za-z0-9])"
    )

# A numeric literal is accepted only when its punctuation is unambiguous. The trailing
# lookahead rejects a digit or a comma-followed-by-digit, so "1,234" matches as one
# grouped token while "1,23" (comma as European decimal separator? thousands group gone
# wrong?) matches nothing at all -- guessing a locale would score a wrong number, so
# ambiguity must fail. The leading lookbehinds keep digits inside identifiers ("B2",
# "v1.2") and restarts after a malformed comma group ("1,2345,678") out of the results.
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?<!\d,)"
    r"[-+\u2212\uFE63]?"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.?\d+)"
    r"(?:[eE][-+\u2212\uFE63]?\d+)?"
    r"(?!\d)(?!,\d)"
)
# Trailing unit of a quantity: an optional single space then unit characters. Deliberately
# permissive (letters, %, µ, °, /, parentheses, middle dot) because curating a unit
# ontology is the MedCalc task's job, not the parser's.
_UNIT_RE = re.compile(r"\s?([A-Za-z%\u00b5\u03bc\u00b0][A-Za-z%\u00b5\u03bc\u00b0/()\u00b7\-]*)")

_LETTERS = "ABCDE"


class ExtractionPath(StrEnum):
    """Which rung of the priority ladder produced a value.

    The path is reported next to the failure rate because "extraction failed" and
    "extraction only survived via the last-line heuristic" are different diagnoses:
    the former means the contract is broken, the latter that it is merely loose.
    """

    CONTRACT = "contract"
    GUIDED_JSON = "guided_json"
    BOXED = "boxed"
    PROSE = "prose"
    LAST_LINE = "last_line"
    FAILED = "failed"


@dataclass(frozen=True)
class ExtractionResult:
    """One extracted answer.

    ``span`` is the character range of the *syntax the value was read from* (the full
    "Answer: B" match, the whole JSON object, the whole ``\boxed{...}`` construct, the
    trimmed line) -- not of the letter itself -- so a human debugging a miss can jump
    straight to the evidence. It is ``None`` exactly when ``value`` is.
    """

    value: str | None
    path: ExtractionPath
    span: tuple[int, int] | None


@dataclass(frozen=True)
class ExtractionStats:
    """Aggregate over many results, keyed by path.

    ``by_path`` always contains every path (zeros included) so reports have stable
    columns regardless of what a particular sample actually exercised.
    """

    n: int
    n_failed: int
    fail_rate: float
    by_path: dict[str, int]


def normalize_letter(token: str | None, letters: str = _LETTERS) -> str | None:
    """Reduce any decorated single-letter answer to a bare uppercase member of ``letters``.

    This is the one canonical alphabet normalizer, shared with
    :func:`medrl.eval.verifiers.letters.verify_letter`, so eval scoring and RL rewards
    cannot disagree about whether "(b)" equals "B". Anything that is not exactly one
    letter after stripping decoration ("Beta", "A-E", "", ``None``) returns ``None`` --
    callers treat that as "no answer", never as an error. ``letters`` is the task's
    answer alphabet (A-E default; A-J for MMLU-Pro-style 10-option tasks).
    """
    if not token:
        return None
    stripped = token.strip().strip(_LETTER_DECOR).strip()
    if len(stripped) == 1:
        upper = stripped.upper()
        if upper in letters:
            return upper
    return None


def _json_object_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` of every brace-closed, string-aware JSON object in ``text``.

    Scanning is string-aware because a ``"}"`` inside a JSON string must not close the
    object. Every ``{`` is tried (not just top-level ones) so an answer object nested
    inside a wrapper object is still found; that makes the worst case quadratic in the
    text length, which is acceptable for completion-sized inputs with shallow schemas.
    Objects that never close (truncated generation) are skipped.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            escaped = False
            j = i
            while j < n:
                char = text[j]
                if in_str:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_str = False
                elif char == '"':
                    in_str = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        yield (i, j + 1)
                        break
                j += 1
        i += 1


def _boxed_spans(text: str) -> list[tuple[int, int, int, int]]:
    r"""Return ``(start, content_start, content_end, end)`` for each ``\boxed{...}``.

    Braces are matched with a counter, never a regex, because boxed content nests
    (``\boxed{\frac{1}{2}}``) and a regex that stops at the first ``}`` silently returns
    half a formula. A backslash escapes the next character, so literal ``\{``/``\}``
    do not participate in the count. Unterminated boxes (truncated generation) are
    dropped rather than guessed at.
    """
    spans: list[tuple[int, int, int, int]] = []
    for m in _BOXED_OPEN_RE.finditer(text):
        depth = 1
        j = m.end()
        content_end = -1
        while j < len(text):
            char = text[j]
            if char == "\\":
                j += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    content_end = j
                    break
            j += 1
        if content_end >= 0:
            spans.append((m.start(), m.end(), content_end, content_end + 1))
    return spans


def scan_numbers(text: str) -> list[tuple[tuple[int, int], float]]:
    """All unambiguous numeric literals in ``text``, left to right, as ``(span, value)``.

    Shared with :func:`medrl.eval.verifiers.numbers.parse_quantity` so quantity parsing
    and number extraction can never drift apart on commas, unicode minus or exponents.
    """
    found: list[tuple[tuple[int, int], float]] = []
    for m in _NUMBER_RE.finditer(text):
        token = m.group().replace("\u2212", "-").replace("\ufe63", "-").replace(",", "")
        try:
            found.append(((m.start(), m.end()), float(token)))
        except ValueError:  # pragma: no cover - the regex shape makes this unreachable
            continue
    return found


def extract_mcqa(text: str, letters: str = _LETTERS) -> ExtractionResult:
    """Extract a single choice in ``letters`` from a completion. Total: never raises on
    model output (an invalid ``letters`` is a config bug and raises).

    Priority order, first hit wins: CONTRACT (the prompted ``Answer: X`` marker) >
    GUIDED_JSON (a ``{"answer": "B"}`` object from guided decoding, even with trailing
    prose or a ```json fence around it) > BOXED (a letter inside ``\boxed{}``) >
    PROSE (conversational phrases like "the answer is B") >
    LAST_LINE (a bare final letter, the loosest signal, kept last so it cannot shadow a
    stronger one that appeared *earlier* in the text). Within a path the last occurrence
    wins. Empty/``None``-ish input returns FAILED rather than raising, because reward
    code calls this on whatever the rollout produced.
    """
    if not text:
        return ExtractionResult(value=None, path=ExtractionPath.FAILED, span=None)
    letters = _validate_letters(letters)
    answer_re = _ANSWER_RE if letters == _LETTERS else _answer_re(letters)

    matches = list(answer_re.finditer(text))
    if matches:
        m = matches[-1]
        letter = normalize_letter(m.group(1), letters)
        if letter is not None:
            return ExtractionResult(letter, ExtractionPath.CONTRACT, (m.start(), m.end()))

    json_result: ExtractionResult | None = None
    for start, end in _json_object_spans(text):
        obj: Any
        try:
            obj = json.loads(text[start:end])
        except (ValueError, RecursionError):
            continue
        if not isinstance(obj, dict):
            continue
        raw = obj.get("answer")
        if not isinstance(raw, str):
            continue
        letter = normalize_letter(raw, letters)
        if letter is not None:
            json_result = ExtractionResult(letter, ExtractionPath.GUIDED_JSON, (start, end))
    if json_result is not None:
        return json_result

    boxed_result: ExtractionResult | None = None
    for start, content_start, content_end, end in _boxed_spans(text):
        letter = normalize_letter(text[content_start:content_end], letters)
        if letter is not None:
            boxed_result = ExtractionResult(letter, ExtractionPath.BOXED, (start, end))
    if boxed_result is not None:
        return boxed_result

    # PROSE: conversational models that don't use "Answer:" markers but still clearly
    # indicate their choice with phrases like "the answer is B", "I choose option C", etc.
    prose_result: ExtractionResult | None = None
    prose_matches = list(_PROSE_ANSWER_RE.finditer(text))
    if prose_matches:
        # Take the last prose answer (models might change their mind)
        m = prose_matches[-1]
        # Check all groups (different alternatives capture in different groups)
        raw_letter = None
        for i in range(1, 5):  # We have 4 possible groups
            if m.group(i):
                raw_letter = m.group(i)
                break
        if raw_letter:
            letter = normalize_letter(raw_letter, letters)
            if letter is not None:
                prose_result = ExtractionResult(letter, ExtractionPath.PROSE, (m.start(), m.end()))
    if prose_result is not None:
        return prose_result

    # LAST_LINE: the last bare-letter line wins, like every other path -- a model that
    # writes "A" while enumerating options and commits to "B" at the end answered B.
    last_line: ExtractionResult | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            letter = normalize_letter(stripped, letters)
            if letter is not None:
                line_start = offset + (len(line) - len(line.lstrip()))
                last_line = ExtractionResult(
                    letter, ExtractionPath.LAST_LINE, (line_start, line_start + len(stripped))
                )
        offset += len(line)
    if last_line is not None:
        return last_line

    return ExtractionResult(value=None, path=ExtractionPath.FAILED, span=None)


def extract_boxed(text: str) -> str | None:
    """Content of the last brace-matched ``\boxed{...}`` (any content, not just letters).

    Returns the content stripped of surrounding whitespace, so ``\boxed{}`` yields
    ``""`` -- distinct from ``None`` (no box at all). The last box wins because models
    that box several intermediate results put the final answer in the final box.
    """
    if not text:
        return None
    spans = _boxed_spans(text)
    if not spans:
        return None
    _, content_start, content_end, _ = spans[-1]
    return text[content_start:content_end].strip()


def extract_number(text: str) -> float | None:
    """Last unambiguous number in ``text``; boxed content takes priority over prose.

    ``\boxed{7.5}`` beats a "3" mentioned in the reasoning, but only when the box holds
    exactly one number -- a box with several ("3 or 4") is ambiguous and falls through
    to the whole-text scan, which takes the last literal ("last" because final answers
    follow the reasoning that cites earlier candidates). Handles thousands commas,
    unicode minus, exponents and trailing units/percent (the unit is simply not part of
    the match). Returns ``None`` when nothing unambiguous exists, e.g. "1,23", where the
    comma could be a decimal separator and guessing a locale would score a wrong digit.
    """
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        boxed_numbers = scan_numbers(boxed)
        if len(boxed_numbers) == 1:
            return boxed_numbers[0][1]
    numbers = scan_numbers(text)
    if numbers:
        return numbers[-1][1]
    return None


def extraction_stats(results: Iterable[ExtractionResult | None]) -> ExtractionStats:
    """Aggregate extraction outcomes over a sample.

    ``None`` entries (no completion recorded at all -- server error, crash) count as
    FAILED: an absent answer is indistinguishable from an unextractable one for scoring,
    and hiding it would understate the failure rate the run is gated on.
    """
    by_path = {path.value: 0 for path in ExtractionPath}
    n = 0
    for result in results:
        n += 1
        if result is None:
            by_path[ExtractionPath.FAILED.value] += 1
        else:
            by_path[result.path.value] += 1
    n_failed = by_path[ExtractionPath.FAILED.value]
    return ExtractionStats(
        n=n,
        n_failed=n_failed,
        fail_rate=(n_failed / n) if n else 0.0,
        by_path=by_path,
    )
