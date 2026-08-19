"""Config-addressed format rules: the reward-side guardrails.

During RL the policy optimizes whatever the reward measures, and an unconstrained
completion format is one of the cheapest things to degrade into (sprawl, markdown,
answer buried mid-prose). These rules let a YAML reward/eval config compose format
checks by name. They are deliberately *string-parameterled* ``(response, params) -> bool``
so a rule is one registry entry with no Python side file, and deliberately total --
a rule's verdict is part of the reward signal and must never raise on odd model output.

Params are validated eagerly and loudly: a rule whose params are empty or malformed
would otherwise pass or fail vacuously and silently pay reward for nothing -- the exact
failure the guardrail exists to prevent. Every error names its rule so the offending
YAML key is obvious.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from medrl.core.registry import Registry
from medrl.eval.extraction import DECOR_CLOSE, DECOR_OPEN, normalize_letter

FormatRule = Callable[[str, str], bool]

FORMAT_RULES: Registry[FormatRule] = Registry[FormatRule]("format_rule")

# Anchored form of the MCQA output contract: the response must *end* with "Answer: X"
# (decoration tolerated), because a contract line buried mid-text has already failed the
# point of the contract -- a parser must be able to read the answer off the tail.
_ENDS_ANSWER_RE = re.compile(
    r"Answer:\s*["
    + re.escape(DECOR_OPEN)
    + r"]{0,4}\(?([A-Ea-e])\)?["
    + re.escape(DECOR_CLOSE)
    + r"]{0,4}\s*\Z"
)

# Unambiguous CommonMark markers only. Single asterisks/underscores are excluded on
# purpose: "5 * 3 mg" and snake_case words are plain text, and a format rule that fires
# on prose is worse than one that misses exotic markup.
_MARKDOWN_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"```"),  # fenced code block
    re.compile(r"~~~"),  # fenced code block (tilde form)
    re.compile(r"^#{1,6}(\s|$)", re.MULTILINE),  # ATX heading (7+ hashes is not one)
    re.compile(r"\*\*[^*\n]+\*\*"),  # bold
    re.compile(r"__[^_\n]+__"),  # bold (underscore form)
    re.compile(r"(?<!`)`[^`\n]+`(?!`)"),  # inline code span
    re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"),  # link
    re.compile(r"^(\*\s|-\s|\+\s|\d+\.\s)", re.MULTILINE),  # list marker
    re.compile(r"^>\s", re.MULTILINE),  # blockquote
)


def _substring_params(params: str, rule: str) -> list[str]:
    """Newline-separated substrings; an empty list is a config bug, not a no-op."""
    items = [line.strip() for line in params.splitlines() if line.strip()]
    if not items:
        raise ValueError(
            f"{rule}: params must be a newline-separated list of substrings, got {params!r}"
        )
    return items


def _int_param(params: str, rule: str, minimum: int) -> int:
    try:
        value = int(params.strip())
    except ValueError:
        raise ValueError(f"{rule}: params must be an integer, got {params!r}") from None
    if value < minimum:
        raise ValueError(f"{rule}: params must be >= {minimum}, got {value}")
    return value


def _no_params(params: str, rule: str) -> None:
    if params.strip():
        raise ValueError(f"{rule}: takes no params, got {params!r}")


def _contains_all(response: str, params: str) -> bool:
    """Every newline-separated substring appears in the response (e.g. required findings)."""
    return all(required in response for required in _substring_params(params, "contains_all"))


def _contains_none(response: str, params: str) -> bool:
    """No newline-separated forbidden substring appears (e.g. disclaimed advice terms)."""
    return not any(
        forbidden in response for forbidden in _substring_params(params, "contains_none")
    )


def _min_words(response: str, params: str) -> bool:
    """``len(response.split()) >= n``. Minimum 1: a 0 floor is vacuous and a config bug."""
    return len(response.split()) >= _int_param(params, "min_words", 1)


def _max_words(response: str, params: str) -> bool:
    """``len(response.split()) <= n``. 0 is allowed and means "empty response only"."""
    return len(response.split()) <= _int_param(params, "max_words", 0)


def _exact_n_lines(response: str, params: str) -> bool:
    """Line count equals ``n`` (blank lines count; a trailing newline adds no line)."""
    return len(response.splitlines()) == _int_param(params, "exact_n_lines", 0)


def _is_json(response: str, params: str) -> bool:
    """The whole response parses as JSON -- the guided-decoding contract, checked."""
    _no_params(params, "is_json")
    try:
        json.loads(response)
    except (ValueError, RecursionError):
        return False
    return True


def _no_markdown(response: str, params: str) -> bool:
    """None of the unambiguous markdown markers above appear (plain-prose contract)."""
    _no_params(params, "no_markdown")
    return not any(marker.search(response) for marker in _MARKDOWN_MARKERS)


def _ends_with_answer_letter(response: str, params: str) -> bool:
    """The response ends with the ``Answer: X`` contract line.

    Params empty means "any letter" (pure format check); a specific letter (``"B"``,
    case-insensitive) additionally pins the value, which is how a gold-aware reward uses
    this rule without a separate correctness checker.
    """
    if params.strip():
        want = normalize_letter(params)
        if want is None:
            raise ValueError(
                f"ends_with_answer_letter: params must be empty or a single letter A-E, "
                f"got {params!r}"
            )
    else:
        want = None
    m = _ENDS_ANSWER_RE.search(response)
    if m is None:
        return False
    return want is None or normalize_letter(m.group(1)) == want


FORMAT_RULES.register("contains_all", _contains_all)
FORMAT_RULES.register("contains_none", _contains_none)
FORMAT_RULES.register("min_words", _min_words)
FORMAT_RULES.register("max_words", _max_words)
FORMAT_RULES.register("exact_n_lines", _exact_n_lines)
FORMAT_RULES.register("is_json", _is_json)
FORMAT_RULES.register("no_markdown", _no_markdown)
FORMAT_RULES.register("ends_with_answer_letter", _ends_with_answer_letter)
