"""Numeric verification and quantity parsing.

Numeric medical answers (MedCalc-style) are graded by tolerance, not equality: gold
values are derived through multi-step calculations where a legitimately rounded
intermediate shifts the last digit. The default ``rtol=0.005`` absorbs that while still
failing any wrong-digit answer; tighten or loosen it per benchmark rather than editing
this module.

Verification is total: ``None`` predictions, NaN and infinities all return ``False``
rather than raising, since they arrive straight from :func:`extract_number` on whatever
the rollout produced.
"""

from __future__ import annotations

import math
import re

from medrl.eval.extraction import scan_numbers

# Trailing unit of a quantity: an optional single space then unit characters.
# Deliberately permissive (letters, %, µ, °, /, parentheses, middle dot) because curating
# a unit ontology is the MedCalc task's job, not the parser's.
_UNIT_RE = re.compile(r"\s?([A-Za-z%\u00b5\u03bc\u00b0][A-Za-z%\u00b5\u03bc\u00b0/()\u00b7\-]*)")


def verify_number(
    pred: float | None,
    gold: float,
    rtol: float = 0.005,
    atol: float = 1e-8,
) -> bool:
    """Tolerant equality, ``math.isclose`` semantics; ``pred=None`` or non-finite is False.

    ``atol`` exists so a gold of exactly 0 is gradeable: relative tolerance alone is a
    pass/fail cliff around zero (any nonzero prediction fails, however tiny).
    """
    if pred is None:
        return False
    return math.isclose(pred, gold, rel_tol=rtol, abs_tol=atol)


def parse_quantity(text: str) -> tuple[float, str] | None:
    """Parse ``"5.2 mmol/L"`` into ``(5.2, "mmol/L")`` -- the LAST number in ``text``.

    Deliberately *no unit conversion, and no canonicalization either* (µ vs u, mL vs ml,
    case): the MedCalc task normalizes units in its gold, and duplicating that table here
    would give the repo two places to disagree about units. Compare like with like after
    the task's normalization, not this parser's guess.

    Known failure mode, accepted on purpose: the unit is whatever token follows the
    number, so on free prose ("a level of 5.2 indicates ...") the next word is
    indistinguishable from a unit without an ontology. Call this on short answer spans,
    not whole completions. Returns ``None`` when no unambiguous number exists.
    """
    numbers = scan_numbers(text)
    if not numbers:
        return None
    (_, end), value = numbers[-1]
    m = _UNIT_RE.match(text, end)
    unit = m.group(1).rstrip(".") if m else ""
    return value, unit
