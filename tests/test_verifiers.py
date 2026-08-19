"""Tests for the ground-truth verifiers and the config-addressed format rules.

Bad-params coverage matters as much as happy paths here: a format rule that fails
silently vacuous would pay RL reward for nothing, which is the exact skew the rules
exist to prevent.
"""

from __future__ import annotations

import math

import pytest

from medrl.eval.verifiers import FORMAT_RULES, parse_quantity, verify_letter, verify_number
from medrl.eval.verifiers.format_rules import FormatRule

PROSE = "The patient should be monitored closely for signs of deterioration over 24 hours."


# ---------------------------------------------------------------------------------------
# letters
# ---------------------------------------------------------------------------------------


def test_verify_letter_tolerates_decorations_on_either_side() -> None:
    assert verify_letter("B", "B")
    assert verify_letter("b", "B")
    assert verify_letter("(B)", "B")
    assert verify_letter("B.", "b")
    assert verify_letter("B)", "(b)")
    assert verify_letter("**B**", "B")


def test_verify_letter_rejects() -> None:
    assert not verify_letter("A", "B")
    assert not verify_letter(None, "B")  # extraction failed
    assert not verify_letter("", "B")
    assert not verify_letter("Beta", "B")  # not a single letter
    assert not verify_letter("B", "Beta")  # malformed gold: False, not an exception
    assert not verify_letter("F", "B")  # outside the A-E alphabet


# ---------------------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------------------


def test_verify_number_tolerance() -> None:
    assert verify_number(5.2, 5.21)  # within default rtol=0.005
    assert not verify_number(5.2, 5.5)
    assert verify_number(100.0, 100.4)
    assert verify_number(0.0, 1e-9)  # atol carries the gold==0 case
    assert not verify_number(None, 5.0)
    assert not verify_number(math.nan, 5.0)
    assert verify_number(5.0, 5.05, rtol=0.02)  # tolerance is caller-controlled


def test_parse_quantity() -> None:
    assert parse_quantity("5.2 mmol/L") == (5.2, "mmol/L")
    assert parse_quantity("45%") == (45.0, "%")
    assert parse_quantity("7") == (7.0, "")
    assert parse_quantity("1,234 mg") == (1234.0, "mg")
    assert parse_quantity("\u22122.5 kg") == (-2.5, "kg")
    assert parse_quantity("a value of 3.2 mEq/L.") == (3.2, "mEq/L")  # sentence period dropped
    assert parse_quantity("nothing numeric") is None
    assert parse_quantity("1,23") is None  # ambiguous comma, refuse to guess


# ---------------------------------------------------------------------------------------
# format rules: substring and length rules
# ---------------------------------------------------------------------------------------


def _rule(name: str) -> FormatRule:
    return FORMAT_RULES.get(name)


def test_contains_all() -> None:
    rule = _rule("contains_all")
    assert rule("Diagnosis: influenza. Treatment: supportive.", "Diagnosis\nTreatment")
    assert not rule("Diagnosis: influenza only.", "Diagnosis\nTreatment")


def test_contains_none() -> None:
    rule = _rule("contains_none")
    assert rule(PROSE, "aspirin\nibuprofen")
    assert not rule("Start aspirin today.", "aspirin")


def test_word_count_rules() -> None:
    minw = _rule("min_words")
    assert minw("one two three", "3")
    assert not minw("one two", "3")
    maxw = _rule("max_words")
    assert maxw("one two three", "3")
    assert not maxw("one two three four", "3")
    assert maxw("", "0")


def test_exact_n_lines() -> None:
    rule = _rule("exact_n_lines")
    assert rule("line1\nline2", "2")
    assert rule("line1\nline2\n", "2")  # trailing newline adds no line
    assert rule("line1\n\nline2", "3")  # blank lines count
    assert not rule("line1", "2")


def test_is_json() -> None:
    rule = _rule("is_json")
    assert rule('{"answer": "B"}', "")
    assert rule("[1, 2]", "")
    assert not rule("just text", "")
    assert not rule("", "")


def test_no_markdown() -> None:
    rule = _rule("no_markdown")
    assert rule(PROSE, "")
    assert rule("a dose of 5 * 3 mg", "")  # single asterisk is prose, not emphasis
    assert not rule("**Answer: B**", "")
    assert not rule("# Heading\nbody", "")
    assert not rule("```python\nx = 1\n```", "")
    assert not rule("- first\n- second", "")
    assert not rule("> quoted advice", "")
    assert not rule("see [this](http://x.y)", "")


def test_ends_with_answer_letter() -> None:
    rule = _rule("ends_with_answer_letter")
    assert rule("Long reasoning.\nAnswer: B", "")
    assert rule("Long reasoning.\nAnswer: B\n\n", "")
    assert rule("**Answer: B**", "")
    assert not rule("Answer: A\nthen more words", "")  # contract must be the tail
    assert not rule("no contract at all", "")
    assert rule("reasoning\nAnswer: B", "b")
    assert not rule("reasoning\nAnswer: B", "C")


# ---------------------------------------------------------------------------------------
# eager params validation -- every rule names itself in the error
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "params"),
    [
        ("contains_all", ""),
        ("contains_all", " \n \n"),
        ("contains_none", ""),
        ("min_words", "zero"),
        ("min_words", "-1"),
        ("min_words", "0"),  # vacuous floor is a config bug
        ("max_words", "1.5"),
        ("max_words", "x"),
        ("exact_n_lines", "two"),
        ("exact_n_lines", "-2"),
        ("is_json", "unexpected"),
        ("no_markdown", "unexpected"),
        ("ends_with_answer_letter", "Z"),
        ("ends_with_answer_letter", "BE"),
    ],
)
def test_bad_params_raise_with_rule_name(rule: str, params: str) -> None:
    with pytest.raises(ValueError, match=rule):
        _rule(rule)("some response text", params)


def test_registry_contents_and_lookup_errors() -> None:
    expected = {
        "contains_all",
        "contains_none",
        "min_words",
        "max_words",
        "exact_n_lines",
        "is_json",
        "no_markdown",
        "ends_with_answer_letter",
    }
    assert set(FORMAT_RULES.names()) == expected
    assert len(FORMAT_RULES) == len(expected)
    with pytest.raises(KeyError, match="contains_all"):  # did-you-mean hint
        FORMAT_RULES.get("contains_al")
