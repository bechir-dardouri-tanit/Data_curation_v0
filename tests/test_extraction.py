"""Adversarial tests for answer extraction.

Every case here is a behavior we once got wrong with a regex cascade (or would): nested
braces, self-corrections, decoration, contract violations. Extraction is shared with the
RL reward, so a regression silently changes what training pays for.
"""

from __future__ import annotations

import dataclasses

import pytest

from medrl.eval.extraction import (
    ExtractionPath,
    ExtractionResult,
    extract_boxed,
    extract_mcqa,
    extract_number,
    extraction_stats,
    normalize_letter,
)


def test_contract_basic_and_span() -> None:
    text = "Long reasoning paragraph.\nAnswer: B"
    r = extract_mcqa(text)
    assert r.value == "B"
    assert r.path is ExtractionPath.CONTRACT
    assert r.span is not None
    assert text[r.span[0] : r.span[1]] == "Answer: B"


def test_contract_last_occurrence_wins() -> None:
    r = extract_mcqa("Answer: A ... wait, reconsidering ... Answer: C")
    assert r.value == "C"


def test_contract_tolerates_markdown_quotes_and_parens() -> None:
    assert extract_mcqa("**Answer: B**").value == "B"
    assert extract_mcqa("Final: \"Answer: '(C)'\"").value == "C"
    assert extract_mcqa("Answer: (d) some option text").value == "D"


def test_contract_marker_is_case_sensitive() -> None:
    # "ANSWER:" mid-prose must not fish a letter out of the reasoning block.
    assert extract_mcqa("ANSWER: B").path is ExtractionPath.FAILED


def test_contract_ignores_words_starting_with_a_letter() -> None:
    assert extract_mcqa("Answer: Beta blockers").path is ExtractionPath.FAILED


def test_guided_json_with_trailing_prose_and_fence() -> None:
    assert extract_mcqa('Sure! {"answer": "B"} hope that helps').value == "B"
    r = extract_mcqa('```json\n{"answer": "D"}\n```')
    assert r.value == "D"
    assert r.path is ExtractionPath.GUIDED_JSON


def test_guided_json_span_covers_object() -> None:
    text = 'prefix {"reasoning": "...", "answer": "A"} suffix'
    r = extract_mcqa(text)
    assert r.value == "A"
    assert r.span is not None
    assert text[r.span[0] : r.span[1]] == '{"reasoning": "...", "answer": "A"}'


def test_guided_json_nested_and_string_braces() -> None:
    # Wrapper object, a "}" inside a JSON string, and last-object-wins.
    assert extract_mcqa('{"wrapper": {"answer": "C"}}').value == "C"
    assert extract_mcqa('{"answer": "B}"}').value == "B"
    assert extract_mcqa('{"answer": "A"} then {"answer": "E"}').value == "E"


def test_boxed_letter_and_priority() -> None:
    r = extract_mcqa(r"so the answer is $\boxed{C}$ indeed")
    assert r.value == "C"
    assert r.path is ExtractionPath.BOXED
    # Contract outranks a boxed letter that appears later in the text.
    assert extract_mcqa(r"Answer: A ... $\boxed{C}$").value == "A"
    # Guided JSON outranks boxed.
    assert extract_mcqa(r'{"answer": "D"} $\boxed{E}$').value == "D"


def test_boxed_nested_braces_do_not_terminate_match() -> None:
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{{x^2}}") == "{x^2}"
    # ...and a nested-brace box is not a letter, so MCQA falls through.
    assert extract_mcqa(r"answer \boxed{{x^2}}").path is ExtractionPath.FAILED


def test_boxed_last_occurrence_and_missing() -> None:
    assert extract_boxed(r"first \boxed{B} then \boxed{E}") == "E"
    assert extract_boxed("no latex here") is None
    assert extract_boxed(r"unterminated \boxed{3.2") is None


def test_last_line_variants() -> None:
    assert extract_mcqa("reasoning...\n\nmore reasoning\nB").value == "B"
    r = extract_mcqa("reasoning...\n\n(c)")
    assert r.value == "C"
    assert r.path is ExtractionPath.LAST_LINE
    # LAST_LINE never outranks an earlier contract hit.
    assert extract_mcqa("Answer: A\nB").value == "A"


def test_last_line_takes_the_last_occurrence() -> None:
    # Regression: the loop used to return the FIRST bare-letter line, contradicting the
    # documented "within a path the last occurrence wins" -- a model enumerating options
    # as "A" before committing to "B" was scored as A.
    r = extract_mcqa("A\nbecause the labs rule out the others\n\nB")
    assert (r.value, r.path) == ("B", ExtractionPath.LAST_LINE)


def test_french_guillemets_are_decoration() -> None:
    # Regression: "Answer: «B»" failed extraction outright on the FR benchmarks.
    r = extract_mcqa("Réponse très claire.\n\nAnswer: «B»")
    assert (r.value, r.path) == ("B", ExtractionPath.CONTRACT)
    bare = extract_mcqa("text\n«C»")
    assert (bare.value, bare.path) == ("C", ExtractionPath.LAST_LINE)


def test_answer_before_thinking_block_is_still_found() -> None:
    text = "Answer: B\n</think>\nActually, let me reconsider the labs.\nNo final marker."
    assert extract_mcqa(text).value == "B"
    # ...but prose-only answers inside/after thinking fail measurably.
    assert extract_mcqa("hidden reasoning; the answer is C.").path is ExtractionPath.FAILED


def test_failure_modes_never_raise() -> None:
    for bad in [
        "",
        "Answer:",
        "Answer: Z",
        "Answer: 42",
        "\\boxed{",
        "{",
        "\\u0000\\udfffnonsense",
        "{" * 3000 + "}" * 3000,  # json.loads hits RecursionError internally
        "„❤️ \U0001f9ec",
    ]:
        r = extract_mcqa(bad)
        assert isinstance(r, ExtractionResult)
        assert r.value is None or r.value in "ABCDE"
        assert extract_number(bad) is None or isinstance(extract_number(bad), float)
        assert extract_boxed(bad) is None or isinstance(extract_boxed(bad), str)


def test_result_is_frozen() -> None:
    r = extract_mcqa("Answer: B")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.value = "C"  # type: ignore[misc]


def test_normalize_letter_table() -> None:
    assert normalize_letter("B") == "B"
    assert normalize_letter("b") == "B"
    assert normalize_letter("(B)") == "B"
    assert normalize_letter("B.") == "B"
    assert normalize_letter("**B**") == "B"
    assert normalize_letter("Beta") is None
    assert normalize_letter("A-E") is None
    assert normalize_letter("") is None
    assert normalize_letter(None) is None  # type: ignore[arg-type]


def test_extract_number_units_commas_minus_percent() -> None:
    assert extract_number("5.2 mmol/L") == 5.2
    assert extract_number("1,234.5") == 1234.5
    assert extract_number("\u22123.5 mV") == -3.5
    assert extract_number("45%") == 45.0
    assert extract_number("1.5e3") == 1500.0
    assert extract_number("the dose is 3.") == 3.0


def test_extract_number_last_literal_and_boxed_priority() -> None:
    assert extract_number("between 12 and 34 mg") == 34.0
    assert extract_number(r"earlier value 3, final \boxed{7.5}") == 7.5
    # A box holding several numbers is ambiguous: fall through to the text scan.
    assert extract_number(r"\boxed{3 or 4} then 5") == 5.0


def test_extract_number_ambiguous_returns_none() -> None:
    # "1,23" could be a malformed thousands group or a decimal comma: refuse to guess.
    assert extract_number("1,23") is None
    assert extract_number("1,2345,678") is None
    assert extract_number("no digits at all") is None
    # Digits inside identifiers are not numbers.
    assert extract_number("receptor B2 and v1.2") is None


def test_extraction_stats_math() -> None:
    results = [
        extract_mcqa("Answer: B"),
        extract_mcqa("no idea"),
        extract_mcqa(r"\boxed{C}"),
        extract_mcqa('{"answer": "A"}'),
        None,
    ]
    stats = extraction_stats(results)
    assert stats.n == 5
    assert stats.n_failed == 2  # the failure plus the missing completion
    assert stats.fail_rate == 0.4
    assert stats.by_path == {
        "contract": 1,
        "guided_json": 1,
        "boxed": 1,
        "last_line": 0,
        "failed": 2,
    }
    assert sum(stats.by_path.values()) == stats.n


def test_extraction_stats_empty() -> None:
    stats = extraction_stats([])
    assert stats.n == 0
    assert stats.n_failed == 0
    assert stats.fail_rate == 0.0
