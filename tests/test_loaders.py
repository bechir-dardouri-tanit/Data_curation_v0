"""Loader tests against *real captured rows* from each dataset (2026-08-19).

The row fixtures below are verbatim from the Hub (datasets-server first-rows / direct
file reads), so an upstream schema change breaks a test here instead of a GPU run. Only
the pure mapping layer is tested; fetching is exercised on the serving host.
"""

from __future__ import annotations

import pytest

from medrl.eval.loaders import LoaderError, LoadResult
from medrl.eval.tasks.benchmarks import TASKS
from medrl.eval.tasks.spec import VerifyStyle


def _map(task_name: str, row: dict) -> tuple[object, LoadResult]:
    from medrl.eval import loaders

    spec = TASKS.get(task_name)
    out = LoadResult()
    item = getattr(loaders, f"_{TASKS.get(task_name).loader}_row")(spec, row, out)
    return item, out


# --------------------------------------------------------------------------- MedQA
MEDQA_ROW = {
    "id": "11798523-ae15-4a7d-8e75-5281282aeadf",
    "data": {
        "Question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair",
        "Options": {
            "A": "Disclose the error to the patient and put it in the operative report",
            "B": "Tell the attending that he cannot fail to disclose this mistake",
            "C": "Report the physician to the ethics committee",
            "D": "Disclose the error but leave it out of the operative report",
        },
        "Correct Answer": "Tell the attending that he cannot fail to disclose this mistake",
        "Correct Option": "B",
    },
    "subject_name": "",
}


def test_medqa_maps_options_and_gold() -> None:
    item, out = _map("medqa", MEDQA_ROW)
    assert out.drops == {}
    assert item.verify.gold_letter == "B"
    assert item.verify.letters == "ABCD"
    assert item.retry_grammar == "Answer: [A-D]"
    assert "Answer Choices" not in item.messages[1]["content"]
    assert "D. Disclose the error but leave" in item.messages[1]["content"]


# ------------------------------------------------------------------------- MedMCQA
MEDMCQA_ROW = {
    "id": "45258d3d-b974-44dd-a161-c3fccbdadd88",
    "question": "Which of the following is not true for myelinated nerve fibers:",
    "opa": "Impulse through myelinated fibers is slower",
    "opb": "Membrane currents are generated at nodes of Ranvier",
    "opc": "Saltatory conduction of impulses is seen",
    "opd": "Local anesthesia is effective only when the nerve is not covered",
    "cop": 0,
    "choice_type": "single",
    "exp": None, "subject_name": "Physiology", "topic_name": None,
}


def test_medmcqa_single_row_maps_cop_index_to_letter() -> None:
    item, _out = _map("medmcqa", MEDMCQA_ROW)
    assert item.verify.gold_letter == "A"  # cop is 0-indexed
    assert item.verify.letters == "ABCD"


def test_medmcqa_multi_choice_rows_are_dropped_counted() -> None:
    _, out = _map("medmcqa", {**MEDMCQA_ROW, "choice_type": "multi"})
    assert out.drops == {"multi_answer": 1}


# ------------------------------------------------------------------------- MMLU-Pro
MMLU_PRO_ROW = {
    "question_id": 70,
    "question": "Typical advertising regulatory bodies suggest ...",
    "options": [f"opt{i}" for i in range(10)],
    "answer": "I",
    "answer_index": 8,
    "cot_content": "",
    "category": "business",
    "src": "ori_mmlu-business_ethics",
}


def test_mmlu_pro_ten_option_alphabet_and_gold() -> None:
    item, _out = _map("mmlu_pro", MMLU_PRO_ROW)
    assert item.verify.letters == "ABCDEFGHIJ"
    assert item.verify.gold_letter == "I"
    assert item.retry_grammar == "Answer: [A-J]"


def test_mmlu_pro_health_subset_filters_other_categories() -> None:
    _, out = _map("mmlu_pro_health", MMLU_PRO_ROW)  # subset=health, row is business
    assert out.drops == {"other_category": 1}


# ----------------------------------------------------------------------- MedXpertQA
MEDXPERT_ROW = {
    "id": "Text-0",
    "question": "Which patient scenario represents the most appropriate indication?\n"
                "Answer Choices: (A) first (B) second (C) third (D) fourth (E) fifth",
    "options": {"A": "first", "B": "second", "C": "third", "D": "fourth", "E": "fifth"},
    "label": "E",
    "medical_task": "Basic Science", "body_system": "Skeletal", "question_type": "Reasoning",
}


def test_medxpertqa_strips_embedded_choices() -> None:
    item, _out = _map("medxpertqa_text", MEDXPERT_ROW)
    assert "Answer Choices" not in item.messages[1]["content"]
    assert item.verify.gold_letter == "E"
    assert item.verify.letters == "ABCDE"
    # the dataset's own rendering of the options is not duplicated into the prompt
    assert item.messages[1]["content"].count("first") == 1


# --------------------------------------------------------------------------- MediQAl
MEDIQAL_ROW = {
    "id": "21347",
    "clinical_case": "Un homme de 52 ans, éthylique chronique, consulte pour une diarrhée",
    "question": "Parmi les maladies suivantes, quelle(s) est(sont) la(les) cause(s) de diarrhée motrice ?",
    "answer_a": "Hyperthyroïdie",
    "answer_b": "Tumeur carcinoïde",
    "answer_c": "Cancer médullaire du corps thyroïde",
    "answer_d": "Déconjugaision des sels biliaires dans l'iléon",
    "answer_e": "Diabète avec neuropathie végétative",
    "correct_answers": "C",
    "task": "QCM", "medical_subject": "Hepato-Gastroenterology", "question_type": "Understanding",
}


def test_mediqal_single_answer_row_maps() -> None:
    item, _out = _map("mediqal", MEDIQAL_ROW)
    assert item.verify.gold_letter == "C"
    assert item.verify.letters == "ABCDE"
    assert "Un homme de 52 ans" in item.messages[1]["content"]


def test_mediqal_multi_answer_rows_are_dropped() -> None:
    _, out = _map("mediqal", {**MEDIQAL_ROW, "correct_answers": "A,B"})
    assert out.drops == {"multi_answer": 1}


def test_mediqal_gappy_options_are_dropped_not_mislabeled() -> None:
    # answer_c missing while answer_e present: a prefix-mapping would silently relabel
    # option D->C and E->D and point the gold at the wrong text.
    gappy = {**MEDIQAL_ROW, "answer_c": None}
    _, out = _map("mediqal", gappy)
    assert out.drops == {"gappy_or_missing_options": 1}


# -------------------------------------------------------------------- FrenchMedMCQA
FRENCHMED_ROW = {
    "id": "5987fa6bffd499eb439c90679d7fbca822d62bc639d1b9c94c68ae20e46f6004",
    "question": "Parmi les propositions suivantes, indiquer celle qui est exacte. "
                "Dans les conditions physiologiques, le pH le plus élevé est mesuré dans:",
    "answers": {
        "a": "Le suc gastrique", "b": "La bile vésiculaire", "c": "Le suc pancréatique",
        "d": "La salive", "e": "Les sécrétions intestinales",
    },
    "correct_answers": ["c"],
    "subject_name": "Biologie", "nbr_correct_answers": 1,
}


def test_frenchmedmcqa_lowercase_letters_normalize() -> None:
    item, _out = _map("frenchmedmcqa", FRENCHMED_ROW)
    assert item.verify.gold_letter == "C"
    assert item.verify.letters == "ABCDE"


def test_frenchmedmcqa_multi_answer_dropped() -> None:
    _, out = _map("frenchmedmcqa", {**FRENCHMED_ROW, "correct_answers": ["a", "c"]})
    assert out.drops == {"multi_answer": 1}


# ------------------------------------------------------------------------ HealthBench
HEALTHBENCH_ROW = {
    "prompt_id": "hb-1",
    "prompt": [
        {"role": "user", "content": "I'm a 39 year old female."},
        {"role": "user", "content": "I want to confirm some of the new BLS and ACLS updates."},
    ],
    "rubrics": [
        {"criterion": "Correctly states compression depth remains 5-6 cm.", "points": 10,
         "tags": ["level:example", "axis:accuracy"]},
        {"criterion": "Recommends doing CPR without training.", "points": -10,
         "tags": ["level:example", "axis:harmfulness"]},
    ],
    "example_tags": ["theme:communication"],
}


def test_healthbench_signed_points_become_signed_weights() -> None:
    item, out = _map("healthbench_hard", HEALTHBENCH_ROW)
    assert out.drops == {}
    weights = {c.id: c.weight for c in item.verify.criteria}
    assert weights == {"c0": 10.0, "c1": -10.0}
    assert item.verify.style is VerifyStyle.RUBRIC
    assert item.messages[0]["role"] == "system"
    assert "BLS and ACLS updates" in item.messages[1]["content"]
    assert "39 year old female" in item.messages[1]["content"]  # history folded in


# --------------------------------------------------------------------------- MedCalc
MEDCALC_ROW = {
    "Row Number": "1", "Calculator ID": "2",
    "Calculator Name": "Creatinine Clearance (Cockcroft-Gault Equation)",
    "Category": "lab test", "Output Type": "decimal",
    "Note ID": "pmc-7671985-1", "Note Type": "Extracted",
    "Patient Note": "An 87-year-old man was admitted for anorexia.",
    "Question": "What is the patient's Creatinine Clearance in mL/min?",
    "Relevant Entities": "{'sex': 'Male'}",
    "Ground Truth Answer": "25.2381",
    "Lower Limit": "23.97619", "Upper Limit": "26.50001",
    "Ground Truth Explanation": "The formula ...",
}


def test_medcalc_carries_dataset_tolerance_window() -> None:
    item, _out = _map("medcalc", MEDCALC_ROW)
    v = item.verify
    assert (v.gold_number, v.lower, v.upper) == pytest.approx((25.2381, 23.97619, 26.50001))
    assert item.verify.style is VerifyStyle.NUMBER


def test_medcalc_non_numeric_gold_is_dropped_counted() -> None:
    _, out = _map("medcalc", {**MEDCALC_ROW, "Ground Truth Answer": "Yes"})
    assert out.drops == {"non_numeric_gold": 1}


# ---------------------------------------------------------------------------- IFEval
IFEVAL_ROW = {
    "key": 1000,
    "prompt": "Write a 300+ word summary. Do not use any commas.",
    "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
    "kwargs": [
        {"num_words": None, "relation": None, "num_highlights": None},
        {"num_words": 300, "relation": "at least"},
    ],
}


def test_ifeval_instructions_and_kwargs_travel_together() -> None:
    item, _out = _map("ifeval", IFEVAL_ROW)
    assert item.verify.instruction_ids == ("punctuation:no_comma", "length_constraints:number_words")
    assert '"num_words": 300' in item.verify.instruction_kwargs[1]
    assert item.verify.style is VerifyStyle.FORMAT_RULES


# ------------------------------------------------------------------------------ GPQA
GPQA_ROW = {
    "id": "gpqa-1",
    "Question": "What is the mixing angle in neutrino oscillation?",
    "Correct Answer": "theta_13",
    "Incorrect Answer 1": "theta_12",
    "Incorrect Answer 2": "theta_23",
    "Incorrect Answer 3": "theta_31",
}


def test_gpqa_shuffle_is_deterministic_and_gold_tracks_position() -> None:
    item1, _ = _map("gpqa_diamond", GPQA_ROW)
    item2, _ = _map("gpqa_diamond", GPQA_ROW)
    letters = "ABCD"
    gold_text = next(line[3:] for line in item1.messages[1]["content"].splitlines()
                 if line.startswith(f"{item1.verify.gold_letter}. "))
    assert gold_text == "theta_13"
    # same row -> same shuffle (no PYTHONHASHSEED dependence)
    assert item1.verify.gold_letter == item2.verify.gold_letter
    assert item1.messages == item2.messages
    assert item1.verify.gold_letter in letters


def test_malformed_rows_fail_loudly_not_silently() -> None:
    with pytest.raises(LoaderError, match="outside alphabet"):
        _map("medqa", {**MEDQA_ROW,
                       "data": {**MEDQA_ROW["data"], "Correct Option": "Z"}})
    with pytest.raises(LoaderError, match="no id"):
        _map("medqa", {**MEDQA_ROW, "id": None})


# ------------------------------------------------------------------ load_items meta
def test_load_items_fingerprints_evaluated_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance: the audit must pin what was actually graded, not what was asked for."""
    from medrl.eval import loaders

    spec = TASKS.get("medqa").with_overrides(limit=1)
    monkeypatch.setattr(
        loaders, "_FETCHERS",
        {"medqa": lambda s: iter([dict(MEDQA_ROW), dict(MEDQA_ROW)])},
    )
    result = loaders.load_items(spec)
    assert result.meta["n_items"] == 1  # limit applied after mapping
    assert result.meta["rows_seen"] == 1  # fetch stops at the limit
    assert len(result.meta["items_sha256"]) == 16
    assert result.meta["revision"] is None  # unpinned: recorded as such


def test_load_items_fingerprint_tracks_content(monkeypatch: pytest.MonkeyPatch) -> None:
    from medrl.eval import loaders

    spec = TASKS.get("medqa")
    monkeypatch.setattr(loaders, "_FETCHERS", {"medqa": lambda s: iter([dict(MEDQA_ROW)])})
    a = loaders.load_items(spec).meta["items_sha256"]

    changed = dict(MEDQA_ROW)
    changed["data"] = {**changed["data"], "Correct Option": "C"}
    monkeypatch.setattr(loaders, "_FETCHERS", {"medqa": lambda s: iter([changed])})
    b = loaders.load_items(spec).meta["items_sha256"]
    assert a != b  # same id, different gold: different fingerprint
