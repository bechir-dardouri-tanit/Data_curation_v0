"""The benchmark suite, as data.

Selection principle: a benchmark is DECISION only if its minimum detectable effect at our
item counts can resolve the effects we act on (see ``medrl.analysis.stats``); hard,
unsaturated sets are preferred; every train-side source is decontaminated against every
name here. French coverage is first-class (MediQAl), not a garnish.
"""

from __future__ import annotations

from typing import Any

from medrl.core.config import Grade, Language
from medrl.eval.tasks.prompts import mcqa_grammar
from medrl.eval.tasks.spec import TASKS, PromptStyle, TaskSpec, VerifyStyle, register_task

# MMLU-Pro's defining change is 10 answer choices; anything less forces-and-scores
# every F-J-gold item wrong (the grammar forbids the letter, the verifier can't read it).
_MMLU_PRO_LETTERS = "ABCDEFGHIJ"


def _mcqa_kw(letters: str = "ABCDE") -> dict[str, Any]:
    return {
        "prompt_style": PromptStyle.MCQA_LETTER,
        "verify_style": VerifyStyle.LETTER,
        "letters": letters,
        "guided_decoding": mcqa_grammar(letters),
    }

# ---- decision-grade English MCQA / reasoning -----------------------------------------
register_task(TaskSpec(
    name="medqa", loader="medqa", grade=Grade.DECISION, language=Language.EN,
    hf_id="openlifescienceai/medqa", split="test", notes="USMLE; high contamination risk",
    **_mcqa_kw(),
))
register_task(TaskSpec(
    name="medmcqa", loader="medmcqa", grade=Grade.DECISION, language=Language.EN,
    hf_id="openlifescienceai/medmcqa", split="validation", **_mcqa_kw(),
))
register_task(TaskSpec(
    name="mmlu_pro_health", loader="mmlu_pro", grade=Grade.DECISION, language=Language.EN,
    hf_id="TIGER-Lab/MMLU-Pro", subset="health", split="test", requires_judge=False,
    **_mcqa_kw(_MMLU_PRO_LETTERS),
))
register_task(TaskSpec(
    name="medxpertqa_text", loader="medxpertqa", grade=Grade.DECISION, language=Language.EN,
    hf_id="TsinghuaC3I/MedXpertQA", subset="Text", split="test",
    notes="Hard, unsaturated; NO train split -- never trained on", **_mcqa_kw(),
))

# ---- decision-grade French ------------------------------------------------------------
register_task(TaskSpec(
    name="mediqal", loader="mediqal", grade=Grade.DECISION, language=Language.FR,
    hf_id="ANR-MALADES/MediQAl", subset="mcqu", split="test",
    notes="French clinical MCQ, single-answer (unique) config; the sibling 'mcqm' config "
          "is 100% multi-select and incompatible with the single-letter contract",
    **_mcqa_kw(),
))
register_task(TaskSpec(
    name="frenchmedmcqa", loader="frenchmedmcqa", grade=Grade.REPORTING, language=Language.FR,
    hf_id="qanastek/frenchmedmcqa", split="test",
    notes="Pharmacy MCQ, script dataset -- read from the zip directly", **_mcqa_kw(),
))

# ---- rubric-graded --------------------------------------------------------------------
register_task(TaskSpec(
    name="healthbench_hard", loader="healthbench", grade=Grade.DECISION,
    language=Language.EN, hf_id="openai/healthbench", subset="hard", split="test",
    prompt_style=PromptStyle.OPEN_RUBRIC, verify_style=VerifyStyle.RUBRIC,
    requires_judge=True, notes="1000 hardest conversations; judge tier determines comparability",
))
register_task(TaskSpec(
    name="healthbench", loader="healthbench", grade=Grade.REPORTING,
    language=Language.EN, hf_id="openai/healthbench", split="test",
    prompt_style=PromptStyle.OPEN_RUBRIC, verify_style=VerifyStyle.RUBRIC,
    requires_judge=True,
))

# ---- verifiable skills ----------------------------------------------------------------
register_task(TaskSpec(
    name="medcalc", loader="medcalc", grade=Grade.DECISION, language=Language.EN,
    hf_id="ncbi/MedCalc-Bench", split="test",
    prompt_style=PromptStyle.NUMERIC, verify_style=VerifyStyle.NUMBER,
    verify_params={"rtol": "0.005"},
))
register_task(TaskSpec(
    name="ifeval", loader="ifeval", grade=Grade.DECISION, language=Language.EN,
    hf_id="google/IFEval", split="train",
    prompt_style=PromptStyle.OPEN_RUBRIC, verify_style=VerifyStyle.FORMAT_RULES,
    notes="Verifiable instruction following; guard against reward-hacked formatting",
))

# ---- forgetting guardrails (block promotion, never generate wins) ----------------------
register_task(TaskSpec(
    name="gpqa_diamond", loader="gpqa", grade=Grade.GUARDRAIL, language=Language.EN,
    hf_id="Idavidrein/gpqa", split="diamond", notes="Forgetting guardrail", **_mcqa_kw(),
))
register_task(TaskSpec(
    name="mmlu_pro", loader="mmlu_pro", grade=Grade.GUARDRAIL, language=Language.EN,
    hf_id="TIGER-Lab/MMLU-Pro", split="test", notes="General-capability guardrail",
    **_mcqa_kw(_MMLU_PRO_LETTERS),
))

DECISION_SET = tuple(sorted(
    spec.name for spec in
    (TASKS.get(n) for n in TASKS) if spec.grade is Grade.DECISION
))


def subset_for_grade(*grades: Grade) -> list[str]:
    return [n for n in TASKS if TASKS.get(n).grade in grades]

__all__ = ["DECISION_SET", "TASKS", "register_task", "subset_for_grade"]
