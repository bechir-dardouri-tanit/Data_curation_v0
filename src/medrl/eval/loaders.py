"""Dataset loaders: benchmark rows in, :class:`EvalItem` out.

Two layers, on purpose.

``row_to_item`` is a *pure function* from a row dict to an EvalItem (or ``None`` when the
row is legitimately unusable -- a multi-answer MCQ under a single-letter contract, say).
It is tested against real captured rows on a CPU box, so a schema change upstream fails a
test instead of corrupting a GPU run.

The fetch layer is the only place that knows how to *get* rows: the HF ``datasets``
library for parquet datasets, direct file download for the jsonl/zip datasets the
library cannot load (HealthBench's versioned jsonl files; FrenchMedMCQA's script-based
zip, which ``datasets>=3`` refuses to execute). Fetching is import-guarded so this
module still imports on a CPU box.

Every schema below was verified against the live Hub (2026-08-19) rather than copied from
a README; the row shapes are pinned by the captured-row tests in ``tests/test_loaders.py``.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger
from medrl.eval.extraction import normalize_letter
from medrl.eval.items import EvalItem, VerifySpec
from medrl.eval.scorers.judge import Criterion
from medrl.eval.tasks.prompts import (
    MCQA_SYSTEM,
    NUMERIC_SYSTEM,
    OPEN_RUBRIC_SYSTEM,
    build_mcqa_user,
    build_open_user,
    mcqa_grammar,
)
from medrl.eval.tasks.spec import TaskSpec, VerifyStyle

log = get_logger(__name__)


class GatedDatasetError(RuntimeError):
    """The dataset repo requires accepting terms / an HF token before download."""


class LoaderError(RuntimeError):
    """The dataset did not match the schema this loader was built against."""


@dataclass
class LoadResult:
    """Items plus an audit of what was dropped and why.

    Drops are normal (multi-answer rows under a single-letter contract) but must be
    *counted* -- "we evaluated 311 of 622 items" is a material fact about a reported
    score, not a footnote.
    """

    items: list[EvalItem] = field(default_factory=list)
    drops: dict[str, int] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1


Row = dict[str, Any]
RowMapper = Callable[[TaskSpec, Row, LoadResult], EvalItem | None]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _alphabet(n: int) -> str:
    """The first ``n`` uppercase letters, A-anchored, matching TaskSpec's contiguity rule."""
    if not 2 <= n <= 10:
        raise LoaderError(f"expected 2-10 options, got {n}")
    return "".join(chr(ord("A") + i) for i in range(n))


def _mcqa_item(
    spec: TaskSpec,
    item_id: object,
    question: str,
    options: list[str],
    gold: str,
) -> EvalItem:
    letters = _alphabet(len(options))
    gold_letter = normalize_letter(gold, letters)
    if gold_letter is None:
        raise LoaderError(f"{spec.name}/{item_id}: gold {gold!r} outside alphabet {letters}")
    if item_id is None or str(item_id) == "":
        raise LoaderError(f"{spec.name}: row has no id; item identity would collide")
    return EvalItem(
        benchmark=spec.name,
        item_id=str(item_id),
        messages=(
            {"role": "system", "content": MCQA_SYSTEM},
            {"role": "user", "content": build_mcqa_user(question, options)},
        ),
        verify=VerifySpec(style=VerifyStyle.LETTER, letters=letters, gold_letter=gold_letter),
        retry_grammar=mcqa_grammar(letters),
    )


def _strip_embedded_choices(question: str) -> str:
    """MedXpertQA embeds 'Answer Choices: (A) ...' inside the question text itself."""
    return question.split("\nAnswer Choices:")[0].strip()


# --------------------------------------------------------------------------------------
# row mappers (pure; tested against captured rows)
# --------------------------------------------------------------------------------------

def _medqa_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    data = row.get("data") or {}
    question = (data.get("Question") or "").strip()
    options_by_letter: dict[str, str] = data.get("Options") or {}
    options = [options_by_letter[k] for k in sorted(options_by_letter)]
    gold = str(data.get("Correct Option") or "")
    if not question or len(options) < 2 or not gold:
        out.drop("malformed_row")
        return None
    return _mcqa_item(spec, row.get("id"), question, options, gold)


def _medmcqa_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    if (row.get("choice_type") or "single") != "single":
        out.drop("multi_answer")
        return None
    options = [row[k] for k in ("opa", "opb", "opc", "opd")]
    if any(not o for o in options):
        out.drop("malformed_row")
        return None
    return _mcqa_item(spec, row["id"], row["question"], options, chr(ord("A") + int(row["cop"])))


def _mmlu_pro_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    # subset acts as a *category filter* here (health), not a datasets config name.
    if spec.subset and row.get("category") != spec.subset:
        out.drop("other_category")
        return None
    options = list(row["options"])
    if len(options) < 2:
        out.drop("malformed_row")
        return None
    return _mcqa_item(spec, row["question_id"], row["question"], options, str(row["answer"]))


def _medxpertqa_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    options_by_letter: dict[str, str] = row.get("options") or {}
    options = [options_by_letter[k] for k in sorted(options_by_letter)]
    question = _strip_embedded_choices(row.get("question") or "")
    gold = str(row.get("label") or "")
    if not question or len(options) < 2 or not gold:
        out.drop("malformed_row")
        return None
    return _mcqa_item(spec, row.get("id"), question, options, gold)


def _mediqal_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    correct = [c.strip().upper() for c in str(row.get("correct_answers") or "").split(",") if c.strip()]
    if len(correct) != 1:
        out.drop("multi_answer")
        return None
    options = [row.get(f"answer_{s}") for s in "abcde"]
    present = [(letter, text) for letter, text in zip("ABCDE", options, strict=False) if text]
    # Only a *prefix* of A-E keeps letters aligned with the gold's letter labels; a gap
    # (answer_c missing, answer_e present) would relabel options and misroute the gold.
    if len(present) < 2 or present[-1][0] != "ABCDE"[len(present) - 1]:
        out.drop("gappy_or_missing_options")
        return None
    question = f"{(row.get('clinical_case') or '').strip()}\n\n{(row.get('question') or '').strip()}"
    return _mcqa_item(
        spec, row.get("id"), question,
        [text for _, text in present], correct[0],
    )


def _frenchmedmcqa_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    correct = [str(c).upper() for c in row.get("correct_answers") or []]
    if len(correct) != 1:
        out.drop("multi_answer")
        return None
    answers: dict[str, str] = row.get("answers") or {}
    letters = sorted(answers)
    options = [answers[k] for k in letters]
    if len(options) < 2:
        out.drop("malformed_row")
        return None
    return _mcqa_item(spec, row.get("id"), row["question"], options, correct[0])


def _healthbench_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    prompt = row.get("prompt") or []
    rubrics = row.get("rubrics") or []
    if not prompt or not rubrics:
        out.drop("malformed_row")
        return None
    history = [dict(m) for m in prompt[:-1]]
    question = prompt[-1]["content"]
    criteria = tuple(
        Criterion(id=f"c{i}", text=str(r.get("criterion") or ""), weight=float(r.get("points", 1)))
        for i, r in enumerate(rubrics)
        if r.get("criterion")
    )
    if not criteria:
        out.drop("malformed_row")
        return None
    return EvalItem(
        benchmark=spec.name,
        item_id=str(row.get("prompt_id")),
        messages=(
            {"role": "system", "content": OPEN_RUBRIC_SYSTEM},
            {"role": "user", "content": build_open_user(question, history=history)},
        ),
        verify=VerifySpec(style=VerifyStyle.RUBRIC, criteria=criteria),
    )


def _medcalc_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    def _num(value: object) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    gold = _num(row.get("Ground Truth Answer"))
    if gold is None:
        # Non-numeric output types (e.g. categorical calculator answers) are out of
        # scope for tolerance-graded numeric verification; counted, never silent.
        out.drop("non_numeric_gold")
        return None
    question = f"{(row.get('Patient Note') or '').strip()}\n\n{(row.get('Question') or '').strip()}"
    return EvalItem(
        benchmark=spec.name,
        item_id=str(row.get("Row Number")),
        messages=(
            {"role": "system", "content": NUMERIC_SYSTEM},
            {"role": "user", "content": build_open_user(question)},
        ),
        verify=VerifySpec(
            style=VerifyStyle.NUMBER,
            gold_number=gold,
            lower=_num(row.get("Lower Limit")),
            upper=_num(row.get("Upper Limit")),
            rtol=float(spec.verify_params.get("rtol", "0.005")),
        ),
    )


def _ifeval_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    ids = list(row.get("instruction_id_list") or [])
    if not ids:
        out.drop("malformed_row")
        return None
    kwargs = [json.dumps(k or {}) for k in row.get("kwargs") or []]
    return EvalItem(
        benchmark=spec.name,
        item_id=str(row.get("key")),
        messages=(
            {"role": "system", "content": OPEN_RUBRIC_SYSTEM},
            {"role": "user", "content": build_open_user(row.get("prompt") or "")},
        ),
        verify=VerifySpec(
            style=VerifyStyle.FORMAT_RULES,
            instruction_ids=tuple(ids),
            instruction_kwargs=tuple(kwargs[: len(ids)]),
        ),
    )


def _gpqa_row(spec: TaskSpec, row: Row, out: LoadResult) -> EvalItem | None:
    from medrl.core.hashing import hash_text

    question = (row.get("Question") or "").strip()
    correct = (row.get("Correct Answer") or "").strip()
    wrongs = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
    if not question or not correct or any(not w for w in wrongs):
        out.drop("malformed_row")
        return None
    # GPQA ships the answer position-shuffled already, but the published ordering is
    # public; re-shuffle deterministically per-item so letter priors carry no signal.
    order: list[str] = [correct, *(str(w) for w in wrongs)]
    key = int(hash_text(str(row.get("id", question)))[:12], 16)
    for i in range(len(order) - 1, 0, -1):
        j = (key >> (4 * i)) % (i + 1)
        order[i], order[j] = order[j], order[i]
    gold = "ABCD"[order.index(correct)]
    return _mcqa_item(spec, row.get("id"), question, order, gold)


# --------------------------------------------------------------------------------------
# fetch layer (import-guarded)
# --------------------------------------------------------------------------------------

def _hf_rows(spec: TaskSpec, *, config_name: str | None) -> Iterator[Row]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only on slim hosts
        raise LoaderError("the `datasets` package is required: uv pip install -e '.[eval]'") from exc
    try:
        ds = load_dataset(spec.hf_id, config_name, split=spec.split)
    except Exception as exc:
        if _is_gated(exc):
            raise GatedDatasetError(
                f"{spec.hf_id} is gated: accept the terms on the Hub and provide credentials "
                "(hf auth login, or HF_TOKEN in the environment) before running this benchmark"
            ) from exc
        raise
    for row in ds:
        yield dict(row)


def _is_gated(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "gated" in text or "401" in text or "403" in text or "agree" in text or "access" in text


def _downloaded_files(spec: TaskSpec, pattern: str) -> list[str]:
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(spec.hf_id or "") if _glob_match(f, pattern)]
    if len(files) != 1:
        raise LoaderError(
            f"{spec.hf_id}: expected exactly one file matching {pattern!r}, found {files}; "
            "pin the filename in the task spec to restore determinism"
        )
    return [hf_hub_download(repo_id=spec.hf_id or "", filename=files[0])]


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(name.split("/")[-1], pattern)


def _healthbench_rows(spec: TaskSpec) -> Iterator[Row]:
    # HealthBench publishes versioned jsonl files whose names change; resolve by prefix
    # and refuse ambiguity rather than silently grading a different mix.
    pattern = "hard_*.jsonl" if spec.subset == "hard" else "*oss_eval.jsonl"
    (path,) = _downloaded_files(spec, pattern)
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield dict(json.loads(line))


def _frenchmedmcqa_rows(spec: TaskSpec) -> Iterator[Row]:
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id=spec.hf_id or "", filename="DEFT-2023-FULL.zip")
    # The repo is a script dataset (datasets>=3 refuses to execute scripts), so the zip
    # is read directly; the split files live at the archive root.
    with zipfile.ZipFile(zip_path) as zf:
        member = f"{spec.split}.json"
        if member not in zf.namelist():
            raise LoaderError(f"DEFT-2023-FULL.zip has no {member}; members={zf.namelist()}")
        rows = json.load(io.TextIOWrapper(zf.open(member), encoding="utf-8"))
    yield from rows


_FETCHERS: dict[str, Callable[[TaskSpec], Iterator[Row]]] = {
    "medqa": lambda s: _hf_rows(s, config_name=None),
    "medmcqa": lambda s: _hf_rows(s, config_name=None),
    "mmlu_pro": lambda s: _hf_rows(s, config_name=None),
    "medxpertqa": lambda s: _hf_rows(s, config_name=s.subset),
    "mediqal": lambda s: _hf_rows(s, config_name=s.subset),
    "healthbench": _healthbench_rows,
    "frenchmedmcqa": _frenchmedmcqa_rows,
    "medcalc": lambda s: _hf_rows(s, config_name=None),
    "ifeval": lambda s: _hf_rows(s, config_name=None),
    "gpqa": lambda s: _hf_rows(s, config_name=s.subset),
}

_MAPPERS: dict[str, RowMapper] = {
    "medqa": _medqa_row,
    "medmcqa": _medmcqa_row,
    "mmlu_pro": _mmlu_pro_row,
    "medxpertqa": _medxpertqa_row,
    "mediqal": _mediqal_row,
    "frenchmedmcqa": _frenchmedmcqa_row,
    "healthbench": _healthbench_row,
    "medcalc": _medcalc_row,
    "ifeval": _ifeval_row,
    "gpqa": _gpqa_row,
}


def load_items(spec: TaskSpec) -> LoadResult:
    """Fetch and map one benchmark's rows into items, applying ``limit`` last.

    The limit applies *after* dropping so ``limit: 100`` always means "100 evaluated
    items", never "the first 100 rows of which some number survive".
    """
    result = LoadResult()
    fetch = _FETCHERS.get(spec.loader)
    mapper = _MAPPERS.get(spec.loader)
    if fetch is None or mapper is None:
        raise LoaderError(f"no loader registered for {spec.loader!r} (task {spec.name})")

    n_rows = 0
    for row in fetch(spec):
        n_rows += 1
        item = mapper(spec, row, result)
        if item is not None:
            result.items.append(item)
        if spec.limit is not None and len(result.items) >= spec.limit:
            break
    result.meta = {"rows_seen": n_rows, "n_items": len(result.items), "prompt_style": spec.prompt_style.value}
    if not result.items:
        raise LoaderError(
            f"{spec.name}: no usable items from {spec.hf_id} (drops={result.drops}); "
            "the upstream schema likely changed"
        )
    log.info(
        "loaded %s: %d items from %d rows (drops=%s)",
        spec.name, len(result.items), n_rows, result.drops,
    )
    return result


__all__ = [
    "GatedDatasetError",
    "LoadResult",
    "LoaderError",
    "load_items",
]
