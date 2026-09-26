"""S9 -- answer verification, split by answer_type.

The plan's "~30% removal" only anchors on rows that HAVE a gold answer, and the
dataset audit showed ~95% of Pool B does not (II-Medical-Reasoning-SFT carries
no answer field at all). So this stage is honest about scope:

* mcqa     -> parse the final 'Answer: <letter>' from the trace and compare with
              the gold letter (eval extraction/verifiers reused verbatim);
* numeric  -> parse a quantity and compare within the eval tolerance
              (verify_number, rtol from thresholds; MedCalc-style windows when
              the source recorded lower/upper in meta);
* free_text/none -> NO f_answer_wrong (they are judged by S11 criteria instead;
              FineMed rows additionally carry source quality labels in meta).

Flags written: f_answer_wrong; plus f_answer_right_reasoning_contradicts when a
trace reaches the gold letter while explicitly discarding it mid-reasoning
(cheap deterministic probe: the contract line contradicts the last-letter
pattern -- the judge-authored criterion pair covers the deep cases in S11).
"""

from __future__ import annotations

import re

from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageManifest, utcnow
from medrl.curation.thresholds import THRESHOLDS

_FINAL_LETTER = re.compile(r"Answer:\s*\(?([A-J])\)?\s*$", re.IGNORECASE | re.MULTILINE)
_CONTRADICT = re.compile(
    r"(?:actually|wait|hmm|no,|on second thought|correcting myself|I was wrong)", re.IGNORECASE
)


def _gold_letter(item: CorpusItem) -> str | None:
    """Gold letter from answer/meta.options position."""
    if item.answer and len(item.answer.strip()) == 1 and item.answer.strip().isalpha():
        return item.answer.strip().upper()
    options = item.meta.get("options")
    if options and item.answer:
        # answer may be an option's full text -- find its index letter
        for i, opt in enumerate(options):
            if str(opt).strip() == item.answer.strip():
                return chr(ord("A") + i)
    return None


def verify_mcqa(item: CorpusItem) -> dict[str, bool]:
    """{f_answer_wrong, f_answer_right_reasoning_contradicts} for one mcqa row."""
    trace = (item.thinking or "") + "\n" + (item.messages[-1]["content"] if item.messages else "")
    matches = list(_FINAL_LETTER.finditer(trace))
    gold = _gold_letter(item)
    if not matches or gold is None:
        return {"f_answer_wrong": False, "f_answer_right_reasoning_contradicts": False}
    said = matches[-1].group(1).upper()
    wrong = said != gold
    contradicts = (not wrong) and bool(_CONTRADICT.search(trace[-2000:]))
    return {"f_answer_wrong": wrong, "f_answer_right_reasoning_contradicts": contradicts}


def verify_numeric(item: CorpusItem) -> bool:
    """Parse the last number in the trace and compare within tolerance."""
    gold = item.answer
    if gold is None:
        return False
    try:
        gold_v = float(str(gold).replace(",", "").replace("%", "").strip())
    except ValueError:
        return False
    text = item.thinking or ""
    if item.messages:
        text += "\n" + item.messages[-1]["content"]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return True  # nothing parsed -> treat as wrong (no valid numeric answer)
    try:
        said = float(nums[-1].replace(",", ""))
    except ValueError:
        return True
    rtol, atol = THRESHOLDS.numeric_rtol, THRESHOLDS.numeric_atol
    if item.meta.get("lower") is not None and item.meta.get("upper") is not None:
        return not (float(item.meta["lower"]) <= said <= float(item.meta["upper"]))
    return abs(said - gold_v) > atol + rtol * abs(gold_v)


def run_answers(
    run_id: str,
    *,
    input_stage: str = "08_concept",
    output_stage: str = "09_answers",
) -> StageManifest:
    started = utcnow()
    inp = store.stage_dir(run_id, input_stage)
    out = store.reset_dir(store.stage_dir(run_id, output_stage))
    manifest = StageManifest(
        run_id=run_id, stage=output_stage, started_at=started,
        thresholds={"numeric_rtol": THRESHOLDS.numeric_rtol},
        config={},
    )
    counts = {"mcqa": 0, "numeric": 0, "free_text": 0, "none": 0}
    n_wrong = n_contra = 0
    buf: list[CorpusItem] = []

    for it in store.iter_items(inp):
        counts[it.answer_type] = counts.get(it.answer_type, 0) + 1
        updates: dict = {}
        if it.answer_type == "mcqa" and not it.flags.f_answer_wrong:
            flags = verify_mcqa(it)
            if flags["f_answer_wrong"]:
                updates["flags"] = it.flags.model_copy(update={"f_answer_wrong": True})
                n_wrong += 1
            elif flags["f_answer_right_reasoning_contradicts"]:
                updates["flags"] = it.flags.model_copy(
                    update={"f_answer_right_reasoning_contradicts": True})
                n_contra += 1
        elif it.answer_type == "numeric" and not it.flags.f_answer_wrong:
            if verify_numeric(it):
                updates["flags"] = it.flags.model_copy(update={"f_answer_wrong": True})
                n_wrong += 1
        buf.append(it.model_copy(update=updates) if updates else it)
        if len(buf) >= 50_000:
            store.write_items(buf, out)
            buf.clear()
    if buf:
        store.write_items(buf, out)

    manifest.rows_in = store.count_items(inp)
    manifest.rows_out = store.count_items(out)
    total = max(manifest.rows_in, 1)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.flag_rates = {"f_answer_wrong": {"_all": n_wrong / total}}
    manifest.notes = {"by_answer_type": counts, "wrong": n_wrong,
                      "right_but_contradicts": n_contra,
                      "no_answer_rows_skipped": counts["free_text"] + counts["none"]}
    return manifest


__all__ = ["run_answers", "verify_mcqa", "verify_numeric"]
