"""S12 -- difficulty labelling by pass@k sampling, routing rows to mixture bands.

Why pass@k: a row's training value depends on whether the *labelling policy* can
solve it, not on any intrinsic score. 8 seeded samples at passk_temperature give
a pass rate with just enough resolution to place the row in one of four bands
(band_downsample=1.0 -> downsample; [band_sft1_low, 1.0) -> sft1; [band_rl_low,
band_sft1_low) -> rl; [0, band_rl_low) -> hold; the boundary convention is
inclusive-lower/exclusive-upper per thresholds.py, with both ends inclusive).
The RL band is the gradient band: too easy and the advantage collapses to zero,
too hard and every rollout is penalized -- which is exactly what a pass rate
measures and a label never could.

Three behaviours this stage owns beyond "sample and grade":

* **Think-incomplete handling** (the audit's key finding). A sample whose
  reasoning opens ``<think>`` and never closes it is a *budget* failure, not a
  wrong answer: counting it as incorrect would inflate difficulty for exactly
  the long-thinking rows the RL band most needs. Each incomplete sample is
  retried once with more tokens (same seed: with a seeded sampler the retry
  reproduces the truncated trajectory and finishes it, so it is the same sample
  completed, not a fresh draw); if it still never closes, the repeat is marked
  missing and leaves the pass-rate denominator. If more than
  MAX_MISSING_REPEATS of the base repeats end up missing, the item is not
  rateable: difficulty=None plus ``meta['difficulty_note']`` -- an honest None
  beats a rate computed from 5 of 8 samples.

* **Adaptive top-up**. A pass-count from 8 samples sits exactly on a band edge
  far too often (4/8, 7/8, 8/8). Counts in passk_topup_k get passk_topup_samples
  more samples -- but only when the outcome could actually move the band
  (:func:`band_stable`), so easy/hold items never pay for samples that cannot
  change their routing.

* **Determinism**. Seeds derive from the item id (never Python's ``hash()``,
  which is process-salted -- the same reason eval/generate.py uses
  ``core.hashing.hash_text``), so a re-run with the same config reproduces the
  same completions, and ``run_generation``'s resume-by-key makes an interrupted
  stage continue instead of re-paying for finished samples.

Scope, documented: only rows with a gold answer and answer_type in {mcqa,
numeric} are labelled (a pass rate needs a gradeable gold); free_text/none rows
keep difficulty=None/band=None -- they are SFT material judged by S11 criteria,
not pass@k candidates. The stage writes no ``Flags`` fields (the schema reserves
none for difficulty), so ``flag_rates`` stays empty and the band distribution
travels in ``notes``.

Knob honesty: the audit spec's ">2 of 8 missing" and "+4096 retry tokens" are
not in THRESHOLDS. The retry headroom is derived as ``passk_think_budget // 2``
(= 4096 at current values) and MAX_MISSING_REPEATS is a module constant; both
are recorded in the manifest notes so recalibration touches one file.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medrl.core.hashing import hash_text
from medrl.core.logging import get_logger
from medrl.curation import store
from medrl.curation.schema import CorpusItem, DifficultyBand, StageError, StageManifest, utcnow
from medrl.curation.serving import Gateway, run_generation, write_generation_manifest
from medrl.curation.thresholds import THRESHOLDS, snapshot
from medrl.eval.extraction import ExtractionPath, extract_mcqa, extract_number
from medrl.eval.tasks.prompts import MCQA_SYSTEM, NUMERIC_SYSTEM, build_mcqa_user
from medrl.eval.verifiers import verify_letter, verify_number

log = get_logger(__name__)

INPUT_STAGE = "11_judge"
OUTPUT_STAGE = "12_difficulty"

MAX_MISSING_REPEATS = 2
"""Audit-spec constant: more than 2 missing base repeats makes an item
unrateable. Deliberately not in THRESHOLDS (see module docstring, knob honesty)."""

RETRY_SUFFIX = "::retry"
"""Resume-store distinction between a sample and its +token retry: run_generation
resumes by exact key, so a retry sharing the main key would be skipped as
already done (the truncated attempt is in the file under that key)."""

_WRITE_CHUNK = 50_000
"""Rows buffered per write_items call -- the answers-stage convention; store's
own default, mirrored here so one streaming loop bounds memory identically."""


# --------------------------------------------------------------------------
# Pure band logic -- the part that routes the mixture; unit-tested to the edge.
# --------------------------------------------------------------------------


def route_band(rate: float) -> DifficultyBand:
    """Map a pass rate to its mixture band.

    Convention (thresholds.py: inclusive-lower / exclusive-upper, ends
    inclusive): ``[0, band_rl_low) -> hold``; ``[band_rl_low, band_sft1_low) ->
    rl``; ``[band_sft1_low, 1.0) -> sft1``; ``{1.0} -> downsample``. So 0.0 ->
    hold, 0.1 -> rl, 0.4 -> rl, 0.5 -> sft1, 1.0 -> downsample. Rates between
    band_rl_high (0.4) and band_sft1_low (0.5) -- reachable only after a top-up
    (e.g. 7/16) -- stay rl, and rates in (0.9, 1.0) stay sft1: the exclusive
    upper edges are band_sft1_low and band_downsample, and band_rl_high names
    the band's documented ceiling, not a routing edge.

    Raises ValueError for anything outside [0, 1] (NaN included): a rate outside
    the unit interval is an orchestration bug that must fail loudly, not route
    as hold.
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"pass rate outside [0, 1]: {rate!r}")
    if rate >= THRESHOLDS.band_downsample:
        return "downsample"
    if rate >= THRESHOLDS.band_sft1_low:
        return "sft1"
    if rate >= THRESHOLDS.band_rl_low:
        return "rl"
    return "hold"


def band_stable(successes: int, n: int) -> bool:
    """True when no possible top-up count can move the item across a band edge.

    A top-up adds passk_topup_samples samples, so the achievable rates span
    [successes / (n + topup), (successes + topup) / (n + topup)]. route_band is
    monotone in the rate, so the extremes bracket every reachable band: if they
    route alike, no intermediate count can differ, and spending passk_topup_samples
    more samples cannot change the routing -- skip it. (With the shipped knobs
    the window is 8/16 wide, so stability is rare; the helper exists so the
    pilot's recalibrated topup/edge knobs cannot silently turn the top-up into
    unconditional cost.)
    """
    if n <= 0 or not (0 <= successes <= n):
        raise ValueError(f"need 0 <= successes ({successes}) <= n ({n})")
    topup = THRESHOLDS.passk_topup_samples
    n_after = n + topup
    return route_band(successes / n_after) == route_band((successes + topup) / n_after)


def aggregate(repeats: list[bool | None]) -> tuple[float | None, str]:
    """Pass rate over completed repeats; None entries are missing and leave the
    denominator. Returns (rate, note); note explains missing repeats and is
    non-empty whenever the rate rests on fewer samples than were asked for.

    More than MAX_MISSING_REPEATS missing repeats returns (None, note): the
    audit's guard against trusting a pass rate computed from a minority of the
    pass. An all-missing list that still passes that guard (only possible when
    fewer than 3 repeats were asked) is unrateable for the complementary reason.
    """
    missing = sum(1 for r in repeats if r is None)
    valid = len(repeats) - missing
    if missing > MAX_MISSING_REPEATS:
        note = (
            f"unrateable: {missing} of {len(repeats)} repeats missing after retry "
            f"(>{MAX_MISSING_REPEATS})"
        )
        return None, note
    if valid == 0:
        return None, "unrateable: no valid repeats"
    note = f"missing_repeats={missing}/{len(repeats)}" if missing else ""
    rate = sum(1 for r in repeats if r) / valid
    return rate, note


def sample_seed(item_id: str, repeat: int, seed_base: int) -> int:
    """Deterministic per-(item, repeat) seed:
    ``seed_base + hash(id) % 10**6 * passk_samples + repeat``.

    The multiplier is passk_samples (8), so each item owns a block of 8
    consecutive seeds -- repeats of one item never collide. Top-up repeats
    (8..15) continue into the next block, which can share seeds with a
    *different* item's block; that is harmless because a seed only shapes
    sampling within one prompt, and within an item all repeats stay distinct.
    hash_text (blake2b) is stable across processes; Python's hash() is not.
    """
    digest = int(hash_text(item_id)[:8], 16)
    return seed_base + (digest % 10**6) * THRESHOLDS.passk_samples + repeat


def think_incomplete(reasoning: str | None, content: str | None) -> bool:
    """A sample whose think trace opened and never closed.

    Same predicate as S2's truncation check (structural.py THINK_OPEN/THINK_CLOSE,
    the eval-side convention), applied per surface: with vLLM's reasoning parser
    the trace lives in ``reasoning``; without it the raw ``<think>`` block stays
    inline in ``content``. An open-without-close on either surface means the
    sample ran out of tokens mid-thought.
    """
    for text in (reasoning or "", content or ""):
        if "<think>" in text and "</think>" not in text:
            return True
    return False


# --------------------------------------------------------------------------
# Golds, prompts, grading.
# --------------------------------------------------------------------------


def _gold_letter(item: CorpusItem) -> str | None:
    """Gold MCQA letter: single-letter answer, or the option matching full text.

    Mirrors stages/answers.py::_gold_letter (stages never import each other);
    the letter-identity *comparison* stays in eval.verifiers.verify_letter, the
    single authority, so both stages still agree on what "equal" means.
    """
    answer = (item.answer or "").strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    options = item.meta.get("options")
    if options and answer:
        values = options.values() if isinstance(options, dict) else options
        for i, opt in enumerate(values):
            if str(opt).strip() == answer:
                return chr(ord("A") + i)
    return None


def _gold_number(item: CorpusItem) -> float | None:
    """Gold numeric value, parsed with the answers-stage convention (commas and
    percent stripped); None when the gold is not a parseable number."""
    try:
        return float(str(item.answer).replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return None


def _letters_of(item: CorpusItem) -> str:
    """The item's option alphabet: one letter per option, from A (extraction is
    alphabet-shaped; the count must match the prompt's rendered options)."""
    options = item.meta.get("options")
    n = len(options) if options else 0
    n = max(2, min(n, 26))
    return "".join(chr(ord("A") + i) for i in range(n))


def _options_list(item: CorpusItem) -> list[str | tuple[str, str]]:
    """meta['options'] in build_mcqa_user's shape; dicts arrive as (letter, text)."""
    options = item.meta.get("options")
    if isinstance(options, dict):
        return [(str(k), str(v)) for k, v in sorted(options.items())]
    return [str(o) for o in (options or [])]


def _question_of(item: CorpusItem) -> str:
    """User turns up to the first assistant turn. The assistant turn is source
    text that may quote or contain the gold answer -- prompting with it would
    leak the label the pass rate is supposed to measure."""
    parts: list[str] = []
    for m in item.messages:
        if m.get("role") == "assistant":
            break
        if m.get("role") == "user":
            parts.append(m.get("content", ""))
    return "\n".join(parts).strip()


def sampling_messages(item: CorpusItem) -> list[dict[str, str]]:
    """The zero-shot labelling prompt: the shared eval contract system prompts
    (the single place the 'Answer: <LETTER>' phrasing lives) plus the question
    with options rendered. Never the stored assistant response or thinking."""
    question = _question_of(item)
    if item.answer_type == "mcqa":
        return [
            {"role": "system", "content": MCQA_SYSTEM},
            {"role": "user", "content": build_mcqa_user(question, _options_list(item))},
        ]
    return [
        {"role": "system", "content": NUMERIC_SYSTEM},
        {"role": "user", "content": question},
    ]


def labelable(item: CorpusItem) -> bool:
    """Whether this row gets a pass@k label: a gradeable gold of MCQA or numeric
    type, with options to render for MCQA (a letter question without options
    cannot be asked). Everything else keeps difficulty=None by design."""
    if item.answer_type not in ("mcqa", "numeric"):
        return False
    if not (item.answer and item.answer.strip()):
        return False
    if not _question_of(item):
        return False
    if item.answer_type == "mcqa":
        return bool(item.meta.get("options")) and _gold_letter(item) is not None
    return _gold_number(item) is not None


def grade_sample(item: CorpusItem, content: str | None, reasoning: str | None) -> bool:
    """One completed sample against the item's gold. Extraction failure grades
    False -- for a pass rate, an answer that defeats extraction did not pass
    (the same verdict verify_letter gives a None prediction). The content surface
    is graded first, the reasoning surface only as fallback: models put the
    contract line after ``</think>``, which is exactly what the parser keeps in
    content."""
    if item.answer_type == "mcqa":
        letters = _letters_of(item)
        result = extract_mcqa(content or "", letters)
        if result.path is ExtractionPath.FAILED and reasoning:
            result = extract_mcqa(reasoning, letters)
        gold_letter = _gold_letter(item)
        return gold_letter is not None and verify_letter(result.value, gold_letter, letters)

    pred = extract_number(content or "")
    if pred is None and reasoning:
        pred = extract_number(reasoning)
    gold_value = _gold_number(item)
    if gold_value is None:
        return False
    lower, upper = item.meta.get("lower"), item.meta.get("upper")
    if lower is not None and upper is not None:  # MedCalc-style window, as S9
        return pred is not None and float(lower) <= pred <= float(upper)
    return verify_number(pred, gold_value, THRESHOLDS.numeric_rtol, THRESHOLDS.numeric_atol)


# --------------------------------------------------------------------------
# Generation orchestration: main pass -> think-retry -> adaptive top-up.
# --------------------------------------------------------------------------


def _read_records(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """Records for `wanted` keys from a generations JSONL (last line wins), so a
    re-read parses the whole file but retains only this round's records."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn trailing line: the run_generation convention
            key = rec.get("key")
            if key in wanted:
                out[key] = rec
    return out


def _classify(item: CorpusItem, rec: dict[str, Any] | None) -> tuple[bool | None, bool]:
    """(graded, needs_retry) for one sample record. An errored request is a
    missing repeat (None) -- an infrastructure failure must not count as a wrong
    answer; an unclosed think trace asks for the +token retry."""
    if rec is None or rec.get("error"):
        return None, False
    if think_incomplete(rec.get("reasoning"), rec.get("content")):
        return None, True
    return grade_sample(item, rec.get("content"), rec.get("reasoning")), False


def _job(
    item: CorpusItem, repeat: int, *, max_tokens: int, seed_base: int, retry: bool = False
) -> dict[str, Any]:
    """One run_generation job. Keys are ``<id>::<repeat>`` (the task convention);
    the retry appends RETRY_SUFFIX. top_p uses gen_top_p -- the closest existing
    knob (no passk_top_p; identical to run_generation's built-in default)."""
    return {
        "key": f"{item.id}::{repeat}" + (RETRY_SUFFIX if retry else ""),
        "messages": sampling_messages(item),
        "max_tokens": max_tokens,
        "temperature": THRESHOLDS.passk_temperature,
        "top_p": THRESHOLDS.gen_top_p,
        "seed": sample_seed(item.id, repeat, seed_base),
    }


async def _label_pass(
    gateway: Gateway,
    items: list[CorpusItem],
    gen_path: Path,
    *,
    seed_base: int,
    client_factory: Any | None,
    gen_rounds: list[dict[str, Any]],
) -> dict[str, tuple[float | None, str]]:
    """Run all sampling rounds; return {item_id: (rate | None, note)}.

    Rounds: main pass -> one +token retry per think-incomplete sample ->
    aggregate -> adaptive top-up (counts in passk_topup_k only, and only when
    band_stable says the routing could move) -> same retry treatment for top-up
    samples. One coroutine so the whole stage shares a single event loop (the
    gateway's AIMD semaphore must not straddle loops).
    """
    k = THRESHOLDS.passk_samples
    topup = THRESHOLDS.passk_topup_samples
    main_tokens = THRESHOLDS.passk_max_tokens
    # The audit's "+4096" retry headroom, derived from an existing knob: the
    # failure mode is a think trace that outgrew its budget, so the retry adds
    # half a think budget (8192 // 2 == 4096 at shipped values).
    retry_tokens = main_tokens + THRESHOLDS.passk_think_budget // 2
    by_id = {it.id: it for it in items}
    records: dict[str, dict[str, Any]] = {}

    async def run_round(jobs: list[dict[str, Any]]) -> None:
        if not jobs:
            return
        stats = await run_generation(gateway, jobs, gen_path, client_factory=client_factory)
        records.update(_read_records(gen_path, {j["key"] for j in jobs}))
        gen_rounds.append(stats)

    # -- main pass ---------------------------------------------------------
    await run_round(
        [_job(it, r, max_tokens=main_tokens, seed_base=seed_base) for it in items for r in range(k)]
    )
    repeats: dict[str, list[bool | None]] = {}
    pending: list[tuple[CorpusItem, int]] = []
    for it in items:
        reps: list[bool | None] = []
        for r in range(k):
            graded, incomplete = _classify(it, records.get(f"{it.id}::{r}"))
            if incomplete:
                pending.append((it, r))
            reps.append(graded)
        repeats[it.id] = reps

    # -- think-incomplete retry (once; still incomplete -> missing) ---------
    await run_round(
        [_job(it, r, max_tokens=retry_tokens, seed_base=seed_base, retry=True) for it, r in pending]
    )
    for it, r in pending:
        graded, _ = _classify(it, records.get(f"{it.id}::{r}{RETRY_SUFFIX}"))
        repeats[it.id][r] = graded

    # -- adaptive top-up ----------------------------------------------------
    outcomes: dict[str, tuple[float | None, str]] = {}
    topup_ids: list[str] = []
    topup_jobs: list[dict[str, Any]] = []
    for it in items:
        rate, note = aggregate(repeats[it.id])
        if rate is None:  # unrateable: no top-up can fix a missing majority
            outcomes[it.id] = (None, note)
            continue
        successes = sum(1 for x in repeats[it.id] if x is True)
        n_valid = sum(1 for x in repeats[it.id] if x is not None)
        if successes in THRESHOLDS.passk_topup_k and not band_stable(successes, n_valid):
            topup_ids.append(it.id)
            topup_jobs.extend(
                _job(it, r, max_tokens=main_tokens, seed_base=seed_base)
                for r in range(k, k + topup)
            )
    await run_round(topup_jobs)
    topup_reps: dict[str, list[bool | None]] = {i: [] for i in topup_ids}
    pending2: list[tuple[CorpusItem, int]] = []
    for i in topup_ids:
        for r in range(k, k + topup):
            graded, incomplete = _classify(by_id[i], records.get(f"{i}::{r}"))
            if incomplete:
                pending2.append((by_id[i], r))
            topup_reps[i].append(graded)
    await run_round(
        [
            _job(it, r, max_tokens=retry_tokens, seed_base=seed_base, retry=True)
            for it, r in pending2
        ]
    )
    for it, r in pending2:
        graded, _ = _classify(it, records.get(f"{it.id}::{r}{RETRY_SUFFIX}"))
        topup_reps[it.id][r - k] = graded

    # -- combine -------------------------------------------------------------
    for it in items:
        if it.id in outcomes:
            continue
        all_reps = repeats[it.id] + topup_reps.get(it.id, [])
        valid = [x for x in all_reps if x is not None]
        rate = (sum(1 for x in valid if x) / len(valid)) if valid else None
        notes: list[str] = []
        base_missing = sum(1 for x in repeats[it.id] if x is None)
        if base_missing:
            notes.append(f"missing_repeats={base_missing}/{len(repeats[it.id])}")
        if it.id in topup_reps:
            notes.append(f"topup+{topup}")
        outcomes[it.id] = (rate, "; ".join(notes))
    return outcomes


# --------------------------------------------------------------------------
# Stage entry.
# --------------------------------------------------------------------------


def stage_entry(
    run_id: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    gateway: Gateway | None = None,
    generation_out: Path | None = None,
    seed_base: int = 0,
    client_factory: Any | None = None,
) -> StageManifest:
    """S12: label pass@k difficulty on 11_judge -> 12_difficulty.

    Streams the input snapshot once, sampling and grading only labelable rows
    (gold answer + mcqa/numeric type); every row -- labelled or not -- is written
    through unchanged otherwise, so rows_out == rows_in. Optional dirs let tests
    run against tmp_path; ``generation_out`` pins the resume file (default:
    ``generations.jsonl`` beside the snapshot, so a re-run resumes its own
    completions by key).

    A missing input snapshot or labelable rows without a gateway is a
    StageError: a silent zero-label pass would look like "everything is
    unlabelled" downstream instead of the orchestration bug it is.
    """
    started = utcnow()
    inp = input_dir if input_dir is not None else store.stage_dir(run_id, INPUT_STAGE)
    out = output_dir if output_dir is not None else store.stage_dir(run_id, OUTPUT_STAGE)
    if not inp.is_dir():
        raise StageError(f"S12: input snapshot {inp} does not exist -- run {INPUT_STAGE} first")
    out.mkdir(parents=True, exist_ok=True)  # explicit dirs skip stage_dir's mkdir
    gen_path = generation_out if generation_out is not None else out / "generations.jsonl"

    eligible: list[CorpusItem] = []
    skipped: Counter[str] = Counter()
    rows_in = 0
    buf: list[CorpusItem] = []
    for it in store.iter_items(inp):
        rows_in += 1
        if labelable(it):
            eligible.append(it)
        else:
            skipped[
                "no_gold_answer" if it.answer_type in ("mcqa", "numeric") else it.answer_type
            ] += 1
            buf.append(it)
        if len(buf) >= _WRITE_CHUNK:
            store.write_items(buf, out)
            buf.clear()
    if buf:
        store.write_items(buf, out)

    gen_rounds: list[dict[str, Any]] = []
    outcomes: dict[str, tuple[float | None, str]] = {}
    if eligible:
        if gateway is None:
            raise StageError(
                f"S12: {len(eligible)} labelable rows but no gateway -- construct "
                "serving.Gateway(ServerHandle(model, port)) to run pass@k sampling"
            )
        gen_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # manifest write precedes run_generation's
        write_generation_manifest(gen_path, gateway.handle.model, "unknown", seed_base)
        outcomes = asyncio.run(
            _label_pass(
                gateway,
                eligible,
                gen_path,
                seed_base=seed_base,
                client_factory=client_factory,
                gen_rounds=gen_rounds,
            )
        )

    by_band: Counter[str] = Counter()
    per_source_rows: Counter[str] = Counter()
    per_source_labelled: dict[str, Counter[str]] = defaultdict(Counter)
    unrateable = 0
    buf = []
    for it in eligible:
        per_source_rows[it.source] += 1
        rate, note = outcomes[it.id]
        if rate is None:
            unrateable += 1
            meta = dict(it.meta)
            meta["difficulty_note"] = note or "unrateable: no pass rate"
            buf.append(
                it.model_copy(update={"difficulty": None, "difficulty_band": None, "meta": meta})
            )
            continue
        band = route_band(rate)
        by_band[band] += 1
        per_source_labelled[it.source][band] += 1
        updates: dict[str, Any] = {"difficulty": rate, "difficulty_band": band}
        if note:
            meta = dict(it.meta)
            meta["difficulty_note"] = note
            updates["meta"] = meta
        buf.append(it.model_copy(update=updates))
        if len(buf) >= _WRITE_CHUNK:
            store.write_items(buf, out)
            buf.clear()
    if buf:
        store.write_items(buf, out)

    rows_out = store.count_items(out)
    manifest = StageManifest(
        run_id=run_id,
        stage=OUTPUT_STAGE,
        started_at=started,
        config={
            "input_dir": str(inp),
            "output_dir": str(out),
            "generation_out": str(gen_path),
            "seed_base": seed_base,
            "seed_formula": "seed_base + hash_text(id) % 10**6 * passk_samples + repeat",
            "retry_max_tokens": THRESHOLDS.passk_max_tokens + THRESHOLDS.passk_think_budget // 2,
            "label_rule": "answer present and answer_type in {mcqa, numeric}",
            "model": gateway.handle.model if gateway else None,
        },
        thresholds=snapshot(THRESHOLDS),
        rows_in=rows_in,
        rows_out=rows_out,
        input_sha256=store.content_sha256(inp),
        output_sha256=store.content_sha256(out),
        # S12 sets no Flags fields (the schema reserves none for difficulty);
        # the routing distribution travels in notes instead of a pseudo flag rate.
        flag_rates={},
        notes={
            "labelled": sum(by_band.values()),
            "unrateable_too_few_valid_repeats": unrateable,
            "not_labelled": dict(skipped),
            "by_band": dict(by_band),
            "labelled_rates": {
                src: {
                    "_all": (sum(per_source_labelled[src].values()) / n) if n else 0.0,
                    **{
                        band: per_source_labelled[src][band] / n
                        for band in sorted(per_source_labelled[src])
                    },
                }
                for src, n in sorted(per_source_rows.items())
            },
            "gen_rounds": gen_rounds,
            "knob_notes": [
                "retry headroom = passk_think_budget // 2 (= 4096 at shipped values); "
                "the audit's +4096 is not a THRESHOLDS knob",
                f"MAX_MISSING_REPEATS = {MAX_MISSING_REPEATS} (audit spec; not in THRESHOLDS)",
                "top_p = gen_top_p (no passk_top_p knob; equals run_generation's default)",
            ],
        },
    )
    assert rows_out == rows_in, "flags-not-deletes: S12 never drops"
    return manifest


__all__ = [
    "INPUT_STAGE",
    "MAX_MISSING_REPEATS",
    "OUTPUT_STAGE",
    "RETRY_SUFFIX",
    "aggregate",
    "band_stable",
    "grade_sample",
    "labelable",
    "route_band",
    "sample_seed",
    "sampling_messages",
    "stage_entry",
    "think_incomplete",
]
