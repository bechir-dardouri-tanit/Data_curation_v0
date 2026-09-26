"""S11 -- batch data-quality judging of the survivor corpus on three binary axes.

The plan's judge pass (CURATION_IMPLEMENTATION_PLAN.md, S11) scores every row the cheap
stages did not condemn -- precisely ``not item.flags.any()`` -- on three axes persisted
to the schema's reserved ``q_coherence`` / ``q_clinical`` / ``q_format`` columns:
coherence (self-consistent, on-topic, non-circular), clinical safety (contraindications,
escalation/uncertainty, dosing sanity) and formatting (terminal MCQA answer line, units
on numerics, no roleplay leakage). Criteria live in ``configs/curation/judge_axes.yaml``
and are binary and weighted on purpose: a 0/1 axis verdict stays interpretable after
threshold recalibration and is queryable mixture SQL, unlike a holistic 1-10 score
nobody can reproduce. The axes grade the trained response surface only -- trace defects
are S2's deterministic checks -- so the prompt carries the user turn and the assistant
turn and nothing else.

Wrapping, not reimplementing: criteria are the eval judge's weighted
:class:`~medrl.eval.scorers.judge.Criterion` and verdict parsing delegates to that
module's tolerant met-ids extractor (the plan forbids a second parser: if eval and
curation ever disagreed on what a judge reply means, the corpus and the reported
benchmarks would drift apart silently). Requests run through
:mod:`medrl.curation.serving` -- Gateway + ``run_generation``, the pipeline's one
serving path. The judge model (``Qwen/Qwen3.5-4B``) serves with thinking OFF at the
server level (plan 1.4), so the gateway sends plain chat and replies stay a few dozen
tokens.

Reliability of an unreliable instrument: a reply that yields no parseable met-ids JSON
is re-asked once with the eval judge's nudge, then scored as conservatively zero. A row
we could not verify must never enter the quality pool as verified, and a 1.8M-call batch
must never die on one malformed reply; the conservative zeros are counted in the
manifest notes so the honest failure rate stays visible.

Resume, two layers. First, before a re-run replaces its own snapshot (the runner
contract), the previous snapshot's q_* values are harvested by row id and overlaid on
the freshly streamed input -- a re-run streams the upstream stage, whose q_* columns
are always empty, so the prior verdicts live only here; rows left fully judged by the
interrupted pass are then skipped outright and partially judged rows re-ask only their
missing axes. Second, ``verdicts.jsonl`` beside the snapshot caches raw judge replies
keyed ``<item_id>::<axis>`` (``run_generation``'s resume contract), so verdicts for
rows the process died in front of are never re-billed either.

Flagged rows pass through untouched and are never sent to the model: GPU judging is the
most expensive per-row work in the pipeline, and rows the mixtures will never see
should not buy a verdict.

Threshold note: the pass threshold defaults to ``THRESHOLDS.band_sft1_low`` because
thresholds.py has no judge-specific knob (constraint: knobs only from THRESHOLDS, add
none). The value is equal (0.5) and the borrow is recorded in the manifest notes; a
``judge_pass_threshold`` entry should land with the axis calibration.
"""

from __future__ import annotations

import asyncio
import json
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from medrl.core.logging import get_logger
from medrl.core.paths import repo_root
from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow
from medrl.curation.serving import Gateway, ServerHandle, run_generation
from medrl.curation.thresholds import THRESHOLDS, snapshot
from medrl.eval.scorers.judge import Criterion, _parse_met_ids

log = get_logger(__name__)

INPUT_STAGE = "09_answers"
OUTPUT_STAGE = "11_judge"
"""Graph note: the plan places S11 after S10 grounding, which is not built; 09_answers
is the last built upstream stage, so it is the default input until S7/S8/S10 land."""

JUDGE_MODEL = "Qwen/Qwen3.5-4B"
"""Fast-tier judge per the plan. Its verdicts are filter inputs only after the
calibration gate (4B-vs-stronger-judge agreement on ~2k labeled rows) passes."""

JUDGE_PORT = 8102
"""Plan 1.4: the judge vLLM serves on 8102 with thinking off at the server level."""

AXES_FILENAME = "judge_axes.yaml"
DEFAULT_AXES_FILE = repo_root() / "configs" / "curation" / AXES_FILENAME

VERDICTS_FILENAME = "verdicts.jsonl"
"""Raw judge replies cached beside the snapshot; the resume layer that makes an
interrupted corpus-wide pass cheap (run_generation resumes by key)."""

RETRY_KEY_SUFFIX = "::retry1"
"""Second (and last) attempt key: distinct from the primary so run_generation's
key-resume never mistakes the re-ask for already-done work."""

VERDICT_MAX_TOKENS = 512
"""The reply is a tiny JSON object; the cap mirrors eval LLMJudge's documented default
so a misconfigured thinking-on server ruminates into a truncation, not a long bill."""

CHUNK_ITEMS = 10_000
"""Judged rows per generation round: bounds peak job memory and makes partial progress
land in the verdict cache frequently."""

WRITE_ROWS = 50_000
"""Snapshot flush size (the answers/embed convention)."""

AXIS_TO_Q: dict[str, str] = {
    "coherence": "q_coherence",
    "clinical": "q_clinical",
    "formatting": "q_format",
}
"""The three schema-reserved S11 axes; any other axis in the YAML is a StageError."""

_SYSTEM = "You are a precise rubric grader. Respond with JSON only."
_RETRY_NOTE = " Respond with JSON only."
"""The eval judge's exact system string and retry nudge (scorers/judge.py) -- the
nudge duplicates a sentence the system already contains; kept verbatim anyway so both
graders see byte-identical retry pressure."""

_THRESHOLD_SOURCE = (
    "pass threshold defaults to THRESHOLDS.band_sft1_low: thresholds.py has no judge "
    "knob; the value (0.5) is the plan's axis pass threshold -- add "
    "judge_pass_threshold at calibration"
)

_PROMPT = """You are grading a medical AI assistant's response against binary criteria.

Question:
{question}

Assistant response:
{response}

Criteria (grade each one independently):
{criteria}

A criterion is met only if the response clearly satisfies it. Judge the response alone,
not what it could have said. Respond with JSON only, of the exact form:
{{"met": ["<id>", ...]}}
listing the ids of every met criterion. Omitted ids count as not met."""


def build_axis_prompt(item: CorpusItem, criteria: Sequence[Criterion]) -> str:
    """The judge prompt for one (item, axis): question + response + that axis's criteria.

    Pure, so the exact model-visible bytes are unit-testable. Criteria render id-first
    (the met-ids protocol references ids) and without weights: weights are stage-side
    arithmetic, and showing them only invites the judge to ration attention instead of
    deciding met/not-met per criterion.
    """
    question = "\n".join(m["content"] for m in item.messages if m["role"] == "user")
    response = "\n".join(m["content"] for m in item.messages if m["role"] == "assistant")
    listed = "\n".join(f"- id={c.id!r}: {c.text}" for c in criteria)
    return _PROMPT.format(question=question, response=response, criteria=listed)


def parse_verdict(text: str, criteria: Sequence[Criterion]) -> set[str]:
    """Met ids from a judge reply, restricted to the axis's known ids.

    Delegates to the eval judge's tolerant extractor (whole reply first, then the first
    ``{...}`` block -- judges decorate JSON with prose and fences far more often than
    they emit malformed JSON). Unparseable input yields the empty set, the caller's
    conservative zero, and ids the axis never defined are dropped: a judge inventing
    ids must not be able to move a met-fraction.
    """
    ids = _parse_met_ids(text)
    if ids is None:
        return set()
    return ids & {c.id for c in criteria}


def axis_score(met_ids: set[str], criteria: Sequence[Criterion]) -> float:
    """Weighted met-fraction in [0, 1]: met weight over total weight.

    Axis weights are positive by design (harm lives in the flag stages), so this equals
    the eval rubric's positive-denominator HealthBench aggregate.
    """
    total = sum(c.weight for c in criteria)
    if total <= 0:
        return 0.0
    return sum(c.weight for c in criteria if c.id in met_ids) / total


def axis_verdict(met_ids: set[str], criteria: Sequence[Criterion], threshold: float) -> int:
    """0/1 axis outcome: met-fraction >= threshold (inclusive) -> 1."""
    return int(axis_score(met_ids, criteria) >= threshold)


def load_axes(path: Path) -> dict[str, tuple[Criterion, ...]]:
    """Parse judge_axes.yaml into eval-judge Criteria, validated against AXIS_TO_Q.

    Fail-loud before any GPU spend: an axis set that does not match the three schema
    columns, duplicate ids inside an axis, or a malformed/zero-weight criterion (one
    that cannot change a score is a rubric bug, per the eval judge) stops the stage.
    """
    raw = yaml.safe_load(path.read_text())
    axes = raw.get("axes") if isinstance(raw, dict) else None
    if not isinstance(axes, dict) or not axes:
        raise StageError(f"{path}: expected a top-level 'axes' mapping")
    unknown = sorted(set(axes) - set(AXIS_TO_Q))
    missing = sorted(set(AXIS_TO_Q) - set(axes))
    if unknown or missing:
        raise StageError(
            f"{path}: axes must be exactly {sorted(AXIS_TO_Q)} "
            f"(unknown={unknown}, missing={missing})"
        )
    out: dict[str, tuple[Criterion, ...]] = {}
    for name, spec in sorted(axes.items()):
        entries = spec.get("criteria") if isinstance(spec, dict) else None
        if not isinstance(entries, list) or not entries:
            raise StageError(f"{path}: axis {name!r} needs a non-empty criteria list")
        try:
            criteria = tuple(
                Criterion(id=str(e["id"]), text=str(e["text"]), weight=float(e.get("weight", 1.0)))
                for e in entries
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StageError(f"{path}: bad criterion in axis {name!r}: {exc}") from exc
        ids = [c.id for c in criteria]
        if len(set(ids)) != len(ids):
            raise StageError(f"{path}: axis {name!r} has duplicate criterion ids")
        out[name] = criteria
    return out


def _stable_seed(key: str) -> int:
    """Per-key deterministic seed: re-runs reproduce; retries seed differently."""
    return zlib.crc32(key.encode("utf-8"))


def _judge_jobs(
    item: CorpusItem,
    missing: Sequence[str],
    axes: Mapping[str, tuple[Criterion, ...]],
    *,
    retry: bool = False,
) -> list[dict[str, Any]]:
    """``run_generation`` job dicts for one item's missing axes -- one prompt per axis.

    Grading at temperature 0: the axes are a measurement, and a measurement that
    resamples itself between runs cannot be calibrated.
    """
    system = _SYSTEM + _RETRY_NOTE if retry else _SYSTEM
    suffix = RETRY_KEY_SUFFIX if retry else ""
    jobs: list[dict[str, Any]] = []
    for axis in missing:
        key = f"{item.id}::{axis}{suffix}"
        jobs.append(
            {
                "key": key,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": build_axis_prompt(item, axes[axis])},
                ],
                "max_tokens": VERDICT_MAX_TOKENS,
                "temperature": 0.0,
                "seed": _stable_seed(key),
            }
        )
    return jobs


def _merge_line(line: bytes, records: dict[str, dict[str, Any]]) -> None:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return  # a torn last line from a killed process must not kill the re-run
    if isinstance(rec, dict) and "key" in rec:
        records[str(rec["key"])] = rec


def _scan_verdicts(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Existing verdict records (key -> line object) plus the file's end offset.

    One full scan up front builds the resume set; :func:`_tail_verdicts` then merges
    only lines appended after that offset, so a chunked corpus run reads each JSONL
    byte exactly once.
    """
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    if path.exists():
        with path.open("rb") as f:
            for line in f:
                offset += len(line)
                _merge_line(line, records)
    return records, offset


def _tail_verdicts(path: Path, records: dict[str, dict[str, Any]], offset: int) -> int:
    """Merge lines appended since ``offset``; returns the new end offset."""
    if not path.exists():
        return offset
    with path.open("rb") as f:
        f.seek(offset)
        for line in f:
            offset += len(line)
            _merge_line(line, records)
    return offset


def _needs_retry(verdicts: Mapping[str, Mapping[str, Any]], key: str) -> bool:
    """True when the primary slot holds no parseable verdict (missing, error, or junk).

    The eval judge's retry condition verbatim: only unparseable output is re-asked --
    a parseable empty met-list is a legitimate "fails every criterion", not a failure.
    """
    rec = verdicts.get(key)
    if not rec:
        return True
    return _parse_met_ids(str(rec.get("content") or "")) is None


def _resolve_met(
    verdicts: Mapping[str, Mapping[str, Any]], key: str, criteria: Sequence[Criterion]
) -> tuple[set[str], bool]:
    """(met ids, got a parseable verdict) for one (item, axis).

    The retry line supersedes the primary -- a re-ask replaces, not increments. An
    error record (run_generation writes one per failed job) carries no content, so the
    other slot still gets its chance before the caller's conservative zero.
    """
    for k in (key + RETRY_KEY_SUFFIX, key):
        rec = verdicts.get(k)
        if not rec:
            continue
        content = rec.get("content")
        if not content:
            continue
        ids = _parse_met_ids(str(content))
        if ids is not None:
            return ids & {c.id for c in criteria}, True
    return set(), False


def stage_entry(
    run_id: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    axes_path: Path | None = None,
    model: str = JUDGE_MODEL,
    port: int = JUDGE_PORT,
    gateway: Gateway | None = None,
    client_factory: Any | None = None,
    verdicts_path: Path | None = None,
    pass_threshold: float | None = None,
    limit: int | None = None,
) -> StageManifest:
    """S11: judge every un-flagged row of the input snapshot on the three axes.

    Streams the input once: flagged rows pass through untouched, survivors missing
    axis verdicts go through the gateway (retry-once on unparseable replies, then
    conservative zero), and every row lands in the output snapshot with ``q_*``
    filled. Already-judged rows (verdicts harvested from a previous snapshot, or set
    upstream) pass through untouched. ``gateway``/``client_factory`` are the injection
    seams tests use instead of a live vLLM server; ``limit`` slices pilot runs;
    ``pass_threshold`` overrides the borrowed THRESHOLDS default. Production calls
    take only ``run_id``.
    """
    started = utcnow()
    inp = input_dir if input_dir is not None else store.stage_dir(run_id, INPUT_STAGE)
    out = output_dir if output_dir is not None else store.stage_dir(run_id, OUTPUT_STAGE)
    if not inp.is_dir():
        raise StageError(f"S11: input snapshot {inp} does not exist -- run {INPUT_STAGE} first")
    out.mkdir(parents=True, exist_ok=True)
    # Runner resume contract: a re-run replaces its own snapshot. Before replacing it,
    # harvest the previous run's verdicts by row id (resume layer 1); the verdict cache
    # is stage state too and survives so finished rows are never re-billed.
    prior_q: dict[str, tuple[int | None, int | None, int | None]] = {}
    if any(out.glob("part-*.parquet")):
        for done in store.iter_items(out):
            prior_q[done.id] = (done.q_coherence, done.q_clinical, done.q_format)
    for stale in out.glob("part-*.parquet"):
        stale.unlink()

    axes_file = axes_path if axes_path is not None else DEFAULT_AXES_FILE
    if not axes_file.is_file():
        raise StageError(f"S11: axes file {axes_file} not found -- pass axes_path")
    axes = load_axes(axes_file)
    threshold = THRESHOLDS.band_sft1_low if pass_threshold is None else pass_threshold
    gw = gateway if gateway is not None else Gateway(ServerHandle(model=model, port=port))
    vpath = verdicts_path if verdicts_path is not None else out / VERDICTS_FILENAME

    manifest = StageManifest(
        run_id=run_id,
        stage=OUTPUT_STAGE,
        started_at=started,
        config={
            "input_dir": str(inp),
            "output_dir": str(out),
            "axes_path": str(axes_file),
            "axes_sha256": store.source_sha256(axes_file),
            "axes": {axis: [c.id for c in criteria] for axis, criteria in axes.items()},
            "model": model,
            "base_url": gw.handle.base_url,
            "pass_threshold": threshold,
            "verdicts_path": str(vpath),
            "temperature": 0.0,
            "max_tokens": VERDICT_MAX_TOKENS,
        },
        thresholds=snapshot(THRESHOLDS),
    )

    verdicts, offset = _scan_verdicts(vpath)
    survivors: dict[str, int] = defaultdict(int)
    axis_passes: dict[str, dict[str, int]] = {axis: defaultdict(int) for axis in AXIS_TO_Q}
    gen_stats: dict[str, int] = {"ran": 0, "resumed": 0, "failed": 0}
    zeros: dict[str, int] = defaultdict(int)
    out_buf: list[CorpusItem] = []
    chunk: list[tuple[CorpusItem, list[str]]] = []
    n_seen = n_flagged = n_judged = n_skipped = 0
    n_retry_jobs = 0

    def _tally(it: CorpusItem) -> None:
        """Axis pass rates accumulate over the WHOLE survivor population (judged this
        run or previously), so rates stay comparable across resumed runs."""
        survivors[it.source] += 1
        for axis, qf in AXIS_TO_Q.items():
            if getattr(it, qf) == 1:
                axis_passes[axis][it.source] += 1

    def _flush() -> None:
        if out_buf:
            store.write_items(out_buf, out)
            out_buf.clear()

    def _round(jobs: list[dict[str, Any]]) -> None:
        """One run_generation pass, then merge the lines it appended."""
        nonlocal offset
        if not jobs:
            return
        stats = asyncio.run(run_generation(gw, jobs, vpath, client_factory=client_factory))
        offset = _tail_verdicts(vpath, verdicts, offset)
        for name in gen_stats:
            gen_stats[name] += int(stats.get(name, 0))

    def _close_chunk() -> None:
        """Generate, retry, resolve and emit the chunk's judged rows."""
        nonlocal n_retry_jobs
        if not chunk:
            return
        jobs = [j for item, missing in chunk for j in _judge_jobs(item, missing, axes)]
        _round(jobs)
        retries: list[dict[str, Any]] = []
        for item, missing in chunk:
            for axis in missing:
                if _needs_retry(verdicts, f"{item.id}::{axis}"):
                    retries.extend(_judge_jobs(item, [axis], axes, retry=True))
        n_retry_jobs += len(retries)
        _round(retries)
        for item, missing in chunk:
            updates: dict[str, int] = {}
            for axis in missing:
                met, parseable = _resolve_met(verdicts, f"{item.id}::{axis}", axes[axis])
                if not parseable:
                    zeros[axis] += 1
                updates[AXIS_TO_Q[axis]] = axis_verdict(met, axes[axis], threshold)
            judged = item.model_copy(update=updates)
            _tally(judged)
            out_buf.append(judged)
        chunk.clear()

    for item in store.iter_items(inp):
        if limit is not None and n_seen >= limit:
            break
        n_seen += 1
        if item.flags.any():
            n_flagged += 1  # never judged: the mixtures will never see it
            out_buf.append(item)
        else:
            prior = prior_q.get(item.id)
            if prior is not None:
                item = item.model_copy(update=dict(zip(AXIS_TO_Q.values(), prior, strict=True)))
            missing = [axis for axis in AXIS_TO_Q if getattr(item, AXIS_TO_Q[axis]) is None]
            if missing:
                n_judged += 1
                chunk.append((item, missing))
            else:
                n_skipped += 1  # fully judged by a previous pass (or the input itself)
                _tally(item)
                out_buf.append(item)
        if len(out_buf) >= WRITE_ROWS:
            _flush()
        if len(chunk) >= CHUNK_ITEMS:
            _close_chunk()
    _close_chunk()
    _flush()

    if n_seen == 0:
        raise StageError(f"S11: input snapshot {inp} is empty -- run {INPUT_STAGE} first")
    manifest.rows_in = n_seen
    manifest.rows_out = store.count_items(out)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)

    total_survivors = sum(survivors.values())
    axis_pass_rates: dict[str, dict[str, float]] = {}
    for axis, qf in AXIS_TO_Q.items():
        per_source = {src: axis_passes[axis][src] / n for src, n in sorted(survivors.items())}
        per_source["_all"] = (
            sum(axis_passes[axis].values()) / total_survivors if total_survivors else 0.0
        )
        axis_pass_rates[qf] = per_source

    manifest.notes = {
        "judged": n_judged,
        "skipped": n_skipped,
        "flagged_passthrough": n_flagged,
        "axis_pass_rates": axis_pass_rates,
        "unparseable_after_retry": dict(zeros),
        "retry_jobs": n_retry_jobs,
        "gen": gen_stats,
        "verdicts_cached": len(verdicts),
        "pass_threshold_source": _THRESHOLD_SOURCE,
    }
    assert manifest.rows_in == manifest.rows_out, "flags-not-deletes: S11 never drops"
    log.info(
        "S11 %s: judged=%d skipped=%d flagged=%d pass_rates=%s",
        run_id,
        n_judged,
        n_skipped,
        n_flagged,
        {qf: rates["_all"] for qf, rates in axis_pass_rates.items()},
    )
    return manifest


__all__ = [
    "AXES_FILENAME",
    "AXIS_TO_Q",
    "DEFAULT_AXES_FILE",
    "INPUT_STAGE",
    "JUDGE_MODEL",
    "JUDGE_PORT",
    "OUTPUT_STAGE",
    "RETRY_KEY_SUFFIX",
    "VERDICTS_FILENAME",
    "VERDICT_MAX_TOKENS",
    "axis_score",
    "axis_verdict",
    "build_axis_prompt",
    "load_axes",
    "parse_verdict",
    "stage_entry",
]
