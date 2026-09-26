"""S4 -- lexical decontamination of the corpus against the eval benchmark suite.

Training on (near-)copies of graded eval items inflates reported scores into
fiction; the eval registry (``medrl.eval.tasks.benchmarks``) states the policy:
*every* train-side source is decontaminated against *every* registered name.
This stage therefore resolves its benchmark set from that registry by default --
the index and the graded suite cannot drift apart because there is one list.

Wraps, never reimplements, ``medrl.data.decontam.BenchmarkIndex``: the same
``medrl.data.dedup.ngram_hashes`` primitive S3 dedup uses, so dedup and
decontamination cannot disagree about what "a 13-gram" is. Both sides of the
comparison go through ``question_text`` -- the KEY DELTA vs the data-side
``build_benchmark_index``, which joins every message content (system boilerplate
plus gold answer/explanation). We index and query *user-role content only*,
because that is the text the graded suite actually presents: a hit means the
corpus embeds something a model could memorise and replay at eval time. System
prompts are identical boilerplate across a benchmark (shared n-grams that
deflate every Jaccard score), and gold explanations are never shown at eval
time -- matching on them would flag legitimate clinical prose as leakage.

Flags-not-deletes: a hit sets ``f_contam_ngram`` and records the matched
benchmark name in ``contam_benchmark`` (strongest match, ties broken by
lexicographic key -- the index's own hit order is set-iteration order and not
stable across processes). No row is dropped; exclusion is a downstream query.

A benchmark that cannot be loaded raises rather than warns: a silently missing
index silently passes that benchmark's contaminated rows, which is the exact
failure this stage exists to prevent.

Durability/audit: the manifest carries ``per_benchmark_hits`` and a canary
block -- verbatim benchmark questions re-run through the live query path (must
all be detected) plus ``generate_negative_controls`` variants (whose detection
count measures over-aggressiveness at the operating point), so each run ships
its own evidence that detection worked.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger
from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow
from medrl.curation.thresholds import THRESHOLDS, snapshot
from medrl.data.decontam import BenchmarkIndex, BenchmarkItem, generate_negative_controls

log = get_logger(__name__)

STAGE_NAME = "04_decontam_ngram"
INPUT_STAGE = "03_dedup"

# Operational canary sizing, not a decision threshold: how many indexed items the
# canary samples (deterministic stride over sorted keys) and which control
# mutations generate_negative_controls runs. Recorded in the manifest config.
_CANARY_ITEMS = 8
_CANARY_SEED = 42
_CANARY_MODIFICATIONS = ("paraphrase", "shuffle", "negate", "perturb")


def question_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """User-role contents joined by blank lines -- the only text this stage compares.

    Used on BOTH sides (benchmark index and corpus query) so the comparison stays
    symmetric; benchmark prompt boilerplate and gold answers never enter it.
    """
    return "\n\n".join(
        str(msg["content"]) for msg in messages if msg.get("role") == "user" and msg.get("content")
    )


def build_question_index(benchmarks: Sequence[str]) -> BenchmarkIndex:
    """Index each named benchmark's user-role question text for n-gram lookup.

    Names must come from the eval registry (``medrl.eval.tasks.benchmarks.TASKS``);
    anything else is a StageError, because an index built over a typo cannot
    decontaminate against the suite that will actually grade the runs. Load
    failures raise for the same reason -- see the module docstring.
    """
    from medrl.eval.loaders import load_items
    from medrl.eval.tasks.benchmarks import TASKS

    index = BenchmarkIndex(ngram_n=THRESHOLDS.contam_ngram_n)
    for name in benchmarks:
        if name not in TASKS:  # Registry.get raises KeyError; we raise StageError with its hint
            known = ", ".join(sorted(TASKS))
            raise StageError(f"S4: benchmark {name!r} is not in the eval registry (known: {known})")
        spec = TASKS.get(name)
        try:
            result = load_items(spec)
        except Exception as exc:
            raise StageError(f"S4: cannot index benchmark {name!r}: {exc}") from exc
        n_before = len(index.items)
        for it in result.items:
            text = question_text(it.messages)
            if text.strip():  # an empty string would n-gram to hash('') and match nothing real
                index.add(BenchmarkItem(benchmark=name, item_id=it.item_id, text=text))
        log.info(f"S4 indexed {len(index.items) - n_before} question texts from {name}")
    if not index.items:
        raise StageError("S4: benchmark index is empty -- refusing to run a no-op decontamination")
    return index


def match_question(
    index: BenchmarkIndex, question: str
) -> tuple[bool, str | None, dict[str, float]]:
    """Query one question text against the index at the S4 operating point.

    Returns ``(hit, strongest_benchmark, per_benchmark_best_overlap)``. The
    strongest match maximises overlap with ties broken by the lexicographically
    smallest ``benchmark::item_id`` key, so ``contam_benchmark`` is reproducible
    across runs regardless of hash randomisation.
    """
    hits = index.query_ngram(question, threshold=THRESHOLDS.contam_ngram_threshold)
    if not hits:
        return False, None, {}
    best_per_benchmark: dict[str, float] = {}
    for key, overlap in hits:
        bench = index.items[key].benchmark
        if overlap > best_per_benchmark.get(bench, -1.0):
            best_per_benchmark[bench] = overlap
    strongest_key = min(hits, key=lambda kv: (-kv[1], kv[0]))[0]
    return True, index.items[strongest_key].benchmark, best_per_benchmark


def run_canary(
    index: BenchmarkIndex,
    *,
    n_items: int = _CANARY_ITEMS,
    seed: int = _CANARY_SEED,
) -> dict[str, Any]:
    """Detection evidence recorded into the manifest, from the index itself.

    Positives are verbatim indexed benchmark questions re-queried through the
    live path (self-match, overlap 1.0) -- they must all be detected or the run
    is not decontaminating. Negatives are ``generate_negative_controls`` variants
    of the same items; every one detected is a false positive at the current
    threshold. That count is expected to be non-zero for word-swap paraphrases
    under character n-grams: it is the calibration evidence the B7 pilot needs,
    recorded per run rather than argued in a review.
    """
    keys = sorted(index.items)
    if not keys:
        return {"status": "skipped", "reason": "empty index"}
    stride = max(1, len(keys) // n_items)
    sampled = [index.items[k] for k in keys[::stride][:n_items]]

    positives_detected = 0
    for item in sampled:
        detected, _, _ = match_question(index, item.text)
        positives_detected += int(detected)

    neg_detected: dict[str, int] = {}
    neg_total = 0
    for control in generate_negative_controls(
        sampled, modifications=list(_CANARY_MODIFICATIONS), seed=seed
    ):
        neg_total += 1
        detected, _, _ = match_question(index, control.text)
        if detected:
            neg_detected[control.modification] = neg_detected.get(control.modification, 0) + 1

    return {
        "status": "ok",
        "n_positive": len(sampled),
        "n_positive_detected": positives_detected,
        "n_negative_controls": neg_total,
        "n_negative_detected": sum(neg_detected.values()),
        "negative_detected_by_modification": neg_detected,
        "modifications": list(_CANARY_MODIFICATIONS),
    }


def _flag_stream(
    items: Iterable[CorpusItem],
    index: BenchmarkIndex,
    per_source_total: Counter[str],
    per_source_flagged: Counter[str],
    per_benchmark_hits: Counter[str],
) -> Iterator[CorpusItem]:
    """Yield every item unchanged or flag-updated, filling the manifest counters.

    ``per_benchmark_hits`` counts rows per benchmark matched, so one row matching
    two benchmarks increments both (rows flagged total is ``flag_rates['_all']``).
    """
    for item in items:
        per_source_total[item.source] += 1
        hit, strongest, per_benchmark = match_question(index, question_text(item.messages))
        if not hit:
            yield item
            continue
        assert strongest is not None  # match_question guarantees a name on a hit
        per_source_flagged[item.source] += 1
        for bench in per_benchmark:
            per_benchmark_hits[bench] += 1
        yield item.model_copy(
            update={
                "flags": item.flags.model_copy(update={"f_contam_ngram": True}),
                "contam_benchmark": strongest,
            }
        )


def _resolve_benchmarks(benchmarks: Sequence[str] | None) -> list[str]:
    """Explicit names, or every name the eval registry grades -- never a hand list."""
    if benchmarks is not None:
        return list(benchmarks)
    from medrl.eval.tasks.benchmarks import TASKS

    return sorted(TASKS)


def run_decontam_ngram(
    run_id: str,
    *,
    benchmarks: Sequence[str] | None = None,
    index: BenchmarkIndex | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream 03_dedup -> 04_decontam_ngram, flagging benchmark-contaminated rows.

    ``index`` (and the dir overrides) exist so tests and subset runs can inject a
    synthetic index and tmp_path dirs; production resolves every registry
    benchmark and the run-derived stage directories.
    """
    inp = input_dir if input_dir is not None else store.stage_dir(run_id, INPUT_STAGE)
    out = output_dir if output_dir is not None else store.stage_dir(run_id, STAGE_NAME)
    # store.stage_dir mkdirs its own default; an injected (subset/test) dir must too.
    if input_dir is not None:
        inp.mkdir(parents=True, exist_ok=True)
    if output_dir is not None:
        out.mkdir(parents=True, exist_ok=True)
    names = _resolve_benchmarks(benchmarks)
    if index is None:
        index = build_question_index(names)
    canary = run_canary(index)

    per_source_total: Counter[str] = Counter()
    per_source_flagged: Counter[str] = Counter()
    per_benchmark_hits: Counter[str] = Counter()
    store.write_items(
        _flag_stream(
            store.iter_items(inp), index, per_source_total, per_source_flagged, per_benchmark_hits
        ),
        out,
    )

    rates = {
        src: (per_source_flagged[src] / total if total else 0.0)
        for src, total in sorted(per_source_total.items())
    }
    n_total = sum(per_source_total.values())
    rates["_all"] = (sum(per_source_flagged.values()) / n_total) if n_total else 0.0

    return {
        "input_dir": inp,
        "stage_dir": out,
        "benchmarks": names,
        "index_items": len(index.items),
        "rows_in": store.count_items(inp),
        "rows_out": store.count_items(out),
        "flag_rates": rates,
        "per_benchmark_hits": dict(sorted(per_benchmark_hits.items())),
        "canary": canary,
    }


def stage_entry(
    run_id: str,
    *,
    benchmarks: Sequence[str] | None = None,
    index: BenchmarkIndex | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> StageManifest:
    """Runner adapter: run S4 over the run's snapshots and return its manifest."""
    started = utcnow()
    result = run_decontam_ngram(
        run_id,
        benchmarks=benchmarks,
        index=index,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    manifest = StageManifest(
        run_id=run_id,
        stage=STAGE_NAME,
        started_at=started,
        config={
            "benchmarks": result["benchmarks"],
            "index_scope": "user_role_question_text_only",
            "ngram_n": THRESHOLDS.contam_ngram_n,
            "ngram_threshold": THRESHOLDS.contam_ngram_threshold,
            "n_index_items": result["index_items"],
            "canary_items": _CANARY_ITEMS,
            "input_dir": str(result["input_dir"]),
            "output_dir": str(result["stage_dir"]),
        },
        thresholds=snapshot(),
        rows_in=result["rows_in"],
        rows_out=result["rows_out"],
        input_sha256=store.content_sha256(result["input_dir"]),
        output_sha256=store.content_sha256(result["stage_dir"]),
        flag_rates={"f_contam_ngram": result["flag_rates"]},
        notes={"per_benchmark_hits": result["per_benchmark_hits"], "canary": result["canary"]},
    )
    assert manifest.rows_in == manifest.rows_out, "flags-not-deletes: S4 never drops"
    return manifest


__all__ = [
    "INPUT_STAGE",
    "STAGE_NAME",
    "build_question_index",
    "match_question",
    "question_text",
    "run_canary",
    "run_decontam_ngram",
    "stage_entry",
]
