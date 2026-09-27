"""S8 -- concept-level dedup + cross-lingual decontamination on linked concept ids.

The stage with no substitute. A French translation of an English question shares
no word 13-grams (invisible to S3/S4) and sits below the embedding thresholds
(invisible to S5/S6) -- every text-level signal breaks under translation. What
survives translation is the terminology: both surfaces resolve to the same
concept ids at S7-style linking, so concept identity catches the pair exactly
where the lexical and semantic passes went blind.

Two joins over the linked ids:

* self-join (dup): rows whose question-CUI SET is identical AND whose
  ``answer_cui`` is identical carry the same training signal at concept
  resolution -> ``f_dup_concept`` + ``dup_of`` (canonical: the lower id, S6's
  rule -- these rows are downstream of S3's licence-aware keep-rule, so id
  order is the reproducible total order, not a slight on it).
* eval-join (contam): a row whose question concepts overlap an eval item at
  Jaccard >= ``concept_jaccard`` AND whose ``answer_cui`` equals that item's is
  a replayable eval item in new clothes -> ``f_contam_concept`` +
  ``contam_benchmark``.

The asymmetry between the two joins is deliberate. Dedup claims *identity*, so
it requires exact set equality: Jaccard-based grouping would chain-merge
distinct questions that merely share a prevalent concept (every case mentioning
hypertension would become one row). Contamination claims *leakage*, where the
dangerous object is the translated/paraphrased eval item, so it tolerates the
fuzzy bound -- and pays for that tolerance with the second condition. Either
condition alone over-flags: question concepts alone flag every row that
mentions a common concept the suite also mentions, and the answer concept alone
flags every row whose gold answer is the suite's gold answer. Only
"same question, same answer" at concept resolution is evidence of leakage.

The answer_cui participates in the DUP join for the same reason S3 puts the
gold answer in ``dedup_text``: two rows about the same concepts that assert
different answers are different training signal, not duplicates. Rows whose
answers never linked (``answer_cui`` None on both sides) still compare -- for
Pool B the exact-set requirement IS the whole identity, and translation pairs
of answer-less rows are precisely the cross-lingual duplicates this stage
exists to catch. Rows with NO linked question concepts carry no identity at all
and never join (S3's empty-transcript convention); merging every unlinked row
into one group would flag the unlinked corpus instead of deduplicating it.

Eval items are injected, never discovered. The S4-style index over real
terminologies requires the licence-gated S7 stack, so the suite arrives as the
explicit ``eval_items`` parameter (records, or plain
``(benchmark, cuis, answer_cui)`` tuples); ``None`` skips the contam half and
says so in the manifest -- the S6 precedent (absent benchmark vectors ->
self-join only), never a silent no-op.

The ids are opaque strings. Whether they are CUIs, SNOMED CT sctids or OMOP
concept_ids cannot matter to the set algebra; the namespace label is recorded
verbatim in the manifest config so a UMLS->OMOP backbone swap is a diff in the
manifest, not a code change (plan section 1.3).

The plan sketched DuckDB set-ops over the parquet ``cuis`` columns. Sets are
small (only linked rows participate), so python sets plus one inverted index
(CUI -> eval records) keep this stage importable without the DuckDB pin and
deterministic without a database; the DuckDB path can replace the two passes
behind the same function signatures when S7 makes the linked row counts real.

Flags-not-deletes throughout: a hit sets its flag and the row stays, so every
dedup/contam decision stays auditable and reversible at the mixture-query
level, and re-running the stage over its own output changes no row.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow
from medrl.curation.thresholds import THRESHOLDS, snapshot

STAGE = "08_concept"
INPUT_STAGE = "06_decontam_sem"
"""S7 linking is licence-gated and not built, so the default input is the last
BUILT upstream snapshot. The stage reads only the ``cuis``/``answer_cui``
columns, never a particular producer: when 07_linking lands, the runner wiring
points ``input_dir`` at it and this module does not change."""

STAGE_FLAGS: tuple[str, ...] = ("f_dup_concept", "f_contam_concept")

DUP_IDENTITY = "sorted question-CUI set + answer_cui"
CONTAM_RULE = "question-CUI jaccard >= concept_jaccard AND answer_cui equal"
"""Recorded verbatim in the manifest so the derived reports quote the decision
rule as run, not as later remembered."""

_MATCH_DECIMALS = 6
"""Rounding convention shared with S6's stored similarities -- display
stability only; every decision is made on the raw float before rounding."""


@dataclass(frozen=True, slots=True)
class ConceptEvalItem:
    """One eval item's concept fingerprint, as the contam join consumes it.

    ``cuis`` is the eval QUESTION's concept set; ``answer_cui`` the item's gold
    answer concept. Injected by tests and pilot hand-runs today, built against
    real terminologies by the S4-style index when S7 lands -- the stage
    parameter stays the same either way.
    """

    benchmark: str
    cuis: frozenset[str]
    answer_cui: str | None = None


def _coerce_eval_item(
    record: ConceptEvalItem | tuple[str, Iterable[str], str | None],
) -> ConceptEvalItem:
    """Accept the dataclass or its plain ``(...)`` record form -- one call site."""
    if isinstance(record, ConceptEvalItem):
        return record
    benchmark, cuis, answer_cui = record
    return ConceptEvalItem(benchmark=benchmark, cuis=frozenset(cuis), answer_cui=answer_cui)


# --------------------------------------------------------------------------
# The two joins -- pure functions over row sequences, unit-testable without
# parquet, deterministic in the row multiset (never in the part layout).
# --------------------------------------------------------------------------


def concept_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Set Jaccard over concept ids; 0.0 when either side is empty.

    The empty-side short-circuit is load-bearing, not a convenience: 0/0 must
    never read as a match, or two equally-unlinked rows would contaminate each
    other. Rows with concepts always reach the intersection test.
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if not inter:
        return 0.0
    return inter / len(sa | sb)


def dup_key(item: CorpusItem) -> tuple[tuple[str, ...], str | None] | None:
    """Concept identity for the dup join; ``None`` when the row carries none.

    The question CUIs enter as a SORTED SET: order and multiplicity are linker
    artefacts, not meaning. ``None`` for rows with no linked question concepts
    -- an empty set identifies nothing, and grouping all-unlinked rows together
    would flag the unlinked corpus rather than deduplicate it.
    """
    if not item.cuis:
        return None
    return (tuple(sorted(set(item.cuis))), item.answer_cui)


def dup_pass(items: Sequence[CorpusItem]) -> dict[str, str]:
    """Group rows by exact concept identity -> ``{loser id: canonical id}``.

    Canonical = lowest id in the group, a pure function of the row multiset:
    the grouping reads a dict keyed by content, so neither the snapshot's part
    layout nor the input order can decide the winner.
    """
    groups: dict[tuple[tuple[str, ...], str | None], list[str]] = defaultdict(list)
    for it in items:
        key = dup_key(it)
        if key is not None:
            groups[key].append(it.id)

    dup_of: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = min(members)
        for mid in members:
            if mid != canonical:
                dup_of[mid] = canonical
    return dup_of


def contam_pass(
    items: Sequence[CorpusItem],
    eval_items: Sequence[ConceptEvalItem],
) -> tuple[dict[str, tuple[str, float]], dict[str, int]]:
    """Eval join -> ``({row id: (benchmark, best jaccard)}, {benchmark: rows})``.

    A hit requires BOTH the question-Jaccard bound and the answer-concept
    equality (see the module docstring for why either alone over-flags).
    Records with no concepts or no answer concept can never produce a
    qualifying match and are not indexed.

    The inverted index (CUI -> eval records) bounds the work to rows sharing at
    least one concept with the suite; Jaccard runs only for those candidates.
    ``per_benchmark_rows`` counts a benchmark once per row it matched (S4's
    convention), so its sum may exceed the flagged-row count when one row
    matches several benchmarks -- each catch rate stays readable per suite.
    """
    records = [
        (i, rec) for i, rec in enumerate(eval_items) if rec.cuis and rec.answer_cui is not None
    ]
    index: dict[str, list[int]] = defaultdict(list)
    for i, rec in records:
        for cui in rec.cuis:
            index[cui].append(i)

    threshold = THRESHOLDS.concept_jaccard
    hits: dict[str, tuple[str, float]] = {}
    per_benchmark_rows: Counter[str] = Counter()
    for it in items:
        if not it.cuis or it.answer_cui is None:
            continue
        candidates: set[int] = set()
        for cui in set(it.cuis):
            candidates.update(index.get(cui, ()))
        if not candidates:
            continue
        cui_set = set(it.cuis)
        best: tuple[float, str] | None = None
        matched: set[str] = set()
        for i in candidates:
            rec = eval_items[i]
            if rec.answer_cui != it.answer_cui:
                continue
            j = concept_jaccard(cui_set, rec.cuis)
            if j < threshold:
                continue
            matched.add(rec.benchmark)
            # Strict improvement only: strongest Jaccard wins, ties break to the
            # lexicographically smallest benchmark -- order-independent, so
            # contam_benchmark is reproducible across runs and processes.
            if best is None or j > best[0] or (j == best[0] and rec.benchmark < best[1]):
                best = (j, rec.benchmark)
        if best is None:
            continue
        assert matched, "a hit implies at least one qualifying benchmark"
        hits[it.id] = (best[1], round(best[0], _MATCH_DECIMALS))
        for bench in matched:
            per_benchmark_rows[bench] += 1
    return hits, dict(per_benchmark_rows)


# --------------------------------------------------------------------------
# Reporting + stage entry
# --------------------------------------------------------------------------


def flag_rates(items: Sequence[CorpusItem], flags: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-source set-rate per flag over the snapshot, plus the ``_all`` rate.

    Mirrors ``dedup_lex.flag_rates`` (stage modules do not import each other):
    every source in the snapshot gets a row, 0.0 included, so derived reports
    render per-source tables without special-casing absent sources. Rates count
    flags CARRIED in the output, so an idempotent re-run reports the same rates.
    """
    total_by_source: dict[str, int] = defaultdict(int)
    set_by_source: dict[str, dict[str, int]] = {f: defaultdict(int) for f in flags}
    for it in items:
        total_by_source[it.source] += 1
        dumped = it.flags.model_dump()
        for f in flags:
            if dumped[f]:
                set_by_source[f][it.source] += 1

    out: dict[str, dict[str, float]] = {}
    for f in flags:
        rates = {
            src: set_by_source[f].get(src, 0) / total for src, total in total_by_source.items()
        }
        rates["_all"] = sum(set_by_source[f].values()) / len(items) if items else 0.0
        out[f] = rates
    return out


def stage_entry(
    run_id: str,
    *,
    eval_items: Sequence[ConceptEvalItem | tuple[str, Iterable[str], str | None]] | None = None,
    cui_namespace: str = "cui",
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> StageManifest:
    """S8: concept dedup + cross-lingual decontamination -> the ``08_concept`` snapshot.

    ``eval_items`` is the explicit contract for the eval side: inject synthetic
    fingerprints in tests and pilot runs; production passes the terminology
    index once S7 makes it real. Without it the contam half is SKIPPED and the
    manifest says so -- the dedup half still runs, because concept dedup needs
    no eval data. Optional dirs exist for tests (tmp_path instead of /scratch);
    production calls take only ``run_id``.
    """
    started = utcnow()
    inp = input_dir if input_dir is not None else store.stage_dir(run_id, INPUT_STAGE)
    out = output_dir if output_dir is not None else store.stage_dir(run_id, STAGE)
    if not inp.is_dir():
        raise StageError(f"S8: input snapshot {inp} does not exist -- run {INPUT_STAGE} first")
    out.mkdir(parents=True, exist_ok=True)  # explicit dirs skip stage_dir's mkdir
    # Resume contract (runner docstring): a re-run replaces its own snapshot only.
    # write_items APPENDS part files, so a previous run's parts must go first or
    # the re-run would double every row.
    for stale in out.glob("part-*.parquet"):
        stale.unlink()

    records = [_coerce_eval_item(r) for r in eval_items] if eval_items is not None else []

    # Sort by id: both passes must be a pure function of the row multiset.
    items = sorted(store.iter_items(inp), key=lambda it: it.id)
    if not items:
        raise StageError(f"S8: input snapshot {inp} is empty -- run {INPUT_STAGE} first")

    dup_of = dup_pass(items)
    if records:
        hits, per_benchmark_rows = contam_pass(items, records)
        contam_status = "ok"
    else:
        hits, per_benchmark_rows = {}, {}
        contam_status = "skipped"

    out_items: list[CorpusItem] = []
    for it in items:
        flags = it.flags
        updates: dict[str, Any] = {}
        if it.id in dup_of and not flags.f_dup_concept:
            # A previous S8 run already decided this row: never re-point dup_of.
            updates["dup_of"] = dup_of[it.id]
            flags = flags.model_copy(update={"f_dup_concept": True})
        if it.id in hits:
            benchmark, _jaccard = hits[it.id]
            if not flags.f_contam_concept:
                flags = flags.model_copy(update={"f_contam_concept": True})
            if it.contam_benchmark is None:
                # First attribution wins across stages (S6's convention): an
                # earlier n-gram/semantic hit keeps its benchmark name.
                updates["contam_benchmark"] = benchmark
        if flags is not it.flags:  # model_copy only ran when a flag actually flipped
            updates["flags"] = flags
        out_items.append(it.model_copy(update=updates) if updates else it)

    store.write_items(out_items, out)

    manifest = StageManifest(
        run_id=run_id,
        stage=STAGE,
        started_at=started,
        config={
            "input_dir": str(inp),
            "output_dir": str(out),
            "cui_namespace": cui_namespace,
            "n_eval_items": len(records),
            "dup_identity": DUP_IDENTITY,
            "contam_rule": CONTAM_RULE,
        },
        thresholds=snapshot(),
    )
    manifest.rows_in = store.count_items(inp)
    manifest.rows_out = store.count_items(out)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.flag_rates = flag_rates(out_items, STAGE_FLAGS)
    notes: dict[str, Any] = {
        "concept_dup_groups": len(set(dup_of.values())),
        "concept_dup_rows": sum(1 for it in out_items if it.flags.f_dup_concept),
        "contam_status": contam_status,
        "contam_rows": sum(1 for it in out_items if it.flags.f_contam_concept),
        "per_benchmark_hits": dict(sorted(per_benchmark_rows.items())),
        "rows_without_cuis": sum(1 for it in items if not it.cuis),
    }
    if contam_status == "skipped":
        notes["contam_skip_reason"] = (
            "no eval concept records injected -- pass eval_items "
            "(the S4-style terminology index arrives with S7)"
        )
    manifest.notes = notes
    assert manifest.rows_in == manifest.rows_out, "flags-not-deletes: S8 never drops"
    return manifest


__all__ = [
    "CONTAM_RULE",
    "DUP_IDENTITY",
    "INPUT_STAGE",
    "STAGE",
    "STAGE_FLAGS",
    "ConceptEvalItem",
    "concept_jaccard",
    "contam_pass",
    "dup_key",
    "dup_pass",
    "flag_rates",
    "stage_entry",
]
