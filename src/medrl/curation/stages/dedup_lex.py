"""S3 -- lexical deduplication: exact, word n-gram and MinHash-LSH passes.

Flags-not-deletes: a duplicate row stays in the corpus with ``f_dup_exact`` or
``f_dup_minhash`` set and ``dup_of`` pointing at the canonical row chosen by the
keep-rule (permissive licence -> deeper reasoning trace -> lowest id, the plan's
lane-ranked rule). Downstream mixtures exclude flagged rows by query, so every
dedup decision stays auditable and reversible at the SQL level.

Relationship to ``medrl.data.dedup`` (wrap, do not reimplement):

* the exact pass hashes with the data-side convention verbatim -- ``_normalize_text``
  then sha256, exactly what ``exact_dedup`` hashes. The grouping differs only
  because flags-not-deletes needs the loser->winner mapping ``exact_dedup`` never
  returns.
* the n-gram pass keeps ``ngram_dedup``'s algorithm (inverted shingle index,
  Jaccard verify) but lifts shingling from char-level to word-level 13-grams --
  the plan section 3.2 reconciliation: word-level matches the eval-side convention.
* the MinHash pass builds signatures and LSH with datasketch directly because the
  data-side ``minhash_dedup`` hardcodes unigram hashing; shingles here are word
  5-grams. LSH candidates are verified with exact Jaccard over the shingle sets
  (``ngram_overlap``) -- banding is probabilistic and must never be the judge.

The dedup identity is the role-tagged transcript plus the gold ``answer`` field,
NOT the thinking trace: lane 2 of the keep-rule keeps the longer trace among
duplicates, which only means something if rows differing solely in thinking land
in the same group. Rows whose transcript normalizes to empty can never duplicate
(the data-side skips them too); genuinely empty rows are S2's ``f_empty`` problem.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow
from medrl.curation.thresholds import THRESHOLDS, snapshot
from medrl.data.dedup import _normalize_text, ngram_overlap

STAGE = "03_dedup"
INPUT_STAGE = "02_structural"

#: Flags this stage owns; ``flag_rates`` reports exactly these, nothing downstream.
STAGE_FLAGS: tuple[str, ...] = ("f_dup_exact", "f_dup_minhash")

RULE_LICENCE = "licence"
RULE_THINKING = "thinking"
RULE_SOURCE_ID = "source_id"

# sha256 hexdigest truncation -- the data-side ngram_hashes shingle convention.
_DIGEST_CHARS = 16
_PERMISSIVE_LICENCE_PREFIXES = ("apache", "mit")
_CC_LICENCE_PREFIX = "cc"
_UNKNOWN_LICENCES: frozenset[str] = frozenset({"", "unknown"})


# --------------------------------------------------------------------------
# Canonical keep-rule (plan section 4, S3)
# --------------------------------------------------------------------------


def licence_tier(licence: str) -> int:
    """Lane-1 bucket: apache/mit (0) > cc-* (1) > unknown (2) > everything else (3).

    Deliberately a total function: an unparsable licence string lands in the
    proprietary bucket rather than crashing the pass -- licence honesty is S0's
    job, this is only a keep-rule ordering.
    """
    norm = licence.strip().lower().replace(" ", "").replace("_", "-")
    if norm.startswith(_PERMISSIVE_LICENCE_PREFIXES):
        return 0
    if norm.startswith(_CC_LICENCE_PREFIX):
        return 1
    if norm in _UNKNOWN_LICENCES:
        return 2
    return 3


def _trace_depth(item: CorpusItem) -> int:
    """Lane-2 depth: the thinking trace when present, else the assistant payload.

    Read literally from the plan: an item's depth is its trace, or its answer
    when it has none -- so a trace-bearing row and a bare row compare those two
    measures against each other rather than treating None as zero length.
    """
    if item.thinking is not None:
        return len(item.thinking)
    return sum(len(m["content"]) for m in item.messages if m["role"] == "assistant")


def _keep_key(item: CorpusItem) -> tuple[int, int, str]:
    """Sort key encoding the lane ranking: permissive first, deeper first, id first."""
    return (licence_tier(item.licence), -_trace_depth(item), item.id)


def apply_keep_rule(group: Sequence[CorpusItem]) -> tuple[CorpusItem, str | None]:
    """Canonical row of a duplicate group plus the lane that decided it.

    Lanes are ranked: permissive licence, then deeper reasoning trace (or longer
    assistant content for rows without a trace), then lowest row id as the total
    order. The reported lane is the FIRST one that separates the winner from
    every loser; a singleton wins with lane ``None`` because nothing was decided.
    """
    ranked = sorted(group, key=_keep_key)
    winner = ranked[0]
    if len(ranked) == 1:
        return winner, None
    rest = ranked[1:]
    if licence_tier(winner.licence) < min(licence_tier(r.licence) for r in rest):
        return winner, RULE_LICENCE
    if _trace_depth(winner) > max(_trace_depth(r) for r in rest):
        return winner, RULE_THINKING
    return winner, RULE_SOURCE_ID


# --------------------------------------------------------------------------
# Text identity: exact key and word shingles
# --------------------------------------------------------------------------


def dedup_text(item: CorpusItem) -> str:
    """Transcript identity for both passes (thinking deliberately excluded).

    The gold ``answer`` field is part of identity: two rows sharing a question
    but carrying contradictory gold answers are different training signal, not
    duplicates, and must not be merged.
    """
    parts = [f"{m['role']}\x00{m['content']}" for m in item.messages]
    if item.answer is not None:
        parts.append(f"answer\x00{item.answer}")
    return "\n".join(parts)


def exact_key(item: CorpusItem) -> str | None:
    """sha256 over the data-side normalization (``exact_dedup``'s convention).

    ``None`` for an empty transcript: such a row can never duplicate (the
    data-side skips empties too), and S2 already marks genuine empties.
    """
    normalized = _normalize_text(dedup_text(item))
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_shingles(text: str, n: int) -> set[str]:
    """Word-level n-gram sha256 digests -- ``ngram_hashes`` lifted from chars to words.

    Below ``n`` words the whole text is one shingle (the data-side convention for
    short text), which keeps tiny rows out of the near-dup passes: two short rows
    only share a shingle if they normalize equal, and the exact pass claimed
    those already.
    """
    words = _normalize_text(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()[:_DIGEST_CHARS]}
    return {
        hashlib.sha256(" ".join(words[i : i + n]).encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
        for i in range(len(words) - n + 1)
    }


# --------------------------------------------------------------------------
# Passes -- each returns (loser id -> canonical id, keep-rule lane -> group count)
# --------------------------------------------------------------------------


def _groups_to_claims(
    groups: Mapping[str, Sequence[str]],
    by_id: Mapping[str, CorpusItem],
) -> tuple[dict[str, str], dict[str, int]]:
    """Leader-starred groups -> loser->winner claims + per-lane group counts."""
    dup_of: dict[str, str] = {}
    lanes: dict[str, int] = defaultdict(int)
    for members in groups.values():
        if len(members) < 2:
            continue
        winner, lane = apply_keep_rule([by_id[mid] for mid in members])
        assert lane is not None, "a multi-member group always decides on some lane"
        lanes[lane] += 1
        for mid in members:
            if mid != winner.id:
                dup_of[mid] = winner.id
    return dup_of, dict(lanes)


def exact_pass(items: Sequence[CorpusItem]) -> tuple[dict[str, str], dict[str, int]]:
    """Group rows by exact content key; every loser points at its group's canonical row."""
    by_key: dict[str, list[CorpusItem]] = defaultdict(list)
    for it in items:
        key = exact_key(it)
        if key is not None:
            by_key[key].append(it)

    dup_of: dict[str, str] = {}
    lanes: dict[str, int] = defaultdict(int)
    for group in by_key.values():
        if len(group) < 2:
            continue
        winner, lane = apply_keep_rule(group)
        assert lane is not None, "a multi-member group always decides on some lane"
        lanes[lane] += 1
        for member in group:
            if member.id != winner.id:
                dup_of[member.id] = winner.id
    return dup_of, dict(lanes)


def ngram_pass(items: Sequence[CorpusItem]) -> tuple[dict[str, str], dict[str, int]]:
    """Deterministic word 13-gram Jaccard pass (``ngram_dedup`` at word level).

    Greedy leader grouping in id order, matching the data-side first-wins
    semantics: a row joins the lowest-id already-kept leader it verifies against,
    otherwise it becomes a leader. The keep-rule re-decides the winner per group
    afterwards, so insertion order can never pick the canonical row.
    """
    shingles = {it.id: word_shingles(dedup_text(it), THRESHOLDS.ngram_dup_n) for it in items}
    by_id = {it.id: it for it in items}
    index: dict[str, list[str]] = defaultdict(list)  # shingle -> leader ids
    groups: dict[str, list[str]] = {}

    for it in sorted(items, key=lambda x: x.id):
        sh = shingles[it.id]
        if not sh:
            continue
        candidates: set[str] = set()
        for s in sh:
            candidates.update(index.get(s, ()))
        verified = [
            c for c in candidates if ngram_overlap(sh, shingles[c]) >= THRESHOLDS.ngram_dup_jaccard
        ]
        if verified:
            leader = min(verified)
            groups[leader].append(it.id)
        else:
            groups[it.id] = [it.id]
            for s in sh:
                index[s].append(it.id)
    return _groups_to_claims(groups, by_id)


def minhash_pass(items: Sequence[CorpusItem]) -> tuple[dict[str, str], dict[str, int]]:
    """MinHash-LSH over word 5-gram shingles, exact-Jaccard verified.

    Datasketch is driven directly (the data-side ``minhash_dedup`` shingles
    unigrams); the signature itself is order-independent, but shingles are
    inserted sorted anyway so the byte stream a debugger sees is stable.
    """
    from datasketch import MinHash, MinHashLSH  # data extra; lazy like data/dedup.py

    num_perm = THRESHOLDS.minhash_num_perm
    shingles = {it.id: word_shingles(dedup_text(it), THRESHOLDS.minhash_shingle_n) for it in items}
    by_id = {it.id: it for it in items}
    lsh = MinHashLSH(threshold=THRESHOLDS.minhash_jaccard, num_perm=num_perm)
    groups: dict[str, list[str]] = {}

    for it in sorted(items, key=lambda x: x.id):
        sh = shingles[it.id]
        if not sh:
            continue
        mh = MinHash(num_perm=num_perm)
        for s in sorted(sh):
            mh.update(s.encode("utf-8"))
        # Exact-Jaccard verify over LSH candidates (plan 3.2): banding nominates,
        # the shingle sets judge -- LSH luck can neither confirm nor refute a pair.
        candidates = lsh.query(mh)
        verified = [
            c for c in candidates if ngram_overlap(sh, shingles[c]) >= THRESHOLDS.minhash_jaccard
        ]
        if verified:
            leader = min(verified)
            groups[leader].append(it.id)
        else:
            lsh.insert(it.id, mh)
            groups[it.id] = [it.id]
    return _groups_to_claims(groups, by_id)


# --------------------------------------------------------------------------
# Reporting + stage entry
# --------------------------------------------------------------------------


def flag_rates(items: Sequence[CorpusItem], flags: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-source set-rate per flag over the final snapshot, plus the ``_all`` rate.

    Every source in the snapshot gets a row (0.0 included) so the derived report
    can render per-source tables without special-casing absent sources.
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


def _rule_totals(*parts: Mapping[str, int]) -> dict[str, int]:
    """Sum per-lane counts across passes, keeping all three lanes present at zero."""
    merged: dict[str, int] = {RULE_LICENCE: 0, RULE_THINKING: 0, RULE_SOURCE_ID: 0}
    for part in parts:
        for lane, n in part.items():
            merged[lane] = merged.get(lane, 0) + n
    return merged


def stage_entry(
    run_id: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> StageManifest:
    """Runner adapter for S3: read the S2 snapshot, write the flagged ``03_dedup`` one.

    Exact pass first; the two near-dup passes run only over its survivors, so a
    row carries at most one dup flag and the passes' costs shrink cascade-style.
    Optional dirs exist for tests (tmp_path instead of /scratch); production calls
    take only ``run_id``.
    """
    started = utcnow()
    inp = input_dir or store.stage_dir(run_id, INPUT_STAGE)
    out = output_dir or store.stage_dir(run_id, STAGE)
    out.mkdir(parents=True, exist_ok=True)
    # Resume contract (runner docstring): a re-run replaces its own snapshot only.
    for stale in out.glob("part-*.parquet"):
        stale.unlink()

    manifest = StageManifest(
        run_id=run_id,
        stage=STAGE,
        started_at=started,
        config={
            "input_dir": str(inp),
            "output_dir": str(out),
            "passes": ["exact", "ngram", "minhash"],
        },
        thresholds=snapshot(),
    )

    # Sort by id so group formation is a pure function of the row multiset, not
    # of the input snapshot's parquet part layout.
    items = sorted(store.iter_items(inp), key=lambda it: it.id)
    if not items:
        raise StageError(f"S3: input snapshot {inp} is empty -- run S2 first")

    exact_dup, exact_lanes = exact_pass(items)
    survivors = [it for it in items if it.id not in exact_dup]
    ngram_dup, ngram_lanes = ngram_pass(survivors)
    survivors = [it for it in survivors if it.id not in ngram_dup]
    minhash_dup, minhash_lanes = minhash_pass(survivors)

    # (flag field, canonical id) per loser; the near-dup passes share
    # f_dup_minhash -- there is no separate n-gram flag in the schema.
    claims: dict[str, tuple[str, str]] = {}
    for loser, winner in exact_dup.items():
        claims[loser] = ("f_dup_exact", winner)
    for loser, winner in {**ngram_dup, **minhash_dup}.items():
        claims[loser] = ("f_dup_minhash", winner)

    out_items: list[CorpusItem] = []
    for it in items:
        claim = claims.get(it.id)
        if claim is None:
            out_items.append(it)
            continue
        flag_name, winner = claim
        out_items.append(
            it.model_copy(
                update={
                    "flags": it.flags.model_copy(update={flag_name: True}),
                    "dup_of": winner,
                }
            )
        )

    store.write_items(out_items, out)

    manifest.rows_in = store.count_items(inp)
    manifest.rows_out = store.count_items(out)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.flag_rates = flag_rates(out_items, STAGE_FLAGS)
    manifest.notes = {
        "exact_groups": sum(exact_lanes.values()),
        "ngram_groups": sum(ngram_lanes.values()),
        "minhash_groups": sum(minhash_lanes.values()),
        "kept_by_rule": _rule_totals(exact_lanes, ngram_lanes, minhash_lanes),
    }
    assert manifest.rows_in == manifest.rows_out, "flags-not-deletes: S3 never drops"
    return manifest


__all__ = [
    "INPUT_STAGE",
    "RULE_LICENCE",
    "RULE_SOURCE_ID",
    "RULE_THINKING",
    "STAGE",
    "STAGE_FLAGS",
    "apply_keep_rule",
    "dedup_text",
    "exact_key",
    "exact_pass",
    "flag_rates",
    "licence_tier",
    "minhash_pass",
    "ngram_pass",
    "stage_entry",
    "word_shingles",
]
