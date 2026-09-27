"""S6 -- semantic dedup + semantic decontamination on the S5 vectors.

Two joins, both full-resolution cosine on fp16-restored vectors:

* self-join: near-duplicate corpus items (cos >= semantic_dup_threshold) get
  ``f_dup_semantic`` + ``dup_of`` (deterministic: the lower id is canonical --
  the licence-aware S3 rule governs exact/near-exact; here both rows are
  already survivors, so id order is the reproducible tie-break);
* benchmark-join: corpus questions landing near an eval question
  (cos >= semantic_contam_threshold) get ``f_contam_semantic`` +
  ``contam_benchmark``.

⚠️ Both thresholds are bge-m3-era values (THRESHOLDS marks them PILOT): they
MUST be recalibrated for Qwen3-Embedding on the B7 pilot before full scale --
the plan's warning, enforced here by the PILOT marker.

ANN recall runs on MRL-truncated dims; every flagged pair is re-scored at full
dims before a flag is set, so truncation can only cost recall of candidates,
never precision of flags. Small corpora take a brute-force path (exact, and
the unit-test path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow
from medrl.curation.stages.embed import bytes_to_vec
from medrl.curation.thresholds import THRESHOLDS

_BRUTE_FORCE_MAX_ITEMS = 50_000
"""Self-join brute-force ceiling: exact O(n^2) is the small-corpus/unit-test
path only. Above this a missing ann.usearch index is a StageError -- the
fallback would be a weeks-long full-dim O(n^2) run at 2.5M rows, and an S5
crash between part-write and index-build must not silently trigger it."""


@dataclass(slots=True, frozen=True)
class PairDecision:
    other_id: str
    cosine: float
    is_dup: bool


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


def decide_self(
    vec: np.ndarray, other_id: str, other_vec: np.ndarray, threshold: float
) -> PairDecision:
    """Full-dim decision for one (item, candidate) pair -- pure, unit-testable."""
    c = cosine(vec, other_vec)
    return PairDecision(other_id=other_id, cosine=round(c, 6), is_dup=c >= threshold)


def decide_contam(
    vec: np.ndarray, benchmark: str, bench_vec: np.ndarray, threshold: float
) -> PairDecision:
    c = cosine(vec, bench_vec)
    return PairDecision(other_id=benchmark, cosine=round(c, 6), is_dup=c >= threshold)


def _self_join_bruteforce(
    blobs: dict[str, bytes], threshold: float
) -> dict[str, tuple[str, float]]:
    """Exact O(n^2) self-join for small n; canonical = lower id in each pair."""
    vecs = [(iid, bytes_to_vec(blob)) for iid, blob in blobs.items()]
    vecs.sort(key=lambda t: t[0])  # lower id first -> seen set wins canonical role
    dup_of: dict[str, tuple[str, float]] = {}
    for i in range(len(vecs)):
        id_i, v_i = vecs[i]
        if id_i in dup_of:
            continue
        for j in range(i + 1, len(vecs)):
            id_j, v_j = vecs[j]
            if id_j in dup_of:
                continue
            c = cosine(v_i, v_j)
            if c >= threshold:
                dup_of[id_j] = (id_i, round(c, 6))
    return dup_of


def _self_join_ann(
    snapshot_dir: str, threshold: float
) -> dict[str, tuple[str, float]]:
    """usearch recall at truncated dims -> full-dim verification -> flags."""
    # Only the fp16 blobs are held in RAM (not the full CorpusItem -- message
    # text is the bulk of an item at 2.5M rows); candidates re-read vectors
    # from this map instead of materializing the whole snapshot as objects.
    blobs: dict[str, bytes] = {}
    for it in store.iter_items(Path(snapshot_dir)):
        if it.embedding is not None:
            blobs[it.id] = it.embedding
    ids = sorted(blobs)
    index_path = Path(snapshot_dir) / "ann.usearch"
    if not index_path.exists():
        if len(ids) > _BRUTE_FORCE_MAX_ITEMS:
            raise StageError(
                f"S6: {len(ids)} embedded items but no ann.usearch index under "
                f"{snapshot_dir} -- the exact O(n^2) fallback is refused at this "
                "scale; re-run 05_embed to rebuild the index"
            )
        return _self_join_bruteforce(blobs, threshold)
    from usearch.index import Index

    index = Index(ndim=THRESHOLDS.embed_dim_index, metric="cos", dtype="f16")
    index.load(str(index_path))
    key_to_id = {int(k): v for k, v in json.loads(
        (index_path.parent / "ann_ids.json").read_text()).items()}
    dup_of: dict[str, tuple[str, float]] = {}
    for iid in ids:
        v_full = bytes_to_vec(blobs[iid])
        hits = index.search(v_full[: index.ndim], count=2)
        for hit in (hits if hits.shape else [hits]):
            other_id = key_to_id.get(int(hit["key"]))
            if other_id is None or other_id == iid:
                continue
            decision = decide_self(v_full, other_id, bytes_to_vec(blobs[other_id]), threshold)
            if decision.is_dup:
                keep, drop = sorted((iid, other_id))
                if drop not in dup_of or dup_of[drop][1] < decision.cosine:
                    dup_of[drop] = (keep, decision.cosine)
    return dup_of


def run_decontam_sem(
    run_id: str,
    *,
    input_stage: str = "05_embed",
    output_stage: str = "06_decontam_sem",
    bench_vectors_path: str | None = None,
) -> StageManifest:
    """Flag semantic duplicates + benchmark-contaminated items on the S5 snapshot.

    ``bench_vectors_path``: JSON {benchmark: [ids...]} + npz of vectors produced
    by embed_benchmarks(); when absent, only the self-join runs.
    """
    started = utcnow()
    inp = store.stage_dir(run_id, input_stage)
    out = store.reset_dir(store.stage_dir(run_id, output_stage))
    manifest = StageManifest(
        run_id=run_id, stage=output_stage, started_at=started,
        thresholds={"semantic_dup": THRESHOLDS.semantic_dup_threshold,
                    "semantic_contam": THRESHOLDS.semantic_contam_threshold},
        config={"bench_vectors": bench_vectors_path},
    )

    dup_threshold = THRESHOLDS.semantic_dup_threshold
    dup_of = _self_join_ann(str(inp), dup_threshold)

    # The benchmark-join stacks every benchmark vector once (row order = the
    # blob's benchmark/entry order, so ties break to the first benchmark, as
    # the old per-pair loop did) and scores each corpus vector with one BLAS
    # gemv against the normalized matrix. The previous per-pair Python cosine
    # loop was O(corpus x benchmark) interpreted pairs: ~1.35e11 cosine calls
    # (~156 core-hours) at 2.5M corpus rows x 54k benchmark questions.
    bench_matrix: np.ndarray | None = None
    bench_of_row: list[str] = []
    if bench_vectors_path:
        blob = json.loads(open(bench_vectors_path).read())
        npz = np.load(bench_vectors_path.replace(".json", ".npz"))
        rows: list[np.ndarray] = []
        for benchmark, entries in blob.items():
            for e in entries:
                rows.append(npz[e["key"]])
                bench_of_row.append(benchmark)
        if rows:
            bench_matrix = np.stack(rows).astype(np.float32)
            norms = np.linalg.norm(bench_matrix, axis=1)
            norms[norms == 0.0] = 1.0
            bench_matrix /= norms[:, None]

    per_bench_hits: dict[str, int] = {}
    n_dup = n_contam = 0
    out_buf: list[CorpusItem] = []

    def flush_buf() -> None:
        if out_buf:
            store.write_items(out_buf, out)
            out_buf.clear()

    for it in store.iter_items(inp):
        updates: dict = {}
        if it.id in dup_of and not it.flags.f_dup_semantic:
            other, c = dup_of[it.id]
            updates["dup_of"] = other
            updates["flags"] = it.flags.model_copy(update={"f_dup_semantic": True})
            n_dup += 1
        if bench_matrix is not None and it.embedding is not None and it.contam_benchmark is None:
            v = bytes_to_vec(it.embedding)
            vnorm = float(np.linalg.norm(v))
            sims = bench_matrix @ (v / vnorm) if vnorm else np.zeros(bench_matrix.shape[0])
            j = int(np.argmax(sims))  # max-cosine pair; first row wins ties
            if sims[j] >= THRESHOLDS.semantic_contam_threshold:
                best = (bench_of_row[j], float(sims[j]))
                updates["contam_benchmark"] = best[0]
                flags = updates.get("flags", it.flags)
                updates["flags"] = flags.model_copy(update={"f_contam_semantic": True})
                per_bench_hits[best[0]] = per_bench_hits.get(best[0], 0) + 1
                n_contam += 1
        out_buf.append(it.model_copy(update=updates) if updates else it)
        if len(out_buf) >= 50_000:
            flush_buf()
    flush_buf()

    manifest.rows_in = manifest.rows_out = store.count_items(inp)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.flag_rates = {"f_dup_semantic": {"_all": n_dup / max(manifest.rows_in, 1)},
                           "f_contam_semantic": {"_all": n_contam / max(manifest.rows_in, 1)}}
    manifest.notes = {"semantic_dup_pairs": n_dup, "semantic_contam_hits": n_contam,
                      "per_benchmark_hits": per_bench_hits}
    return manifest


__all__ = ["decide_contam", "decide_self", "run_decontam_sem"]
