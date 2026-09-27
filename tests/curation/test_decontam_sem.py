"""S6 semantic decontam tests: brute-force self-join, the vectorized benchmark
join's equivalence with the per-pair definition, and the O(n^2) fallback gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError
from medrl.curation.stages import decontam_sem
from medrl.curation.stages.embed import bytes_to_vec, fp16_bytes
from medrl.curation.thresholds import THRESHOLDS


def _item(item_id: str, vec: np.ndarray) -> CorpusItem:
    return CorpusItem(
        id=item_id,
        source="s1",
        messages=[{"role": "user", "content": f"question {item_id}"}],
        answer_type="none",
        embedding=fp16_bytes(vec),
    )


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(store, "stage_dir", lambda run_id, name: tmp_path / name)
    return tmp_path


def test_self_join_flags_near_duplicates_canonical_lower_id(scratch: Path) -> None:
    inp = scratch / "05_embed"
    inp.mkdir(parents=True)
    base = _vec(1)
    items = [
        _item("s:a", base),
        _item("s:b", base + 0.001 * _vec(2)),  # cos ~1 -> duplicate of the lower id
        _item("s:c", _vec(3)),  # orthogonal -> clean
    ]
    store.write_items(items, inp)

    manifest = decontam_sem.run_decontam_sem("run-s", bench_vectors_path=None)

    rows = {it.id: it for it in store.iter_items(scratch / "06_decontam_sem")}
    assert rows["s:b"].flags.f_dup_semantic
    assert rows["s:b"].dup_of == "s:a", "the lower id is canonical"
    assert not rows["s:a"].flags.f_dup_semantic
    assert not rows["s:c"].flags.f_dup_semantic
    assert manifest.notes["semantic_dup_pairs"] == 1


def test_benchmark_join_matches_the_per_pair_definition(scratch: Path) -> None:
    """The gemv rewrite must select exactly the pair loop's answer: max cosine
    over all benchmark vectors, ties to the first benchmark in blob order."""
    corpus = [_item(f"s:{i}", _vec(100 + i)) for i in range(5)]
    contaminated = _item("s:leak", _vec(7))
    corpus.append(contaminated)
    inp = scratch / "05_embed"
    inp.mkdir(parents=True)
    store.write_items(corpus, inp)

    bench_vectors = {
        "medqa": [{"id": "m:1", "key": "m1"}],
        "medmcqa": [{"id": "mc:1", "key": "mc1"}, {"id": "mc:2", "key": "mc2"}],
    }
    vectors = {
        "m1": _vec(7),  # identical direction to s:leak -> cos 1.0
        "mc1": _vec(50),
        "mc2": _vec(7) + 0.01 * _vec(51),  # also near: still below the m1 hit
    }
    blob_path = scratch / "bench.json"
    blob_path.write_text(json.dumps(bench_vectors))
    np.savez(
        scratch / "bench.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()}
    )

    manifest = decontam_sem.run_decontam_sem("run-s", bench_vectors_path=str(blob_path))

    # the reference: the old per-pair loop, verbatim
    best: tuple[str, float] | None = None
    v = bytes_to_vec(contaminated.embedding)
    for benchmark, entries in bench_vectors.items():
        for e in entries:
            bv = np.asarray(vectors[e["key"]], dtype=np.float32)
            denom = float(np.linalg.norm(v) * np.linalg.norm(bv))
            c = float(v @ bv / denom) if denom else 0.0
            if c >= THRESHOLDS.semantic_contam_threshold and (best is None or c > best[1]):
                best = (benchmark, c)

    rows = {it.id: it for it in store.iter_items(scratch / "06_decontam_sem")}
    assert best is not None and best[0] == "medqa"
    assert rows["s:leak"].contam_benchmark == "medqa"
    assert rows["s:leak"].flags.f_contam_semantic
    assert manifest.notes["per_benchmark_hits"] == {"medqa": 1}
    for clean_id in ("s:0", "s:1", "s:2", "s:3", "s:4"):
        assert rows[clean_id].contam_benchmark is None


def test_missing_ann_index_is_refused_at_scale(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ann.usearch above the brute-force ceiling is a StageError, not
    a silent weeks-long O(n^2) full-dim run."""
    monkeypatch.setattr(decontam_sem, "_BRUTE_FORCE_MAX_ITEMS", 3)
    inp = scratch / "05_embed"
    inp.mkdir(parents=True)
    store.write_items([_item(f"s:{i}", _vec(i)) for i in range(5)], inp)

    with pytest.raises(StageError, match=r"ann\.usearch"):
        decontam_sem.run_decontam_sem("run-s", bench_vectors_path=None)
