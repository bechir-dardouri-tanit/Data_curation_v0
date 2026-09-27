"""Data-side decontam tests (medrl.data.decontam.check_contamination): the
per-document query hoist, per-benchmark partitioning, and the embedding-path
de-duplication."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from medrl.data.decontam import (
    BenchmarkIndex,
    BenchmarkItem,
    TrainDocument,
    check_contamination,
)

_LONG_A = (
    "A 67-year-old man presents with crushing substernal chest pain radiating to his left arm "
    "and jaw, associated with diaphoresis and shortness of breath. ECG shows ST elevation. "
    "What is the most likely diagnosis?"
)
_LONG_B = (
    "A 54-year-old woman with rheumatoid arthritis develops a swollen, painful right calf. "
    "What is the most appropriate next diagnostic step for this patient's presentation?"
)
_UNRELATED = (
    "The mitochondrial electron transport chain couples electron transfer to proton pumping "
    "across the inner membrane, and oxidative phosphorylation synthesizes ATP accordingly."
)


def _index() -> BenchmarkIndex:
    idx = BenchmarkIndex(ngram_n=13)
    idx.add(BenchmarkItem(benchmark="medqa", item_id="m1", text=_LONG_A))
    idx.add(BenchmarkItem(benchmark="medmcqa", item_id="mc1", text=_LONG_B))
    return idx


def _doc(doc_id: str, text: str) -> TrainDocument:
    return TrainDocument(id=doc_id, text=text, source="train")


def test_ngram_hits_partition_per_benchmark_with_one_query_per_doc() -> None:
    """Regression: the loop re-queried once per benchmark (12x the hot path at
    12 benchmarks); one unfiltered query per doc, partitioned by the hit's
    benchmark, must give the identical hit set."""
    idx = _index()
    docs = [_doc("d1", _LONG_A), _doc("d2", _LONG_B), _doc("d3", _UNRELATED)]

    calls: list[str] = []
    real_query = BenchmarkIndex.query_ngram

    def spy(self: BenchmarkIndex, text: str, threshold: float, benchmark: Any = None):
        calls.append(benchmark if benchmark is not None else "<unfiltered>")
        return real_query(self, text, threshold, benchmark)

    pytest.MonkeyPatch().setattr(BenchmarkIndex, "query_ngram", spy)
    result = check_contamination(docs, idx, ngram_threshold=0.8)

    assert calls == ["<unfiltered>"] * 3, "exactly one query per document"
    assert result.contaminated_docs == 2
    assert result.clean_docs == {"d3"}
    per_bench = {(h.benchmark, h.benchmark_item_id) for h in result.hits}
    assert per_bench == {("medqa", "m1"), ("medmcqa", "mc1")}
    assert {("medqa", "m1")} <= {
        (h.benchmark, h.benchmark_item_id) for h in result.reports["medqa"].hits
    }
    assert result.reports["medqa"].contaminated_items == 1


def test_ngram_partition_equivalent_to_old_per_benchmark_loop() -> None:
    idx = _index()
    docs = [_doc("d1", _LONG_A), _doc("d2", _LONG_B)]

    # the old algorithm's answer, verbatim: one filtered query per benchmark
    old_hits: set[tuple[str, str, float]] = set()
    for doc in docs:
        for bench in idx.get_benchmark_names():
            for key, overlap in idx.query_ngram(doc.text, threshold=0.8, benchmark=bench):
                old_hits.add((doc.id, idx.items[key].item_id, round(overlap, 9)))

    result = check_contamination(docs, idx, ngram_threshold=0.8)
    new_hits = {(h.train_id, h.benchmark_item_id, round(h.ngram_overlap, 9)) for h in result.hits}
    assert new_hits == old_hits


def test_embedding_hits_skip_ngram_pairs_and_query_once_per_doc(monkeypatch) -> None:
    idx = _index()
    idx.embeddings = np.zeros((2, 2))  # enables the embedding branch
    docs = [_doc("d1", _LONG_A)]

    embed_calls: list[str] = []

    def fake_query_embedding(self, text: str, threshold: float, benchmark: Any = None):
        embed_calls.append(benchmark if benchmark is not None else "<unfiltered>")
        # an overlap with the ngram-hit item (must be skipped) + a new one
        return [("medqa::m1", 0.99), ("medmcqa::mc1", 0.95)]

    monkeypatch.setattr(BenchmarkIndex, "query_embedding", fake_query_embedding)

    result = check_contamination(docs, idx, check_embeddings=True, embedding_threshold=0.9)

    assert embed_calls == ["<unfiltered>"], "one embedding query per doc, not per (doc, benchmark)"
    pairs = [(h.benchmark, h.benchmark_item_id) for h in result.hits]
    # the ngram hit stays exactly once; the embedding duplicate of it is skipped
    assert pairs.count(("medqa", "m1")) == 1
    assert ("medmcqa", "mc1") in pairs
