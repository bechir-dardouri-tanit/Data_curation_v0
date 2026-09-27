"""Decontamination: Detect and remove benchmark contamination from training data.

Covers:
- 13-gram overlap analysis against all benchmarks
- Embedding-NN decontamination (cosine similarity > 0.9)
- Negative-control test generation for contamination detection
- Per-dataset contamination reports

The module ensures training data does not contain benchmark items, which would
inflated evaluation scores.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from medrl.core.logging import get_logger
from medrl.data.dedup import compute_embeddings, ngram_hashes, ngram_overlap

log = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

NGRAM_N = 13
CONTAMINATION_THRESHOLD = 0.8  # N-gram overlap threshold
EMBEDDING_THRESHOLD = 0.9  # Cosine similarity threshold
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_ENCODER_CACHE: dict[str, Any] = {}


def _cached_encoder(model: str) -> Any:
    """One SentenceTransformer per model name, loaded once per process.

    compute_embeddings() instantiates SentenceTransformer from scratch on every
    call (a multi-second weight load); check_contamination() encodes per
    document, so the embedding path re-loaded the model once per (doc,
    benchmark) query -- days of pure weight-loading at 10k docs x 12
    benchmarks. Same encode() defaults as compute_embeddings, so vectors are
    identical.
    """
    encoder = _ENCODER_CACHE.get(model)
    if encoder is None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        encoder = SentenceTransformer(model)
        _ENCODER_CACHE[model] = encoder
    return encoder


# --------------------------------------------------------------------------------------
# Benchmark index for decontamination
# --------------------------------------------------------------------------------------

@dataclass
class BenchmarkItem:
    """One benchmark item for contamination checking."""
    benchmark: str
    item_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.benchmark}::{self.item_id}"


@dataclass
class ContaminationHit:
    """One contamination finding."""
    train_id: str
    benchmark: str
    benchmark_item_id: str
    ngram_overlap: float
    ngram_threshold: float
    embedding_similarity: float | None = None
    embedding_threshold: float | None = None


@dataclass
class ContaminationReport:
    """Aggregated contamination report for one benchmark."""
    benchmark: str
    total_items: int = 0
    contaminated_items: int = 0
    hits: list[ContaminationHit] = field(default_factory=list)
    ngram_threshold: float = CONTAMINATION_THRESHOLD
    embedding_threshold: float | None = EMBEDDING_THRESHOLD

    def contamination_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.contaminated_items / self.total_items

    def summary(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "total_items": self.total_items,
            "contaminated_items": self.contaminated_items,
            "contamination_rate": self.contamination_rate(),
            "ngram_threshold": self.ngram_threshold,
            "embedding_threshold": self.embedding_threshold,
            "hit_count": len(self.hits),
        }


class BenchmarkIndex:
    """Index of all benchmark items for contamination checking."""

    def __init__(self, ngram_n: int = NGRAM_N) -> None:
        """Initialize benchmark index.

        Args:
            ngram_n: N-gram size for overlap detection
        """
        self.ngram_n = ngram_n
        self.items: dict[str, BenchmarkItem] = {}  # key -> item
        self.item_hashes: dict[str, set[str]] = {}  # key -> its cached n-gram hashes
        self.ngram_index: dict[str, set[str]] = defaultdict(set)  # ngram_hash -> keys
        self.by_benchmark: dict[str, set[str]] = defaultdict(set)  # benchmark -> keys
        self.embeddings: np.ndarray | None = None
        self.embedding_ids: list[str] = []
        self._embedding_model: str | None = None

    def add(self, item: BenchmarkItem) -> None:
        """Index a benchmark item."""
        key = item.key()
        self.items[key] = item

        # N-gram index; hashes cached on the item -- query_ngram used to
        # re-hash every candidate's full text per query, an O(candidates x
        # text-length) hot loop that made 50k-item queries take hours.
        hashes = ngram_hashes(item.text, self.ngram_n)
        self.item_hashes[key] = hashes
        for h in hashes:
            self.ngram_index[h].add(key)

        # Benchmark grouping
        self.by_benchmark[item.benchmark].add(key)

    def build_embedding_index(self, model: str = DEFAULT_EMBED_MODEL) -> None:
        """Build embedding index for semantic contamination detection.

        Args:
            model: Sentence transformer model name
        """
        if not self.items:
            return

        texts = [item.text for item in self.items.values()]
        result = compute_embeddings(texts, model=model)
        self.embeddings = result.embeddings
        self.embedding_ids = list(self.items.keys())
        self._embedding_model = model

        log.info(f"built embedding index: {len(self.embedding_ids)} items, model={model}")

    def query_ngram(
        self,
        text: str,
        threshold: float = CONTAMINATION_THRESHOLD,
        benchmark: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find benchmark items with n-gram overlap above threshold.

        Args:
            text: Training text to check
            threshold: Overlap threshold
            benchmark: Optional benchmark filter

        Returns:
            List of (item_key, overlap_score) tuples
        """
        query_hashes = ngram_hashes(text, self.ngram_n)

        # Find candidates by shared n-grams
        candidates = set()
        for h in query_hashes:
            candidates.update(self.ngram_index.get(h, set()))

        # Verify actual overlap using the hashes cached at add() time, with a
        # size-ratio prefilter: Jaccard <= min(|A|,|B|)/max(|A|,|B|), so a
        # candidate whose gram-set size ratio sits below the threshold can
        # never reach it and is skipped without the set intersection.
        hits = []
        q_size = max(len(query_hashes), 1)
        for key in candidates:
            if benchmark is not None and self.items[key].benchmark != benchmark:
                continue
            item_hashes = self.item_hashes.get(key)
            if item_hashes is None:
                item_hashes = ngram_hashes(self.items[key].text, self.ngram_n)
                self.item_hashes[key] = item_hashes
            i_size = len(item_hashes)
            if min(q_size, i_size) / max(q_size, i_size, 1) < threshold:
                continue
            overlap = ngram_overlap(query_hashes, item_hashes)
            if overlap >= threshold:
                hits.append((key, overlap))

        # Sort by overlap descending
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def query_embedding(
        self,
        text: str,
        threshold: float = EMBEDDING_THRESHOLD,
        benchmark: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find benchmark items by embedding similarity.

        Args:
            text: Training text to check
            threshold: Cosine similarity threshold
            benchmark: Optional benchmark filter

        Returns:
            List of (item_key, similarity) tuples
        """
        if self.embeddings is None:
            return []

        # Compute query embedding (cached encoder: one weight load per process,
        # not one per query -- see _cached_encoder)
        query_emb = _cached_encoder(self._embedding_model or DEFAULT_EMBED_MODEL).encode(
            [text], show_progress_bar=False
        )[0]

        # Compute similarities
        similarities = self.embeddings @ query_emb

        # Filter and sort
        hits = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                key = self.embedding_ids[idx]
                if benchmark is None or self.items[key].benchmark == benchmark:
                    hits.append((key, float(sim)))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def get_benchmark_names(self) -> list[str]:
        return sorted(self.by_benchmark.keys())

    def count_items(self, benchmark: str | None = None) -> int:
        if benchmark is None:
            return len(self.items)
        return len(self.by_benchmark.get(benchmark, set()))


# --------------------------------------------------------------------------------------
# Contamination checking
# --------------------------------------------------------------------------------------

@dataclass
class TrainDocument:
    """One training document for contamination checking."""
    id: str
    text: str
    source: str = ""  # Dataset or file source
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecontaminationResult:
    """Result of decontamination process.

    Attributes:
        total_train_docs: Number of training documents checked
        contaminated_docs: Number found to be contaminated
        clean_docs: Set of clean document IDs
        hits: All contamination hits found
        reports: Per-benchmark contamination reports
    """
    total_train_docs: int = 0
    contaminated_docs: int = 0
    clean_docs: set[str] = field(default_factory=set)
    hits: list[ContaminationHit] = field(default_factory=list)
    reports: dict[str, ContaminationReport] = field(default_factory=dict)

    def contamination_rate(self) -> float:
        if self.total_train_docs == 0:
            return 0.0
        return self.contaminated_docs / self.total_train_docs

    def summary(self) -> dict[str, Any]:
        return {
            "total_train_docs": self.total_train_docs,
            "contaminated_docs": self.contaminated_docs,
            "clean_docs": len(self.clean_docs),
            "contamination_rate": self.contamination_rate(),
            "total_hits": len(self.hits),
            "benchmarks_with_contamination": len(self.reports),
        }


def check_contamination(
    train_docs: Sequence[TrainDocument],
    benchmark_index: BenchmarkIndex,
    *,
    ngram_threshold: float = CONTAMINATION_THRESHOLD,
    embedding_threshold: float = EMBEDDING_THRESHOLD,
    check_embeddings: bool = False,
    benchmark_filter: list[str] | None = None,
) -> DecontaminationResult:
    """Check training documents for benchmark contamination.

    Args:
        train_docs: Training documents to check
        benchmark_index: Pre-built index of benchmark items
        ngram_threshold: N-gram overlap threshold
        embedding_threshold: Embedding similarity threshold
        check_embeddings: Whether to run embedding checks
        benchmark_filter: Optional list of benchmarks to check

    Returns:
        DecontaminationResult with all findings
    """
    docs = list(train_docs)
    result = DecontaminationResult(total_train_docs=len(docs))
    if not docs or not benchmark_index.items:
        return result

    benchmarks = benchmark_filter or benchmark_index.get_benchmark_names()
    log.info(
        f"checking {len(docs)} train docs against {len(benchmarks)} benchmarks "
        f"(ngram_threshold={ngram_threshold}, embeddings={check_embeddings})"
    )

    # Initialize reports
    for bench in benchmarks:
        result.reports[bench] = ContaminationReport(
            benchmark=bench,
            total_items=benchmark_index.count_items(bench),
            ngram_threshold=ngram_threshold,
            embedding_threshold=embedding_threshold if check_embeddings else None,
        )

    # Check each document
    contaminated_doc_ids = set()
    wanted_benchmarks = set(benchmarks)

    for doc in docs:
        doc_contaminated = False

        # One unfiltered query per document, partitioned by the hit's own
        # benchmark. The old loop re-queried once per benchmark: each call
        # re-hashed the document and re-scanned the whole candidate set, so 12
        # benchmarks cost 12x the hot path for exactly the same hit set (the
        # benchmark filter is applied per candidate after retrieval anyway).
        ngram_hits = benchmark_index.query_ngram(
            doc.text,
            threshold=ngram_threshold,
        )

        doc_ngram_item_ids: set[str] = set()
        for key, overlap in ngram_hits:
            item = benchmark_index.items[key]
            bench = item.benchmark
            if bench not in wanted_benchmarks:
                continue  # benchmark_filter parity with the per-benchmark loop
            hit = ContaminationHit(
                train_id=doc.id,
                benchmark=bench,
                benchmark_item_id=item.item_id,
                ngram_overlap=overlap,
                ngram_threshold=ngram_threshold,
            )
            result.hits.append(hit)
            result.reports[bench].hits.append(hit)
            doc_ngram_item_ids.add(item.item_id)
            doc_contaminated = True

            if doc.id not in contaminated_doc_ids:
                result.contaminated_docs += 1
                contaminated_doc_ids.add(doc.id)

        # Optional embedding check (only if not already contaminated by ngrams)
        if check_embeddings and benchmark_index.embeddings is not None:
            # Skip ids this doc already hit via ngrams: a per-doc set built in
            # the loop above. (A set rebuilt per doc from EVERY accumulated hit
            # was still O(docs x total hits) -- only this doc's own hits can
            # ever carry its train_id.)
            seen_ids = doc_ngram_item_ids

            # One unfiltered query per document, partitioned by benchmark
            # (identical hit set; the encoder is cached, and this encodes the
            # document once instead of once per benchmark)
            embed_hits = benchmark_index.query_embedding(
                doc.text,
                threshold=embedding_threshold,
            )

            for key, similarity in embed_hits:
                item = benchmark_index.items[key]
                bench = item.benchmark
                if bench not in wanted_benchmarks:
                    continue

                # Skip if already found via ngrams (or an earlier embedding hit)
                if item.item_id in seen_ids:
                    continue
                seen_ids.add(item.item_id)

                hit = ContaminationHit(
                    train_id=doc.id,
                    benchmark=bench,
                    benchmark_item_id=item.item_id,
                    ngram_overlap=0.0,
                    ngram_threshold=ngram_threshold,
                    embedding_similarity=similarity,
                    embedding_threshold=embedding_threshold,
                )
                result.hits.append(hit)
                result.reports[bench].hits.append(hit)
                doc_contaminated = True

                if doc.id not in contaminated_doc_ids:
                    result.contaminated_docs += 1
                    contaminated_doc_ids.add(doc.id)

        if not doc_contaminated:
            result.clean_docs.add(doc.id)

    # Update report statistics
    for report in result.reports.values():
        seen_items = {h.benchmark_item_id for h in report.hits}
        report.contaminated_items = len(seen_items)

    log.info(
        f"contamination check complete: {result.contaminated_docs}/{len(docs)} docs contaminated "
        f"({result.contamination_rate():.1%}), {len(result.hits)} total hits"
    )

    return result


# --------------------------------------------------------------------------------------
# Negative-control test generation
# --------------------------------------------------------------------------------------

@dataclass
class NegativeControl:
    """A negative-control test case."""
    id: str
    text: str
    source_benchmark: str
    source_item_id: str
    modification: str  # "paraphrase", "shuffle", "negate", "perturb"


def generate_negative_controls(
    benchmark_items: Iterable[BenchmarkItem],
    *,
    modifications: list[str] | None = None,
    seed: int = 42,
) -> list[NegativeControl]:
    """Generate negative-control test cases from benchmark items.

    Negative controls are modified versions of benchmark items that should NOT
    be detected as contamination. They test that the decontamination pipeline
    is not overly aggressive.

    Args:
        benchmark_items: Source benchmark items
        modifications: List of modification types
        seed: Random seed for reproducibility

    Returns:
        List of NegativeControl items
    """
    import random

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for negative control generation: "
            "pip install numpy"
        ) from exc

    random.seed(seed)
    np.random.seed(seed)

    if modifications is None:
        modifications = ["paraphrase", "shuffle", "negate", "perturb"]

    items = list(benchmark_items)
    controls = []

    log.info(f"generating negative controls from {len(items)} items, modifications={modifications}")

    for item in items:
        for mod_type in modifications:
            try:
                if mod_type == "paraphrase":
                    # Simple paraphrase by word substitution (placeholder for real model)
                    text = _paraphrase_simple(item.text)
                elif mod_type == "shuffle":
                    # Shuffle sentence order
                    text = _shuffle_sentences(item.text)
                elif mod_type == "negate":
                    # Negate key terms (changes meaning significantly)
                    text = _negate_text(item.text)
                elif mod_type == "perturb":
                    # Add noise/perturbation
                    text = _perturb_text(item.text)
                else:
                    continue

                control = NegativeControl(
                    id=f"neg_{item.item_id}_{mod_type}",
                    text=text,
                    source_benchmark=item.benchmark,
                    source_item_id=item.item_id,
                    modification=mod_type,
                )
                controls.append(control)
            except Exception as e:
                log.debug(f"failed to generate negative control for {item.key()}/{mod_type}: {e}")

    log.info(f"generated {len(controls)} negative controls")
    return controls


def _paraphrase_simple(text: str) -> str:
    """Very simple paraphrase by word swap (placeholder for actual model)."""
    words = text.split()
    # Swap some common synonyms
    swaps = {
        "patient": "individual",
        "treatment": "therapy",
        "disease": "condition",
        "diagnosis": "assessment",
        "symptom": "manifestation",
    }
    result = []
    for w in words:
        result.append(swaps.get(w.lower(), w))
    return " ".join(result)


def _shuffle_sentences(text: str) -> str:
    """Shuffle sentence order while keeping content."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) <= 1:
        return text
    np.random.shuffle(sentences)
    return ". ".join(sentences) + "."


def _negate_text(text: str) -> str:
    """Negate key terms to change meaning."""
    negations = {
        "is": "is not",
        "are": "are not",
        "treat": "does not treat",
        "causes": "does not cause",
        "effective": "ineffective",
    }
    for orig, neg in negations.items():
        text = text.replace(orig, neg)
    return text


def _perturb_text(text: str) -> str:
    """Add random perturbations."""
    words = text.split()
    if len(words) < 3:
        return text
    # Insert a few random words
    filler = ["the", "a", "an", "some"]
    for _ in range(min(2, len(words) // 10)):
        idx = np.random.randint(0, len(words))
        words.insert(idx, np.random.choice(filler))
    return " ".join(words)


# --------------------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------------------

def save_contamination_report(result: DecontaminationResult, path: Path) -> None:
    """Save contamination report to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": result.summary(),
        "benchmarks": {name: r.summary() for name, r in result.reports.items()},
        "sample_hits": [
            {
                "train_id": h.train_id,
                "benchmark": h.benchmark,
                "benchmark_item_id": h.benchmark_item_id,
                "ngram_overlap": h.ngram_overlap,
                "embedding_similarity": h.embedding_similarity,
            }
            for h in result.hits[:200]  # Limit output size
        ],
    }

    with path.open("w") as f:
        json.dump(report, f, indent=2)

    log.info(f"saved contamination report to {path}")


def generate_per_benchmark_reports(
    result: DecontaminationResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate individual reports per benchmark.

    Args:
        result: Decontamination result
        output_dir: Directory to write reports to

    Returns:
        Dict mapping benchmark name to report file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for bench_name, report in result.reports.items():
        report_path = output_dir / f"{bench_name}_contamination.json"
        with report_path.open("w") as f:
            json.dump({
                "benchmark": bench_name,
                "summary": report.summary(),
                "hits": [
                    {
                        "train_id": h.train_id,
                        "benchmark_item_id": h.benchmark_item_id,
                        "ngram_overlap": h.ngram_overlap,
                        "embedding_similarity": h.embedding_similarity,
                    }
                    for h in report.hits
                ],
            }, f, indent=2)
        paths[bench_name] = report_path

    log.info(f"generated {len(paths)} per-benchmark reports in {output_dir}")
    return paths


# --------------------------------------------------------------------------------------
# Loading benchmarks for index
# --------------------------------------------------------------------------------------

def load_benchmark_items_from_loaders(
    benchmark_names: Iterable[str],
) -> Iterator[BenchmarkItem]:
    """Load benchmark items via the eval loaders.

    This provides a bridge between the eval system (which knows how to load
    benchmarks) and the decontamination system.

    Args:
        benchmark_names: Names of benchmarks to load

    Yields:
        BenchmarkItem objects
    """
    from medrl.eval.loaders import load_items
    from medrl.eval.tasks.benchmarks import TASKS

    for name in benchmark_names:
        if name not in TASKS:
            log.warning(f"unknown benchmark: {name}")
            continue

        spec = TASKS[name]  # type: ignore[index]
        try:
            load_result = load_items(spec)
            for item in load_result.items:
                # Extract text from messages
                text_parts = []
                for msg in item.messages:
                    if msg.get("content"):
                        text_parts.append(msg["content"])
                text = "\n\n".join(text_parts)

                yield BenchmarkItem(
                    benchmark=name,
                    item_id=item.item_id,
                    text=text,
                    metadata={"split": spec.split, "revision": spec.revision},
                )
        except Exception as e:
            log.warning(f"failed to load benchmark {name}: {e}")


def build_benchmark_index(
    benchmark_names: list[str],
    *,
    include_embeddings: bool = False,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> BenchmarkIndex:
    """Build a complete benchmark index for decontamination.

    Args:
        benchmark_names: List of benchmark names to index
        include_embeddings: Whether to build embedding index
        embed_model: Model for embeddings

    Returns:
        Populated BenchmarkIndex
    """
    index = BenchmarkIndex(ngram_n=NGRAM_N)

    for item in load_benchmark_items_from_loaders(benchmark_names):
        index.add(item)

    log.info(f"built benchmark index: {len(index.items)} items from {len(benchmark_names)} benchmarks")

    if include_embeddings:
        index.build_embedding_index(model=embed_model)

    return index


__all__ = [
    "CONTAMINATION_THRESHOLD",
    "DEFAULT_EMBED_MODEL",
    "EMBEDDING_THRESHOLD",
    "NGRAM_N",
    # Aliases for compatibility with __init__.py
    "BenchmarkDecontaminator",
    "BenchmarkIndex",
    "BenchmarkItem",
    "ContaminationHit",
    "ContaminationReport",
    "DecontaminationResult",
    "NegativeControl",
    "TrainDocument",
    "build_benchmark_index",
    "check_contamination",
    "exact_decontam",
    "generate_decontam_report",
    "generate_negative_controls",
    "generate_per_benchmark_reports",
    "load_benchmark_items_from_loaders",
    "ngram_decontam",
    "save_contamination_report",
    "semantic_decontam",
]


# =============================================================================
# API aliases for compatibility with __init__.py
# =============================================================================

# BenchmarkDecontaminator is an alias for BenchmarkIndex
BenchmarkDecontaminator = BenchmarkIndex


def exact_decontam(
    train_docs: Sequence[TrainDocument],
    benchmark_texts: Iterable[str],
) -> tuple[list[TrainDocument], list[TrainDocument]]:
    """Exact match decontamination using content hashing.

    Simple and fast decontamination that catches exact duplicates.
    Consider using ngram_decontam for near-duplicate detection.

    Args:
        train_docs: Training documents to check
        benchmark_texts: Benchmark text strings to check against

    Returns:
        Tuple of (clean_docs, contaminated_docs)
    """

    benchmark_hashes = {
        hashlib.sha256(text.lower().strip().encode()).hexdigest()
        for text in benchmark_texts
    }

    clean = []
    contaminated = []

    for doc in train_docs:
        doc_hash = hashlib.sha256(doc.text.lower().strip().encode()).hexdigest()
        if doc_hash in benchmark_hashes:
            contaminated.append(doc)
        else:
            clean.append(doc)

    log.info(
        f"exact_decontam: {len(contaminated)}/{len(train_docs)} contaminated "
        f"({len(clean)} clean)"
    )

    return clean, contaminated


def ngram_decontam(
    train_docs: Sequence[TrainDocument],
    benchmark_texts: Iterable[str],
    *,
    threshold: float = CONTAMINATION_THRESHOLD,
) -> tuple[list[TrainDocument], list[TrainDocument]]:
    """N-gram overlap decontamination.

    Uses 13-gram overlap to detect near-duplicates. More permissive than
    exact_decontam but catches paraphrases and minor edits.

    Args:
        train_docs: Training documents to check
        benchmark_texts: Benchmark text strings to check against
        threshold: Overlap threshold (default 0.8)

    Returns:
        Tuple of (clean_docs, contaminated_docs)
    """
    # Build index from benchmark texts
    index = NGramIndex(ngram_n=NGRAM_N)
    for i, text in enumerate(benchmark_texts):
        index.add(f"benchmark_{i}", text)

    clean = []
    contaminated = []

    for doc in train_docs:
        matches = index.query(doc.text, threshold=threshold)
        if matches:
            contaminated.append(doc)
        else:
            clean.append(doc)

    log.info(
        f"ngram_decontam: {len(contaminated)}/{len(train_docs)} contaminated "
        f"({len(clean)} clean, threshold={threshold})"
    )

    return clean, contaminated


def semantic_decontam(
    train_docs: Sequence[TrainDocument],
    benchmark_texts: Iterable[str],
    *,
    threshold: float = EMBEDDING_THRESHOLD,
    model: str = DEFAULT_EMBED_MODEL,
) -> tuple[list[TrainDocument], list[TrainDocument]]:
    """Semantic decontamination using embeddings.

    Uses sentence transformer embeddings to detect semantic similarity.
    More expensive but catches meaning-preserving paraphrases.

    Args:
        train_docs: Training documents to check
        benchmark_texts: Benchmark text strings to check against
        threshold: Cosine similarity threshold (default 0.9)
        model: Sentence transformer model name

    Returns:
        Tuple of (clean_docs, contaminated_docs)
    """
    from medrl.data.dedup import compute_embeddings

    # Compute embeddings for benchmarks
    bench_texts = list(benchmark_texts)
    bench_embeddings = compute_embeddings(bench_texts, model=model)

    # Encode all documents in ONE batched call: the old loop re-loaded the
    # model and re-encoded per document (a full weight load per doc).
    docs = list(train_docs)
    doc_embeddings = (
        compute_embeddings([d.text for d in docs], model=model).embeddings
        if docs
        else []
    )

    # Check each training doc
    clean = []
    contaminated = []

    for doc, doc_emb in zip(docs, doc_embeddings, strict=True):
        # Compute max similarity to any benchmark
        similarities = bench_embeddings.embeddings @ doc_emb
        if similarities.max() >= threshold:
            contaminated.append(doc)
        else:
            clean.append(doc)

    log.info(
        f"semantic_decontam: {len(contaminated)}/{len(train_docs)} contaminated "
        f"({len(clean)} clean, threshold={threshold})"
    )

    return clean, contaminated


# generate_decontam_report is an alias for save_contamination_report
generate_decontam_report = save_contamination_report


class NGramIndex:
    """Simple n-gram index for decontamination."""

    def __init__(self, ngram_n: int = NGRAM_N) -> None:
        self.ngram_n = ngram_n
        self.index: dict[str, set[str]] = defaultdict(set)
        self.doc_hashes: dict[str, set[str]] = {}

    def add(self, doc_id: str, text: str) -> None:
        hashes = ngram_hashes(text, self.ngram_n)
        self.doc_hashes[doc_id] = hashes
        for h in hashes:
            self.index[h].add(doc_id)

    def query(self, text: str, threshold: float = CONTAMINATION_THRESHOLD) -> set[str]:
        query_hashes = ngram_hashes(text, self.ngram_n)
        candidates = set()
        for h in query_hashes:
            candidates.update(self.index.get(h, set()))
        # Verify overlap
        duplicates = set()
        for doc_id in candidates:
            overlap = ngram_overlap(query_hashes, self.doc_hashes[doc_id])
            if overlap >= threshold:
                duplicates.add(doc_id)
        return duplicates
