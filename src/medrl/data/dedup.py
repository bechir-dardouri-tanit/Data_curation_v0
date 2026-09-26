"""Deduplication utilities for the medrl pipeline.

Provides near-duplicate detection using MinHash (for exact/near-exact duplicates)
and semantic embeddings (for paraphrase detection). Used to remove redundant
training examples that would cause overfitting or waste compute.

Typical usage:
    >>> from medrl.data.dedup import minhash_dedup, cluster_dedup
    >>> items = [{"text": "Aspirin reduces pain."}, {"text": "Aspirin reduces pain."}]
    >>> deduped, hashes = minhash_dedup(items, threshold=0.9)
    >>> len(deduped)
    1
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from medrl.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# MinHash Deduplication
# =============================================================================


def _normalize_text(text: str) -> str:
    """Normalize text for deduplication.

    Lowercases, strips whitespace, and removes extra spaces for more robust
    duplicate detection.

    Args:
        text: The text to normalize.

    Returns:
        Normalized text string.

    Examples:
        >>> _normalize_text("  Hello   World  ")
        'hello world'
    """
    return " ".join(text.lower().strip().split())


def ngram_hashes(text: str, n: int) -> set[str]:
    """Compute n-gram hashes for text.

    Used by both deduplication and decontamination.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        Set of hash strings representing each n-gram.

    Examples:
        >>> hashes = ngram_hashes("Hello world", 2)
        >>> len(hashes) > 0
        True
    """
    import hashlib

    normalized = _normalize_text(text)
    if len(normalized) < n:
        # For short texts, hash the entire text
        return {hashlib.sha256(normalized.encode()).hexdigest()[:16]}
    return {
        hashlib.sha256(normalized[i : i + n].encode()).hexdigest()[:16]
        for i in range(len(normalized) - n + 1)
    }


def ngram_overlap(hashes1: set[str], hashes2: set[str]) -> float:
    """Compute Jaccard overlap between two sets of n-gram hashes.

    Args:
        hashes1: First set of n-gram hashes.
        hashes2: Second set of n-gram hashes.

    Returns:
        Jaccard similarity (0-1).

    Examples:
        >>> h1 = ngram_hashes("Hello world", 2)
        >>> h2 = ngram_hashes("Hello world", 2)
        >>> ngram_overlap(h1, h2)
        1.0
    """
    if not hashes1 or not hashes2:
        return 0.0
    intersection = len(hashes1 & hashes2)
    union = len(hashes1 | hashes2)
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class EmbeddingResult:
    """Result from compute_embeddings.

    Attributes:
        embeddings: NumPy array of embeddings.
        model: Model name used.
    """

    embeddings: Any
    model: str


def compute_embeddings(
    texts: Sequence[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> EmbeddingResult:
    """Compute sentence embeddings for texts.

    Args:
        texts: List of text strings.
        model: Sentence transformer model name.
        batch_size: Batch size for computation.

    Returns:
        EmbeddingResult with embeddings array.

    Raises:
        ImportError: If sentence-transformers is not installed.

    Examples:
        >>> result = compute_embeddings(["Hello world"], batch_size=1)
        >>> result.embeddings.shape[0]
        1
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e

    sentence_model = SentenceTransformer(model)
    embeddings = sentence_model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    return EmbeddingResult(embeddings=embeddings, model=model)


def minhash_dedup(
    items: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    threshold: float = 0.9,
    num_perm: int = 128,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Remove near-duplicates using MinHash LSH.

    MinHash is a probabilistic data structure for estimating Jaccard similarity.
    Locality-Sensitive Hashing (LSH) enables efficient approximate nearest neighbor
    search. Items with similarity above `threshold` are considered duplicates.

    Args:
        items: Sequence of items (dicts or dataclasses).
        text_key: Key to extract text from each item.
        threshold: Similarity threshold (0-1). Higher = stricter deduplication.
        num_perm: Number of MinHash permutations (higher = more accurate, slower).
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (deduped_items, hashes_dict) where `hashes_dict` maps
        item index to its MinHash signature for debugging.

    Raises:
        ImportError: If datasketch is not installed.

    Examples:
        >>> items = [
        ...     {"text": "Aspirin treats pain.", "id": 1},
        ...     {"text": "Aspirin treats pain.", "id": 2},
        ...     {"text": "Ibuprofen treats pain.", "id": 3},
        ... ]
        >>> deduped, _ = minhash_dedup(items, threshold=0.95)
        >>> len(deduped)
        2
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as e:
        raise ImportError(
            "datasketch is required for MinHash deduplication. "
            "Install with: pip install datasketch"
        ) from e

    if not items:
        logger.warning("Empty item list, nothing to deduplicate")
        return [], {}

    # Create LSH index
    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
        weights=(0.5, 0.5),  # Balanced false positive/negative rate
    )

    # Build MinHash signatures and index
    minhashes: list[MinHash] = []
    kept_indices: list[int] = []
    duplicate_count = 0

    for idx, item in enumerate(items):
        text = str(item.get(text_key, ""))
        normalized = _normalize_text(text)

        if not normalized:
            logger.debug(f"Item {idx} has empty text after normalization, skipping")
            continue

        # Create MinHash
        mh = MinHash(num_perm=num_perm, hashfunc=lambda x: int(x, 16))
        for word in normalized.split():
            mh.update(word.encode("utf-8"))

        # Check for duplicates
        duplicates = lsh.query(mh)
        if duplicates:
            duplicate_count += 1
            logger.debug(f"Item {idx} is duplicate of {duplicates[0]}, skipping")
            continue

        # Add to index
        lsh.insert(str(idx), mh)
        minhashes.append(mh)
        kept_indices.append(idx)

    logger.info(
        f"MinHash dedup: {len(items)} -> {len(kept_indices)} items "
        f"({duplicate_count} duplicates removed, threshold={threshold})"
    )

    deduped = [items[i] for i in kept_indices]
    hashes_dict = {str(i): mh.hashvalues.tobytes().hex() for i, mh in zip(kept_indices, minhashes, strict=False)}

    return deduped, hashes_dict


def exact_dedup(
    items: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Remove exact duplicates using content hashing.

    Faster than MinHash but only catches exact matches. Text normalization is
    applied to catch trivial differences like capitalization or extra spaces.

    Args:
        items: Sequence of items (dicts or dataclasses).
        text_key: Key to extract text from each item.

    Returns:
        A tuple of (deduped_items, hashes_dict) where `hashes_dict` maps
        content hash to the text that produced it.

    Examples:
        >>> items = [{"text": "Hello"}, {"text": "hello"}, {"text": "World"}]
        >>> deduped, hashes = exact_dedup(items)
        >>> len(deduped)
        2
    """
    import hashlib

    seen: dict[str, str] = {}  # hash -> normalized text
    kept: list[dict[str, Any]] = []
    duplicate_count = 0

    for item in items:
        text = str(item.get(text_key, ""))
        normalized = _normalize_text(text)

        if not normalized:
            continue

        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if content_hash in seen:
            duplicate_count += 1
            logger.debug(f"Duplicate found: {normalized[:50]}...")
            continue

        seen[content_hash] = normalized
        kept.append(item)

    logger.info(f"Exact dedup: {len(items)} -> {len(kept)} items ({duplicate_count} duplicates)")

    return kept, seen


# =============================================================================
# N-gram Deduplication
# =============================================================================


def ngram_dedup(
    items: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    n: int = 13,
    threshold: float = 0.9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove near-duplicates using n-gram overlap.

    Uses 13-gram hashes by default, providing good specificity for medical text
    while allowing some tolerance for minor edits. More permissive than exact
    dedup but catches near-duplicates that MinHash might miss.

    Args:
        items: Sequence of items (dicts or dataclasses).
        text_key: Key to extract text from each item.
        n: N-gram size (default 13).
        threshold: Jaccard overlap threshold (0-1).

    Returns:
        A tuple of (deduped_items, metadata_dict) with statistics.

    Examples:
        >>> items = [
        ...     {"text": "The patient has a headache and fever."},
        ...     {"text": "The patient has a headache with fever."},
        ...     {"text": "Completely different text here."},
        ... ]
        >>> deduped, _ = ngram_dedup(items, threshold=0.7)
        >>> len(deduped) <= 2
        True
    """
    import hashlib

    if not items:
        logger.warning("Empty item list, nothing to deduplicate")
        return [], {}

    def get_ngram_hashes(text: str) -> set[str]:
        """Compute all n-gram hashes for text."""
        normalized = _normalize_text(text)
        if len(normalized) < n:
            return {hashlib.sha256(normalized.encode()).hexdigest()[:16]}
        return {
            hashlib.sha256(normalized[i:i + n].encode()).hexdigest()[:16]
            for i in range(len(normalized) - n + 1)
        }

    def jaccard_overlap(set1: set[str], set2: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 0.0

    # Index n-grams
    ngram_index: dict[str, list[int]] = {}
    hashes_list: list[set[str]] = []

    for idx, item in enumerate(items):
        text = str(item.get(text_key, ""))
        hashes = get_ngram_hashes(text)
        hashes_list.append(hashes)
        for h in hashes:
            if h not in ngram_index:
                ngram_index[h] = []
            ngram_index[h].append(idx)

    # Find duplicates
    kept: list[bool] = [True] * len(items)
    duplicate_count = 0
    duplicate_pairs: list[tuple[int, int, float]] = []

    for i in range(len(items)):
        if not kept[i]:
            continue
        query_hashes = hashes_list[i]

        # Find candidates via shared n-grams
        candidates = set()
        for h in query_hashes:
            candidates.update(ngram_index.get(h, []))

        # Verify with Jaccard
        for j in candidates:
            if j <= i or not kept[j]:
                continue
            overlap = jaccard_overlap(query_hashes, hashes_list[j])
            if overlap >= threshold:
                kept[j] = False
                duplicate_count += 1
                duplicate_pairs.append((i, j, overlap))
                logger.debug(f"Items {i} and {j} are n-gram duplicates (overlap={overlap:.3f})")

    deduped = [item for item, keep in zip(items, kept, strict=False) if keep]

    metadata = {
        "ngram_n": n,
        "threshold": threshold,
        "duplicates_removed": duplicate_count,
        "duplicate_pairs": duplicate_pairs[:50],  # Limit for output
    }

    logger.info(
        f"N-gram dedup: {len(items)} -> {len(deduped)} items "
        f"({duplicate_count} duplicates removed, n={n}, threshold={threshold})"
    )

    return deduped, metadata


# =============================================================================
# Cross-Stage Deduplication
# =============================================================================


@dataclass
class Document:
    """Document for cross-stage deduplication."""
    id: str
    text: str
    stage: str = ""  # e.g., "sft", "rl", "eval"
    source: str = ""  # dataset or file source


def cross_stage_dedup(
    stage_a: Sequence[Document],
    stage_b: Sequence[Document],
    *,
    method: str = "ngram",
    n: int = 13,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Deduplicate between two data stages (e.g., SFT vs RL).

    Ensures training data does not leak between stages or into evaluation.
    Returns contamination findings in both directions.

    Args:
        stage_a: Documents from stage A (e.g., SFT training)
        stage_b: Documents from stage B (e.g., RL training)
        method: Deduplication method ("ngram" or "minhash")
        n: N-gram size for ngram method
        threshold: Similarity threshold

    Returns:
        Dictionary with 'a_to_b' and 'b_to_a' contamination results.

    Examples:
        >>> sft_docs = [Document("1", "Text about aspirin", stage="sft")]
        >>> rl_docs = [Document("2", "Text about aspirin", stage="rl")]
        >>> result = cross_stage_dedup(sft_docs, rl_docs)
        >>> result["b_to_a"]["contaminated_count"]
        1
    """
    from dataclasses import dataclass

    @dataclass
    class StageResult:
        stage_name: str
        total_count: int
        contaminated_ids: set[str]
        clean_ids: set[str]
        hits: list[tuple[str, str, float]]  # (source_id, target_id, similarity)

    docs_a = list(stage_a)
    docs_b = list(stage_b)

    logger.info(
        f"Cross-stage dedup: {len(docs_a)} stage A docs, {len(docs_b)} stage B docs, "
        f"method={method}, threshold={threshold}"
    )

    # Build index from stage B
    if method == "ngram":
        # Build n-gram index for B
        import hashlib

        ngram_index: dict[str, list[str]] = {}
        doc_hashes_b: dict[str, set[str]] = {}

        def get_ngram_hashes(text: str) -> set[str]:
            normalized = _normalize_text(text)
            if len(normalized) < n:
                return {hashlib.sha256(normalized.encode()).hexdigest()[:16]}
            return {
                hashlib.sha256(normalized[i:i + n].encode()).hexdigest()[:16]
                for i in range(len(normalized) - n + 1)
            }

        def jaccard_overlap(set1: set[str], set2: set[str]) -> float:
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)

        for doc in docs_b:
            hashes = get_ngram_hashes(doc.text)
            doc_hashes_b[doc.id] = hashes
            for h in hashes:
                if h not in ngram_index:
                    ngram_index[h] = []
                ngram_index[h].append(doc.id)

        def query_ngrams(text: str) -> set[tuple[str, float]]:
            query_hashes = get_ngram_hashes(text)
            candidates = set()
            for h in query_hashes:
                candidates.update(ngram_index.get(h, []))
            results = set()
            for cand_id in candidates:
                overlap = jaccard_overlap(query_hashes, doc_hashes_b[cand_id])
                if overlap >= threshold:
                    results.add((cand_id, overlap))
            return results

        query_fn = query_ngrams

    elif method == "minhash":
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError as e:
            raise ImportError("datasketch required for minhash cross-stage dedup") from e

        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes_b: dict[str, MinHash] = {}

        for doc in docs_b:
            mh = MinHash(num_perm=128, hashfunc=lambda x: int(x, 16))
            for word in _normalize_text(doc.text).split():
                mh.update(word.encode("utf-8"))
            minhashes_b[doc.id] = mh
            lsh.insert(doc.id, mh)

        def query_minhash(text: str) -> set[tuple[str, float]]:
            mh = MinHash(num_perm=128, hashfunc=lambda x: int(x, 16))
            for word in _normalize_text(text).split():
                mh.update(word.encode("utf-8"))
            # LSH returns approximate matches
            matches = lsh.query(mh)
            results = set()
            for match_id in matches:
                # Verify with actual Jaccard
                jaccard = mh.jaccard(minhashes_b[match_id])
                if jaccard >= threshold:
                    results.add((match_id, jaccard))
            return results

        query_fn = query_minhash
    else:
        raise ValueError(f"Unknown method: {method}")

    # Check A against B
    a_contaminated = set()
    a_hits = []
    for doc in docs_a:
        matches = query_fn(doc.text)
        for match_id, score in matches:
            a_contaminated.add(doc.id)
            a_hits.append((doc.id, match_id, score))
            logger.debug(f"Cross-stage hit A->B: {doc.id} matches {match_id} (score={score:.3f})")

    result_a = StageResult(
        stage_name="A",
        total_count=len(docs_a),
        contaminated_ids=a_contaminated,
        clean_ids={d.id for d in docs_a} - a_contaminated,
        hits=a_hits,
    )

    # Check B against A
    b_contaminated = set()
    b_hits = []
    for doc in docs_b:
        matches = query_fn(doc.text)
        for match_id, score in matches:
            b_contaminated.add(doc.id)
            b_hits.append((doc.id, match_id, score))
            logger.debug(f"Cross-stage hit B->A: {doc.id} matches {match_id} (score={score:.3f})")

    result_b = StageResult(
        stage_name="B",
        total_count=len(docs_b),
        contaminated_ids=b_contaminated,
        clean_ids={d.id for d in docs_b} - b_contaminated,
        hits=b_hits,
    )

    result = {
        "a_to_b": {
            "stage_name": "A_to_B",
            "total_count": result_a.total_count,
            "contaminated_count": len(result_a.contaminated_ids),
            "clean_count": len(result_a.clean_ids),
            "contamination_rate": len(result_a.contaminated_ids) / result_a.total_count if result_a.total_count > 0 else 0.0,
            "hits": result_a.hits[:100],
        },
        "b_to_a": {
            "stage_name": "B_to_A",
            "total_count": result_b.total_count,
            "contaminated_count": len(result_b.contaminated_ids),
            "clean_count": len(result_b.clean_ids),
            "contamination_rate": len(result_b.contaminated_ids) / result_b.total_count if result_b.total_count > 0 else 0.0,
            "hits": result_b.hits[:100],
        },
        "method": method,
        "threshold": threshold,
    }

    logger.info(
        f"Cross-stage dedup complete: "
        f"A->B: {len(result_a.contaminated_ids)}/{len(docs_a)} contaminated, "
        f"B->A: {len(result_b.contaminated_ids)}/{len(docs_b)} contaminated"
    )

    return result


# =============================================================================
# Semantic Deduplication
# =============================================================================


def semantic_dedup(
    items: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    threshold: float = 0.95,
    batch_size: int = 32,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict[str, Any]]:
    """Remove semantic duplicates using sentence embeddings.

    Uses sentence transformer embeddings to detect paraphrases and semantically
    similar text. More expensive than MinHash but catches meaning-preserving
    rephrasing.

    Args:
        items: Sequence of items (dicts or dataclasses).
        text_key: Key to extract text from each item.
        threshold: Cosine similarity threshold (0-1). Higher = stricter.
        batch_size: Batch size for embedding computation.
        model_name: HuggingFace model name for embeddings.

    Returns:
        Deduplicated items list.

    Raises:
        ImportError: If sentence-transformers is not installed.

    Examples:
        >>> items = [
        ...     {"text": "Aspirin reduces pain."},
        ...     {"text": "Aspirin alleviates pain."},  # Semantic duplicate
        ...     {"text": "Ibuprofen helps with pain."},
        ... ]
        >>> deduped = semantic_dedup(items, threshold=0.9)
        >>> len(deduped) <= 2
        True
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for semantic deduplication. "
            "Install with: pip install sentence-transformers"
        ) from e

    if not items:
        logger.warning("Empty item list, nothing to deduplicate")
        return []

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract texts
    texts = [str(item.get(text_key, "")) for item in items]

    # Compute embeddings
    logger.info(f"Computing embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    # Compute pairwise similarities and deduplicate
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-not-found]

    sim_matrix = cosine_similarity(embeddings)

    kept: list[bool] = [True] * len(items)
    duplicate_count = 0

    for i in range(len(items)):
        if not kept[i]:
            continue
        for j in range(i + 1, len(items)):
            if sim_matrix[i, j] >= threshold and kept[j]:
                duplicate_count += 1
                kept[j] = False
                logger.debug(f"Items {i} and {j} are semantic duplicates (sim={sim_matrix[i,j]:.2f})")

    deduped = [item for item, keep in zip(items, kept, strict=False) if keep]

    logger.info(
        f"Semantic dedup: {len(items)} -> {len(deduped)} items "
        f"({duplicate_count} duplicates removed, threshold={threshold})"
    )

    return deduped


# =============================================================================
# Clustering-Based Deduplication
# =============================================================================


def cluster_dedup(
    items: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    min_cluster_size: int = 2,
    cluster_selection_epsilon: float = 0.1,
    sample_size: int | None = None,
) -> dict[str, list[int]]:
    """Find duplicate clusters using HDBSCAN on embeddings.

    Unlike pairwise deduplication, clustering finds groups of similar items
    without quadratic scaling. Returns cluster assignments for manual review
    or automatic filtering.

    Args:
        items: Sequence of items (dicts or dataclasses).
        text_key: Key to extract text from each item.
        min_cluster_size: Minimum cluster size to report.
        cluster_selection_epsilon: Distance threshold for clustering.
        sample_size: Subsample items for faster clustering on large datasets.

    Returns:
        Dictionary mapping cluster_id to list of item indices. Items not in
        any cluster are not included.

    Raises:
        ImportError: If hdbscan or sentence-transformers is not installed.

    Examples:
        >>> items = [
        ...     {"text": "Aspirin helps pain."},
        ...     {"text": "Aspirin reduces pain."},
        ...     {"text": "Ibuprofen helps pain."},
        ... ]
        >>> clusters = cluster_dedup(items, min_cluster_size=2)
        >>> len(clusters)  # One cluster of two aspirin items
        1
    """
    try:
        from hdbscan import HDBSCAN  # type: ignore[import-not-found]
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "hdbscan and sentence-transformers are required for clustering. "
            "Install with: pip install hdbscan sentence-transformers"
        ) from e

    if len(items) < min_cluster_size:
        logger.info(f"Fewer items ({len(items)}) than min_cluster_size ({min_cluster_size})")
        return {}

    # Subsample if requested
    if sample_size and len(items) > sample_size:
        import random

        indices = random.sample(range(len(items)), sample_size)
        items = [items[i] for i in indices]
        logger.info(f"Subsampled to {sample_size} items for clustering")

    # Compute embeddings
    logger.info(f"Computing embeddings for {len(items)} items...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [str(item.get(text_key, "")) for item in items]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    # Cluster
    logger.info("Running HDBSCAN clustering...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method="eom",
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    labels = clusterer.fit_predict(embeddings)

    # Group by cluster
    clusters: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        if label >= 0:  # -1 is noise
            cluster_id = f"cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(idx)

    # Log summary
    total_clustered = sum(len(v) for v in clusters.values())
    logger.info(
        f"Found {len(clusters)} clusters containing {total_clustered} items "
        f"({len(items) - total_clustered} noise points)"
    )

    return clusters


# =============================================================================
# Hybrid Deduplication Strategy
# =============================================================================


def hybrid_dedup(
    items: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    exact_first: bool = True,
    minhash_threshold: float = 0.9,
    semantic_threshold: float = 0.85,
    semantic_enabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a multi-stage deduplication strategy.

    Combines exact, MinHash, and (optionally) semantic deduplication in order
    of increasing cost but decreasing specificity. Returns both deduped items
    and metadata for monitoring.

    Args:
        items: Sequence of items to deduplicate.
        text_key: Key to extract text from each item.
        exact_first: Whether to run exact dedup before MinHash (recommended).
        minhash_threshold: MinHash similarity threshold.
        semantic_threshold: Semantic similarity threshold.
        semantic_enabled: Whether to run (expensive) semantic dedup.

    Returns:
        Tuple of (deduped_items, metadata_dict) with statistics.

    Examples:
        >>> items = [{"text": t} for t in ["A", "a", "A", "B"]]
        >>> deduped, meta = hybrid_dedup(items)
        >>> meta["final_count"]
        2
    """
    metadata: dict[str, Any] = {"original_count": len(items)}
    current = list(items)

    # Stage 1: Exact dedup
    if exact_first:
        current, _exact_hashes = exact_dedup(current, text_key=text_key)
        metadata["exact_removed"] = metadata["original_count"] - len(current)

    # Stage 2: MinHash dedup
    current, _minhash_sigs = minhash_dedup(current, text_key=text_key, threshold=minhash_threshold)
    removed_by_minhash = len(items) - len(current) - metadata.get("exact_removed", 0)
    metadata["minhash_removed"] = removed_by_minhash

    # Stage 3: Semantic dedup (optional, expensive)
    if semantic_enabled:
        current = semantic_dedup(current, text_key=text_key, threshold=semantic_threshold)
        removed_by_semantic = len(items) - len(current) - metadata.get("exact_removed", 0) - metadata.get("minhash_removed", 0)
        metadata["semantic_removed"] = removed_by_semantic
    else:
        metadata["semantic_removed"] = 0

    metadata["final_count"] = len(current)
    metadata["total_removed"] = metadata["original_count"] - metadata["final_count"]
    metadata["retention_rate"] = metadata["final_count"] / metadata["original_count"] if metadata["original_count"] > 0 else 0.0

    logger.info(
        f"Hybrid dedup complete: {metadata['original_count']} -> {metadata['final_count']} "
        f"({metadata['total_removed']} removed, {metadata['retention_rate']:.1%} retained)"
    )

    return current, metadata


__all__ = [
    "Document",
    "EmbeddingResult",
    "_normalize_text",
    "cluster_dedup",
    "compute_embeddings",
    "cross_stage_dedup",
    "exact_dedup",
    "hybrid_dedup",
    "minhash_dedup",
    "ngram_dedup",
    "ngram_hashes",
    "ngram_overlap",
    "semantic_dedup",
]
