"""S5 -- embed once: one multilingual vector per item, reused by S6/coverage/review.

Model: Qwen3-Embedding-4B served by vLLM (``--runner pooling --max-model-len 512``).
Document side of the asymmetric protocol: corpus questions are embedded with NO
instruction; S6 puts the retrieval instruction on the benchmark-query side only
(model card: omitting the query-side instruction costs 1-5%; applying it to both
sides would skew the symmetric dedup distribution).

Storage: full-dim fp16 bytes on the item (``embedding`` column) + a usearch HNSW
index over MRL-truncated dims for ANN recall; S6 re-scores candidates at full
resolution (recall-at-truncated / precision-at-full, per the plan).

Resume: an interrupted pass continues where it stopped. The only writer of the
``embedding`` column is this stage, so partial progress lives in the previous
05 output; on entry its already-embedded ids are harvested and matched items
are pre-filled during the stream -- they are written through without an HTTP
embed, and ``already_embedded`` in the manifest notes counts them. Output is
one snapshot dir; the harvest happens BEFORE reset_dir clears it (the reset
used to delete exactly the progress the resume contract promised to keep).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow
from medrl.curation.thresholds import THRESHOLDS

EMBED_CONCURRENCY = 32
BATCH_TEXTS = 128
CHUNK_ITEMS = 50_000


def fp16_bytes(vec: list[float] | Any) -> bytes:
    """Full-dim vector -> little-endian fp16 bytes (the store format)."""
    return np.asarray(vec, dtype="<f2").tobytes()


def bytes_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f2").astype(np.float32)


EMBED_MAX_CHARS = 1000
"""Cheap first-pass cap; the TOKEN cap below is the real guarantee.

Char caps cannot bound tokens: arithmetic/URL-like text tokenizes at ~2
tokens/char (the pilot's math rows: 1000 chars -> 800+ tokens -> HTTP 400
against max-model-len 512). Token-aware truncation is the contract.
"""

EMBED_MAX_TOKENS = 480  # margin under the 512 serve budget

_tokenizer = None


def _tok():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B", local_files_only=True)
    return _tokenizer


def embed_truncate(text: str) -> str:
    """Token-budget truncation; falls back to the char cap if the tokenizer
    cannot load (offline edge) -- the 400 then surfaces with its body."""
    text = text.strip()[:EMBED_MAX_CHARS] or " "
    try:
        ids = _tok().encode(text)
        if len(ids) > EMBED_MAX_TOKENS:
            text = _tok().decode(ids[:EMBED_MAX_TOKENS], skip_special_tokens=True)
    except Exception:  # tokenizer unavailable: char cap stands
        pass
    return text


def _question_of(it: CorpusItem) -> str:
    """User content, token-budget truncated (see embed_truncate)."""
    return embed_truncate(next((m["content"] for m in it.messages if m["role"] == "user"), ""))


async def _embed_all(base_url: str, model: str, texts: list[str]) -> list[np.ndarray]:
    """Embed a list of texts in batched, semaphore-gated requests. Order-preserving."""
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    last_err: Exception | None = None
    out: list[np.ndarray | None] = [None] * len(texts)

    async with httpx.AsyncClient(timeout=120) as client:

        async def one(lo: int, batch: list[str]) -> None:
            nonlocal last_err
            async with sem:
                for attempt in range(4):
                    try:
                        r = await client.post(
                            f"{base_url}/v1/embeddings", json={"model": model, "input": batch}
                        )
                        if r.status_code in (429, 500, 502, 503):
                            raise httpx.HTTPError(f"status {r.status_code}")
                        if r.status_code != 200:
                            body = r.text[:200]
                            raise httpx.HTTPError(f"status {r.status_code}: {body}")
                        r.raise_for_status()
                        data = sorted(r.json()["data"], key=lambda d: d["index"])
                        for off, emb in enumerate(data):
                            out[lo + off] = np.asarray(emb["embedding"], dtype=np.float32)
                        return
                    except (httpx.HTTPError, KeyError) as exc:
                        last_err = exc
                        await asyncio.sleep(2**attempt)
                raise StageError(f"embedding batch failed after retries: {last_err}")

        awaits = [one(lo, texts[lo : lo + BATCH_TEXTS]) for lo in range(0, len(texts), BATCH_TEXTS)]
        await asyncio.gather(*awaits)

    if any(v is None for v in out):
        raise StageError("embedding pass left gaps -- refusing to write partial vectors")
    return out  # type: ignore[return-value]


def run_embed(
    run_id: str,
    *,
    base_url: str = "http://127.0.0.1:8101",
    model: str = "Qwen/Qwen3-Embedding-4B",
    input_stage: str = "04_decontam_ngram",
    output_stage: str = "05_embed",
    limit: int | None = None,
) -> StageManifest:
    """Embed every un-embedded item of the input snapshot; write snapshot + ANN index."""
    started = utcnow()
    inp = store.stage_dir(run_id, input_stage)
    out_dir = store.stage_dir(run_id, output_stage)
    # Harvest the previous pass's embeddings BEFORE reset_dir wipes them: those
    # part files are the only place partial progress survives, and deleting
    # them first turned "resume" into a silent full re-embed at the next run.
    embedded_prev: dict[str, bytes] = {}
    if out_dir.resolve() != Path(inp).resolve() and any(out_dir.glob("part-*.parquet")):
        for done in store.iter_items(out_dir):
            if done.embedding is not None:
                embedded_prev[done.id] = done.embedding
    out = store.reset_dir(out_dir)
    manifest = StageManifest(
        run_id=run_id,
        stage=output_stage,
        started_at=started,
        config={
            "model": model,
            "base_url": base_url,
            "dims_store": THRESHOLDS.embed_dim_store,
            "dims_index": THRESHOLDS.embed_dim_index,
        },
        thresholds={
            "semantic_dup": THRESHOLDS.semantic_dup_threshold,
            "semantic_contam": THRESHOLDS.semantic_contam_threshold,
        },
    )

    n_total = n_already = n_embedded = 0
    t0 = time.monotonic()
    dim: int | None = None

    def stream_input() -> Iterator[CorpusItem]:
        nonlocal n_total, n_already
        for i, it in enumerate(store.iter_items(inp)):
            if limit is not None and i >= limit:
                return
            n_total += 1
            if it.embedding is None and it.id in embedded_prev:
                n_already += 1
                yield it.model_copy(update={"embedding": embedded_prev[it.id]})
                continue
            if it.embedding is not None:
                n_already += 1
            yield it

    pending: list[CorpusItem] = []

    def flush(chunk: list[CorpusItem]) -> None:
        """Embed the chunk's un-embedded items, then write the whole chunk."""
        nonlocal dim, n_embedded, pending
        if not chunk:
            return
        missing = [x for x in chunk if x.embedding is None]
        if missing:
            vecs = asyncio.run(_embed_all(base_url, model, [_question_of(x) for x in missing]))
            dim = dim or int(vecs[0].shape[0])
            n_embedded += len(missing)
            got = {x.id: fp16_bytes(v) for x, v in zip(missing, vecs, strict=True)}
            chunk = [
                x if x.embedding is not None else x.model_copy(update={"embedding": got[x.id]})
                for x in chunk
            ]
        store.write_items(chunk, out)
        pending = []

    it: CorpusItem | None
    for it in stream_input():
        pending.append(it)
        if len(pending) >= CHUNK_ITEMS:
            flush(pending)
    flush(pending)

    # a fully-resumed pass embeds nothing, so the dim never got observed this
    # run -- take it from the harvested blobs (the index must not silently
    # build at the THRESHOLDS default and disagree with the stored vectors)
    if dim is None and embedded_prev:
        dim = int(bytes_to_vec(next(iter(embedded_prev.values()))).shape[0])

    index_path = _build_usearch_index(out, dim_hint=dim)
    manifest.rows_in = n_total
    manifest.rows_out = store.count_items(out)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.notes = {
        "embedded": n_embedded,
        "already_embedded": n_already,
        "wall_s": round(time.monotonic() - t0, 1),
        "ann_index": str(index_path),
        "dim": dim,
    }
    if manifest.rows_in != manifest.rows_out:
        raise StageError(
            f"S5: rows_in={manifest.rows_in} != rows_out={manifest.rows_out} "
            "-- flags-not-deletes: S5 never drops"
        )
    return manifest


def _build_usearch_index(snapshot_dir: Path, dim_hint: int | None) -> Path:
    """HNSW over MRL-truncated dims; an id-map JSON sits beside the index."""
    from usearch.index import Index

    index = Index(
        ndim=(dim_hint and min(dim_hint, THRESHOLDS.embed_dim_index)) or THRESHOLDS.embed_dim_index,
        metric="cos",
        dtype="f16",
    )
    id_map: dict[str, str] = {}
    for key, it in enumerate(store.iter_items(snapshot_dir)):
        if it.embedding is None:
            continue
        v = bytes_to_vec(it.embedding)[: index.ndim]
        index.add(key, v)
        id_map[str(key)] = it.id
    index_path = snapshot_dir / "ann.usearch"
    index.save(str(index_path))
    (snapshot_dir / "ann_ids.json").write_text(json.dumps(id_map, sort_keys=True))
    return index_path


__all__ = ["bytes_to_vec", "fp16_bytes", "run_embed"]
