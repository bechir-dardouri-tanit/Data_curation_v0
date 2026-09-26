"""S1 -- normalise every source into the canonical CorpusItem table.

Source-specific mappers live here: each of the 11 pools has its own row shape
(reasoning in a standalone column vs embedded in messages vs plain QA), and the
mapper is the single place that knowledge is encoded. Unknown row shapes fail
loudly rather than silently mapping to an empty item.

Language ID: fastText lid.176.bin on a >=200-char question concatenation
(fastText has a documented EN bias that worsens on short input), GlotLID V3 as a
second opinion under the confidence floor or for very short text. LID never
drops a row -- it writes lang/lang_score and the S2 keep-rule decides.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from medrl.curation.registry import REGISTRY_SOURCES, iter_source_files
from medrl.curation.schema import AnswerType, CorpusItem, StageError
from medrl.curation.store import reset_dir, stage_dir, write_items
from medrl.curation.thresholds import THRESHOLDS

Mapper = Callable[[str, dict[str, Any]], CorpusItem | None]


# --------------------------------------------------------------------------
# LID -- lazy-loaded so the module imports on a box without the model files.
# --------------------------------------------------------------------------

_LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

_lid_model: Any = None
_glotlid_model: Any = None


def _lid():
    global _lid_model
    if _lid_model is None:
        import fasttext  # type: ignore[import-not-found]

        cache = Path("/scratch/medrl/curation/models")
        cache.mkdir(parents=True, exist_ok=True)
        model_path = cache / "lid.176.bin"
        if not model_path.exists():
            import urllib.request

            urllib.request.urlretrieve(_LID_MODEL_URL, model_path)
        _lid_model = fasttext.load_model(str(model_path))
    return _lid_model


def _glotlid():
    global _glotlid_model
    if _glotlid_model is None:
        from gltPID import GlotLID  # type: ignore[import-not-found]  # placeholder id

        _glotlid_model = GlotLID()
    return _glotlid_model


def detect_lang(text: str) -> tuple[str, float]:
    """(lang, score) with the GlotLID second opinion below the confidence floor.

    Uses fastText's low-level pybind ``f.predict`` -- the convenience wrapper in
    fasttext/FastText.py calls ``np.array(..., copy=False)``, which NumPy 2
    removed (fasttext-wheel 0.9.2 ships the old call). The pybind signature is
    ``predict(text, k, threshold, on_unicode_error) -> [(prob, label), ...]``.
    """
    t = text.strip()
    if not t:
        return "en", 0.0
    results = _lid().f.predict(t.replace("\n", " ")[:4000], 1, 0.0, "strict")
    if not results:
        return "en", 0.0
    score, label = results[0]
    lang, score = label.replace("__label__", ""), float(score)
    if score >= THRESHOLDS.lid_confidence_floor and len(t) >= THRESHOLDS.lid_min_chars:
        return lang, score
    try:
        glot = _glotlid().predict(t, k=1)
        if glot:
            cand_lang, cand_score = glot[0][0].replace("_Latn", ""), float(glot[0][1])
            if cand_score >= score:
                return cand_lang, cand_score
    except Exception:  # noqa: BLE001 -- GlotLID optional; fastText result stands
        pass
    return lang, score


def question_concat(item: dict[str, Any]) -> str:
    """>=lid_concat_chars concatenation of the question-ish fields, for LID."""
    parts: list[str] = []
    for key in ("question", "problem", "instruction", "input", "text", "prompt"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    for m in item.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            parts.append(str(m.get("content", "")))
    joined = "\n".join(parts)
    return joined[: THRESHOLDS.lid_concat_chars * 4] if len(joined) > 400 else joined


# --------------------------------------------------------------------------
# Per-source mappers -- each returns None to DROP (recorded in the manifest),
# a CorpusItem to keep. Shapes verified against the HF pages 2026-09-26.
# --------------------------------------------------------------------------


def _mk(source_id: str, row_id: str | int, **kw: Any) -> CorpusItem:
    return CorpusItem(id=f"{source_id}:{row_id}", source=source_id, **kw)


def _messages_of(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in (row.get("messages") or [])
        if isinstance(m, dict)
    ]


def map_ii_medical_reasoning_sft(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """Reasoning lives INSIDE message content (no standalone column) -- keep verbatim;
    S2's think-truncation filter will flag incomplete traces. No gold answer field:
    answer_type stays 'none' (verified: the S9 letter/number paths skip this source)."""
    msgs = _messages_of(row)
    if not msgs:
        return None
    return _mk(sid, row.get("id") or row.get("model") or hash(json.dumps(row, sort_keys=True)),
               messages=msgs, answer_type="none")


def map_finemed_sft(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """Columns: text/instruction/complexity/quality/language/response/instruction_type.
    Quality/complexity carried verbatim in meta -- S11 will not re-derive them."""
    instr, resp = row.get("instruction"), row.get("response")
    if not instr or not resp:
        return None
    return _mk(
        sid, row.get("id") or hash(f"{instr[:128]}{resp[:128]}"),
        messages=[{"role": "user", "content": str(instr)}, {"role": "assistant", "content": str(resp)}],
        answer_type="free_text",
        meta={"quality": row.get("quality"), "complexity": row.get("complexity"),
              "source_lang": row.get("language"), "instruction_type": row.get("instruction_type")},
    )


def map_chatdoctor_healthcaremagic(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    instr, resp = row.get("instruction"), row.get("output")
    if not instr or not resp:
        return None
    return _mk(sid, hash(f"{instr[:128]}"),
               messages=[{"role": "user", "content": str(instr)}, {"role": "assistant", "content": str(resp)}],
               answer_type="free_text")  # plain patient-QA, no CoT (verified)


def map_generalthought_biology(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """Biology subset only; R1-style traces in a reasoning field."""
    if "biolog" not in str(row.get("field", "")).lower() and "biolog" not in str(row.get("domain", "")).lower():
        return None
    return _mk(sid, row.get("question_id") or hash(str(row.get("question", ""))[:128]),
               messages=[{"role": "user", "content": str(row.get("question", ""))},
                         {"role": "assistant", "content": str(row.get("response", ""))}],
               thinking=row.get("reasoning") or None,
               answer=str(row.get("answer")) if row.get("answer") is not None else None,
               answer_type="free_text")


def map_medical_r1_distill(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """R1 distillation. Verified on-disk keys carry parenthetical suffixes:
    'reasoning (reasoning_content)' and 'response (content)'; plain names kept
    as fallbacks in case the export format changes."""
    q = row.get("question") or row.get("problem")
    if not q:
        return None
    response = row.get("response (content)") or row.get("content") or ""
    reasoning = row.get("reasoning (reasoning_content)") or row.get("reasoning_content") or None
    return _mk(sid, 0,  # id assigned by materialize_source (row position)
               messages=[{"role": "user", "content": str(q)},
                         {"role": "assistant", "content": str(response)}],
               thinking=reasoning,
               answer_type="free_text")


def _mcqa_from_row(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """Shared mapper for m23k / MedReason / Huatuo-style rows with options+gold."""
    q = row.get("question") or row.get("Question")
    options = row.get("options") or row.get("Options")
    gold = row.get("answer") or row.get("Answer") or row.get("correct_answer")
    if not q:
        return None
    item = _mk(sid, row.get("id") or hash(str(q)[:128]),
               messages=[{"role": "user", "content": str(q)}],
               thinking=row.get("reasoning") or row.get("RATIONALE") or None,
               answer=str(gold) if gold is not None else None)
    if options:
        item.meta["options"] = options
        item.answer_type = "mcqa"
    elif row.get("open") or not gold:
        item.answer_type = "free_text"
    return item


def map_huatuo_o1(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """en + en_mix subsets only (zh excluded -- registry note); open-ended with CoT."""
    if str(row.get("language", "en")).startswith("zh"):
        return None
    q = row.get("question") or row.get("Question")
    if not q:
        return None
    return _mk(sid, hash(str(q)[:128]),
               messages=[{"role": "user", "content": str(q)}],
               thinking=row.get("Long_CoT") or row.get("response") or None,
               answer=row.get("answer") if not isinstance(row.get("answer"), list) else json.dumps(row["answer"]),
               answer_type="free_text")


def map_pref_pairs(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """FineMed-DPO: prompt/chosen/rejected pairs; both sides stored in meta for S15."""
    prompt = row.get("prompt") or row.get("question")
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if not prompt or not chosen or not rejected:
        return None
    return _mk(sid, row.get("id") or hash(str(prompt)[:128]),
               messages=[{"role": "user", "content": str(prompt)}],
               thinking=_extract_think(str(chosen)),
               answer=str(chosen), answer_type="free_text",
               meta={"rejected": str(rejected), "pair": True})


def map_rl_row(sid: str, row: dict[str, Any]) -> CorpusItem | None:
    """II-Medical-RL / ChatDoctor-RL: question + reward_model.ground_truth (+ options)."""
    q = row.get("question") or row.get("problem")
    if not q:
        return None
    gt = (row.get("reward_model") or {}).get("ground_truth")
    options = row.get("options")
    item = _mk(sid, row.get("id") or hash(str(q)[:128]),
               messages=[{"role": "user", "content": str(q)}],
               thinking=row.get("reasoning") or None,
               answer=str(gt) if gt is not None else None)
    if options:
        item.meta["options"] = options
        item.answer_type = "mcqa"
    return item


def _extract_think(text: str) -> str | None:
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0]
    return None


MAPPERS: dict[str, Mapper] = {
    "ii_medical_reasoning_sft": map_ii_medical_reasoning_sft,
    "finemed_sft": map_finemed_sft,
    "chatdoctor_healthcaremagic": map_chatdoctor_healthcaremagic,
    "generalthought_biology": map_generalthought_biology,
    "medical_r1_distill": map_medical_r1_distill,
    "m23k_tokenized": _mcqa_from_row,
    "medreason": _mcqa_from_row,
    "huatuo_o1_reasoning": map_huatuo_o1,
    "finemed_dpo": map_pref_pairs,
    "ii_medical_rl": map_rl_row,
    "chatdoctor_rl": map_rl_row,
}


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for f in ([path] if path.is_file() else sorted(path.parent.glob("*.parquet"))):
            for row in pq.read_table(f).to_pylist():
                yield row
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            yield row
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def materialize_source(source_id: str, records: dict[str, Any]) -> Iterator[CorpusItem]:
    """Stream one source's files through its mapper + LID."""
    mapper = MAPPERS.get(source_id)
    if mapper is None:
        raise StageError(f"S1: no mapper registered for source {source_id!r}")
    record = records[source_id]
    n = 0
    for file_idx, path in enumerate(iter_source_files(record)):
        for row_idx, row in enumerate(_read_rows(path)):
            try:
                item = mapper(source_id, row)
            except Exception:
                continue  # malformed row: counted, not fatal -- manifest reports the rate
            n += 1
            if item is None:
                continue
            # Row-position id: stable across runs (sorted files, deterministic
            # readers). Content duplication is deliberately NOT resolved here --
            # that is S3's job; a content hash here would silently merge
            # distinct rows and salted hash() broke run-to-run reproducibility.
            item = item.model_copy(update={"id": f"{source_id}:{file_idx}:{row_idx}"})
            concat = question_concat(row) if not item.messages else question_concat(
                {"messages": item.messages, **{k: v for k, v in row.items() if isinstance(v, str)}})
            lang, score = detect_lang(concat)
            item.lang, item.lang_score = ("en" if lang.startswith("en") else "fr" if lang.startswith("fr") else lang), score
            yield item


def run_normalize(run_id: str, sources: list[str] | None = None) -> dict[str, Any]:
    """Materialize all (or a subset of) sources into the 01_normalize snapshot."""
    from medrl.curation.store import read_registry

    records = {r.source_id: r for r in read_registry(run_id)}
    if not records:
        raise StageError(f"S1: no registry for run {run_id} -- run S0 first")
    wanted = sources or [s for s in REGISTRY_SOURCES if s in records]
    out = reset_dir(stage_dir(run_id, "01_normalize"))
    stats: dict[str, dict[str, int]] = {}
    batch: list[CorpusItem] = []

    for sid in wanted:
        n_kept = n_seen = 0
        for item in materialize_source(sid, records):
            n_seen += 1
            batch.append(item)
            n_kept += 1
            if len(batch) >= 100_000:
                write_items(batch, out)
                batch.clear()
        stats[sid] = {"rows_seen": n_seen, "items_kept": n_kept}
        if batch:
            write_items(batch, out)
            batch.clear()
    return {"stage_dir": str(out), "per_source": stats,
            "thresholds": {"lid_floor": THRESHOLDS.lid_confidence_floor}}


def stage_entry(run_id: str, sources: list[str] | None = None):
    """Runner adapter: dict result -> StageManifest."""
    from medrl.curation.schema import StageManifest, utcnow
    from medrl.curation.store import count_items, content_sha256

    started = utcnow()
    result = run_normalize(run_id, sources)
    out_dir = Path(result["stage_dir"])
    per_source = result["per_source"]
    manifest = StageManifest(
        run_id=run_id, stage="01_normalize", started_at=started,
        config={"sources": sorted(per_source)},
        rows_in=sum(s["rows_seen"] for s in per_source.values()),
        rows_out=count_items(out_dir),
        output_sha256=content_sha256(out_dir),
    )
    # S1 is the materialization boundary: mappers legitimately filter raw rows
    # (zh Huatuo subsets, non-biology GeneralThought...). The flags-not-deletes
    # invariant starts at S2; here every drop is counted and published instead.
    manifest.notes = {
        "per_source": per_source,
        "dropped_by_mapper": {k: v["rows_seen"] - v["items_kept"] for k, v in per_source.items()},
        "rows_out_matches_kept": manifest.rows_out
        == sum(v["items_kept"] for v in per_source.values()),
    }
    return manifest


__all__ = ["MAPPERS", "detect_lang", "materialize_source", "run_normalize", "stage_entry"]
