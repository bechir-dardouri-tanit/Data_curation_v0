"""S5 embed tests: the resume contract (partial progress survives an interrupt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem
from medrl.curation.stages import embed


def _item(item_id: str) -> CorpusItem:
    return CorpusItem(
        id=item_id,
        source="s1",
        messages=[{"role": "user", "content": f"clinical question {item_id}"}],
        answer_type="none",
    )


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(store, "stage_dir", lambda run_id, name: tmp_path / name)
    return tmp_path


def test_interrupted_pass_resumes_without_reembedding(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: reset_dir wiped the previous 05 output before the pass looked
    at anything, and the pass-through branch only fired on embeddings the INPUT
    carried (the input stage never does) -- so an interrupt at 2.4M of 2.5M
    items silently re-embedded the whole corpus on the next invocation. The
    harvest of the previous output must make the second pass free."""
    inp = scratch / "04_decontam_ngram"
    inp.mkdir(parents=True)
    store.write_items([_item(f"s:{i}") for i in range(10)], inp)

    calls: list[int] = []

    async def fake_embed_all(base_url: str, model: str, texts: list[str]) -> list[np.ndarray]:
        calls.append(len(texts))
        return [np.full(8, float(i)) for i in range(len(texts))]

    monkeypatch.setattr(embed, "_embed_all", fake_embed_all)

    first = embed.run_embed("run-e", base_url="http://x", model="m")
    assert first.notes["embedded"] == 10 and first.notes["already_embedded"] == 0
    assert calls == [10]

    # second pass: every id was embedded by the first, so no HTTP at all
    second = embed.run_embed("run-e", base_url="http://x", model="m")
    assert calls == [10], "the resumed pass must not re-embed recovered ids"
    assert second.notes["embedded"] == 0
    assert second.notes["already_embedded"] == 10
    assert second.rows_out == 10
    rows = {it.id: it for it in store.iter_items(scratch / "05_embed")}
    assert all(it.embedding is not None for it in rows.values())


def test_resume_fills_only_missing_items(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A snapshot partially written before the crash: harvested ids skip the
    HTTP pass, unseen ids go through the normal embed+chunk path (no per-item
    part files)."""
    inp = scratch / "04_decontam_ngram"
    inp.mkdir(parents=True)
    store.write_items([_item(f"s:{i}") for i in range(6)], inp)

    # a "previous output" containing embeddings for s:0..s:2 only
    prev = scratch / "05_embed"
    prev.mkdir(parents=True)
    store.write_items(
        [
            it.model_copy(update={"embedding": embed.fp16_bytes(np.full(8, 1.0))})
            for it in [_item(f"s:{i}") for i in range(3)]
        ],
        prev,
    )

    calls: list[int] = []

    async def fake_embed_all(base_url: str, model: str, texts: list[str]) -> list[np.ndarray]:
        calls.append(len(texts))
        return [np.full(8, 2.0) for _ in texts]

    monkeypatch.setattr(embed, "_embed_all", fake_embed_all)

    manifest = embed.run_embed("run-e", base_url="http://x", model="m")
    assert calls == [3], "only the un-embedded remainder goes to the server"
    assert manifest.notes["already_embedded"] == 3
    assert manifest.notes["embedded"] == 3
    assert len(list(prev.glob("part-*.parquet"))) == 1  # one flush, not one file per row
    rows = {it.id: it for it in store.iter_items(prev)}
    assert rows["s:0"].embedding == embed.fp16_bytes(np.full(8, 1.0))  # harvested, not re-billed
    assert rows["s:5"].embedding == embed.fp16_bytes(np.full(8, 2.0))
