"""Store round-trip tests: the parquet boundary must be lossless."""

from __future__ import annotations

from pathlib import Path

from medrl.curation import store
from medrl.curation.schema import CorpusItem


def _item(item_id: str, messages: list[dict[str, str]]) -> CorpusItem:
    return CorpusItem(id=item_id, source="s1", messages=messages)


def test_message_extra_keys_survive_the_round_trip(tmp_path: Path) -> None:
    """Regression: messages were a struct(role, content) column and pyarrow
    SILENTLY DROPPED every other key on the write -- an OpenAI `name` or
    `tool_call_id` vanished at the S1 snapshot, with no error anywhere. The
    column is JSON now (same treatment as tools/meta)."""
    items = [
        _item(
            "s:1",
            [
                {"role": "system", "content": "sys", "name": "instructions"},
                {"role": "user", "content": "q", "tool_call_id": "call-17"},
                {"role": "assistant", "content": "a"},
            ],
        ),
        _item("s:2", []),  # empty stays empty (None on disk -> [])
    ]
    out = tmp_path / "snap"
    out.mkdir(parents=True)  # write_items assumes the snapshot dir exists
    n = store.write_items(items, out)

    assert n == 2
    back = {it.id: it for it in store.iter_items(out)}
    assert back["s:1"].messages == items[0].messages
    assert back["s:2"].messages == []


def test_rewriting_a_snapshot_dir_requires_fresh_directory(tmp_path: Path) -> None:
    """write_items' append semantics are per-stage-run contract (chunked writes);
    the docstring pins them so a stage re-run must clear stale parts itself."""
    d = tmp_path / "snap"
    d.mkdir(parents=True)  # write_items assumes the snapshot dir exists
    store.write_items([_item("s:1", [{"role": "user", "content": "a"}])], d)
    store.write_items([_item("s:2", [{"role": "user", "content": "b"}])], d)
    assert store.count_items(d) == 2  # two calls, disjoint ids: the chunked-write case
