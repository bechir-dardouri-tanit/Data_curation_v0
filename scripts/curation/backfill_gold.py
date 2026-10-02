#!/usr/bin/env python
"""Backfill gold answers/options onto an existing pilot snapshot.

The S1 mappers missed the real gold locations (ii_medical_rl `label`,
MedReason's Answer-Choices blob + text-gold). The mappers are fixed for future
runs; this script re-derives ONLY the answer/options/answer_type fields for the
affected sources onto the existing snapshot -- text/thinking/flags untouched,
so downstream structural/dedup/decontam flags stay valid.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/root/medrl/src")

from medrl.curation import store
from medrl.curation.schema import CorpusItem
from medrl.curation.stages import normalize


def rederive(source_id: str, record) -> dict[str, dict]:
    out: dict[str, dict] = {}
    mapper = normalize.MAPPERS[source_id]
    for file_idx, path in enumerate(normalize.iter_source_files(record)):
        for row_idx, row in enumerate(normalize._read_rows(path)):
            try:
                fresh = mapper(source_id, row)
            except Exception:
                continue
            if fresh is None:
                continue
            item_id = f"{source_id}:{file_idx}:{row_idx}"
            out[item_id] = {
                "answer": fresh.answer,
                "answer_type": fresh.answer_type,
                "meta_options": fresh.meta.get("options"),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot-b7")
    ap.add_argument("--input", default="09_answers")
    ap.add_argument("--sources", default="medreason,m23k_tokenized,ii_medical_rl")
    args = ap.parse_args()

    records = {r.source_id: r for r in store.read_registry(args.run)}
    inp = store.reset_dir(store.stage_dir(args.run, args.input + "_backfill"))
    updates: dict[str, dict] = {}
    for sid in args.sources.split(","):
        m = rederive(sid, records[sid])
        updates.update(m)
        print(f"{sid}: {len(m)} rows re-derived", flush=True)

    n_changed = 0
    buf: list[CorpusItem] = []
    for it in store.iter_items(store.stage_dir(args.run, args.input)):
        u = updates.get(it.id)
        if u:
            meta = dict(it.meta)
            if u["meta_options"] is not None:
                meta["options"] = u["meta_options"]
            new = it.model_copy(update={"answer": u["answer"], "answer_type": u["answer_type"], "meta": meta})
            if new.answer != it.answer or new.answer_type != it.answer_type:
                n_changed += 1
            buf.append(new)
        else:
            buf.append(it)
        if len(buf) >= 50_000:
            store.write_items(buf, inp)
            buf.clear()
    if buf:
        store.write_items(buf, inp)
    print(f"backfill complete: {n_changed} rows changed -> {inp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
