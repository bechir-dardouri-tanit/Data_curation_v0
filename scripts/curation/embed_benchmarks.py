#!/usr/bin/env python
"""Embed the eval-benchmark question texts for S6's semantic decontamination.

Produces the two artifacts ``run_decontam_sem(bench_vectors_path=...)`` consumes:

    <out>.json  {"<benchmark>": [{"id": "<bench>::<item_id>", "key": <row>}, ...]}
    <out>.npz   {"<row>": <vector>}  (float32, full store dims)

Reuses the S4 question-only extraction (build_question_index) so the semantic
index covers EXACTLY the texts the n-gram index sees -- the two contamination
checks can never drift apart -- and the S5 embedding path (_embed_all) so the
vectors live in the same space as the corpus side.

Run AFTER the embedding server is up:

    python scripts/curation/embed_benchmarks.py \
        --base-url http://127.0.0.1:8101 \
        --out /scratch/medrl/curation/pilot-b7/bench_vectors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/medrl/src")

from medrl.curation.stages.decontam_ngram import build_question_index
from medrl.curation.stages.embed import _embed_all

DEFAULT_BENCHES = [
    "medqa", "medmcqa", "mmlu_pro_health", "medxpertqa_text", "mediqal",
    "healthbench_hard", "medcalc", "mmlu_pro", "healthbench", "ifeval",
]


def build_benchmark_vectors(
    base_url: str,
    model: str,
    out_path: str | Path,
    benchmarks: list[str] | None = None,
) -> dict[str, int]:
    """Embed benchmark questions; write the JSON+npz pair; return per-bench counts."""
    index = build_question_index(benchmarks or DEFAULT_BENCHES)

    # Group item keys by benchmark, keep the index's own question texts.
    per_bench: dict[str, list[str]] = {}
    texts: list[str] = []
    keys: list[str] = []
    for key, item in sorted(index.items.items()):
        per_bench.setdefault(item.benchmark, []).append(key)
        texts.append(item.text)
        keys.append(key)

    texts = [t.strip()[:1800] or " " for t in texts]  # same budget as embed.py
    vectors = _embed_all(base_url, model, texts)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON references rows by index; npz holds the vectors positionally.
    row = 0
    payload: dict[str, list[dict]] = {}
    for bench in sorted(per_bench):
        payload[bench] = [{"id": k, "key": row + i} for i, k in enumerate(per_bench[bench])]
        row += len(per_bench[bench])
    out.with_suffix(".json").write_text(json.dumps(payload))
    np.savez_compressed(out.with_suffix(".npz"), **{str(i): v for i, v in enumerate(vectors)})
    return {b: len(ks) for b, ks in per_bench.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8101")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--out", required=True, help="output path stem (.json/.npz appended)")
    args = ap.parse_args()

    counts = build_benchmark_vectors(args.base_url, args.model, args.out)
    print(json.dumps({"embedded": counts, "out": str(args.out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
