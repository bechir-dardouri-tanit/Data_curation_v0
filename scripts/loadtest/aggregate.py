"""Aggregate per-model sweep results into one summary table file.

Reads /scratch/medrl/loadtest/results/*.jsonl, merges the known accuracy
scores, and writes results/summary.jsonl (one row per model x cell) plus a
pivoted markdown table at results/summary.md.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("/scratch/medrl/loadtest/results")

SCORES = {  # overall 7-benchmark average, thinking mode used in the eval table
    "luth-0.8b": 23.7,
    "luth-2b": 31.9,
    "medgemma-4b": 35.5,
    "medgemma-1.5-4b": 40.0,
    "qwen35-4b": 53.2,
    "biomistral-7b": 22.9,
    "huatuogpt-o1-7b": 41.5,
    "ii-medical-8b": 46.4,
    "ii-medical-8b-1706": 51.9,
    "medreason-8b": 36.9,
    "apertus-8b": 28.6,
    "qwen35-9b": 61.6,
    "eurollm-9b": 25.8,
    "medgemma-27b": 57.7,
    "qwen38-27b": 64.7,
}

SIZES = {
    "luth-0.8b": 0.8, "luth-2b": 2, "medgemma-4b": 4, "medgemma-1.5-4b": 4,
    "qwen35-4b": 4, "biomistral-7b": 7, "huatuogpt-o1-7b": 7,
    "ii-medical-8b": 8, "ii-medical-8b-1706": 8, "medreason-8b": 8,
    "apertus-8b": 8, "qwen35-9b": 9, "eurollm-9b": 9, "medgemma-27b": 27,
    "qwen38-27b": 27,
}

LONG_PROBE_TAG = "-long"


def main() -> None:
    rows: list[dict] = []
    for f in sorted(RESULTS.glob("*.jsonl")):
        if f.name == "summary.jsonl":
            continue
        for line in f.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    with (RESULTS / "summary.jsonl").open("w") as f:
        for r in rows:
            tag = r.get("tag", "")
            base = tag[:-len(LONG_PROBE_TAG)] if tag.endswith(LONG_PROBE_TAG) else tag
            r = dict(r)
            r["size_b"] = SIZES.get(base)
            r["score"] = SCORES.get(base)
            f.write(json.dumps(r) + "\n")

    cells = ["1", "8", "32", "96", "192", "384"]
    by_model: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("fatal") or "level" not in r or "out_tok_per_s" not in r:
            continue
        tag = r["tag"].replace(LONG_PROBE_TAG, "")
        mode = r.get("mode", "chat")
        key = f"long-{mode}" if r.get("long_only") else f"{mode}-{r['level']}"
        by_model.setdefault(tag, {})[key] = r

    out_lines = []
    for mode in ("decode", "chat"):
        label = ("Steady-state decode throughput (out tok/s), ignore_eos"
                 if mode == "decode"
                 else "Realistic chat traffic (out tok/s)")
        out_lines.append(f"\n### {label}\n")
        lines = [
            "| Model | Size | Score | " + " | ".join(f"C={c}" for c in cells) +
            " | long-in @C=192 |",
            "|" + "---|" * (3 + len(cells) + 1),
        ]
        def speed(t: str, mode: str = mode) -> float:
            return by_model[t].get(f"{mode}-384", {}).get("out_tok_per_s") or 0
        for tag in sorted(by_model, key=speed, reverse=True):
            m = by_model[tag]
            size = SIZES.get(tag)
            score = SCORES.get(tag)
            score_str = f"{score:.1f}%" if score is not None else "?"
            vals = []
            for c in cells:
                v = m.get(f"{mode}-{c}", {}).get("out_tok_per_s")
                vals.append(f"{v:,.0f}" if v else "-")
            lv_in = m.get(f"long-{mode}", {}).get(
                "in_tok_per_s" if mode == "chat" else "out_tok_per_s")
            long_str = f"{lv_in:,.0f}" if lv_in else "-"
            lines.append(
                f"| {tag} | {size}B | {score_str} | " + " | ".join(vals) + f" | {long_str} |"
            )
        out_lines.extend(lines)
    text = "\n".join(out_lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
