"""Build the load-test prompt pool from real benchmark items.

Pulls MedQA (USMLE vignettes) and HealthBench-Hard (open patient/clinician
questions) through the repo's own loaders, then composes a pool with three
input-length buckets so the sweep can measure how request size interacts
with concurrency:

  short  (~150-400 tok)  one MedQA vignette, as rendered by the eval stack
  medium (~700-1100 tok) three concatenated distinct vignettes
  long   (~1900-2600 tok) eight concatenated distinct vignettes
  open   (HealthBench)    free-form clinical questions for chat-style length diversity

Output: JSON {buckets: {name: [{messages, prompt}]}, meta}.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/root/medrl/src")

from medrl.eval.loaders import load_items
from medrl.eval.tasks.spec import TaskSpec


def user_text(item) -> str:
    return next(m["content"] for m in item.messages if m["role"] == "user")


def messages_of(text: str):
    return [{"role": "user", "content": text}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/medrl/loadtest/prompts/pool.json")
    ap.add_argument("--n-medqa", type=int, default=400)
    ap.add_argument("--n-healthbench", type=int, default=140)
    args = ap.parse_args()

    medqa_spec = TaskSpec(
        name="medqa", loader="medqa", grade="decision", language="en",
        hf_id="openlifescienceai/medqa", split="test", subset=None,
        limit=args.n_medqa, requires_judge=False,
        prompt_style="mcqa_letter", verify_style="letter",
    )
    hb_spec = TaskSpec(
        name="healthbench_hard", loader="healthbench", grade="decision", language="en",
        hf_id="openai/healthbench", split="test", subset="hard",
        limit=args.n_healthbench, requires_judge=True,
        prompt_style="open_rubric", verify_style="rubric",
    )

    medqa_items = [user_text(it) for it in load_items(medqa_spec).items]
    hb_items = [user_text(it) for it in load_items(hb_spec).items]
    print(f"pulled {len(medqa_items)} medqa + {len(hb_items)} healthbench prompts")

    rng = random.Random(7)
    rng.shuffle(medqa_items)

    def stitch(texts: list[str]) -> str:
        body = "\n\n---\n\n".join(t.strip() for t in texts)
        # Re-issue the final vignette as the active question so generations stay
        # realistic in length rather than answering every stitched case.
        return body + "\n\n---\n\nAnswer ONLY the final case shown above:\n\n" + texts[-1].strip()

    n_short, n_med_items = 80, 240
    shorts = medqa_items[:n_short]
    med_start = n_short
    med_end = med_start + n_med_items
    mediums = [stitch(medqa_items[i:i + 3]) for i in range(med_start, min(med_end, len(medqa_items)), 3)]
    longs = [stitch(medqa_items[i:i + 8]) for i in range(med_end, len(medqa_items), 8)]

    pool = {
        "short": [{"messages": messages_of(t), "prompt": t} for t in shorts],
        "medium": [{"messages": messages_of(t), "prompt": t} for t in mediums],
        "long": [{"messages": messages_of(t), "prompt": t} for t in longs],
        "open": [{"messages": messages_of(t), "prompt": t} for t in hb_items],
    }
    sizes = {k: [round(len(e["prompt"]) / 3.8) for e in v] for k, v in pool.items()}
    stats = {k: {"count": len(s), "est_tok_mean": round(sum(s) / max(len(s), 1))} for k, s in sizes.items()}
    out = {"pool": pool, "meta": {"stats": stats}}

    with Path(args.out).open("w") as f:
        json.dump(out, f)
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
