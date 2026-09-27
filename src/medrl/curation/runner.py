"""Stage-graph runner: execute stages, seal + save manifests, mirror light records.

Usage (CLI lands in cli/main.py)::

    python -m medrl.curation.runner run --run-id cur-20260926 --stage 00_registry
    python -m medrl.curation.runner run --run-id cur-20260926 --stage 01_normalize

Resume rule: a stage re-runs from its input snapshot; snapshots are immutable
outputs, so re-running a stage overwrites its own output dir and manifest only.
Every successful stage mirrors its manifest + reports into experiments/curation/
for commit -- the durability invariant from the plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from medrl.curation import store
from medrl.curation.schema import StageError, StageManifest, utcnow

STAGES: dict[str, Callable[[str], StageManifest]] = {}


def stage(
    name: str,
) -> Callable[[Callable[[str], StageManifest]], Callable[[str], StageManifest]]:
    def register(fn: Callable[[str], StageManifest]) -> Callable[[str], StageManifest]:
        STAGES[name] = fn
        return fn

    return register


def _wrap(run_id: str, name: str, fn: Callable[[str], StageManifest]) -> StageManifest:
    """Run one stage fn, seal + persist its manifest, mirror, and fail loudly."""
    manifest = fn(run_id)
    manifest = store.seal_manifest(manifest)
    store.save_manifest(manifest)
    copied = store.mirror_light(run_id)
    print(
        json.dumps(
            {
                "stage": name,
                "rows_out": manifest.rows_out,
                "wall_s": manifest.wall_s,
                "mirrored": [str(p) for p in copied],
            }
        )
    )
    return manifest


def run(run_id: str, stages: list[str]) -> int:
    ok = True
    for name in stages:
        if name not in STAGES:
            print(f"unknown stage {name!r}; known: {sorted(STAGES)}", file=sys.stderr)
            return 2
        try:
            _wrap(run_id, name, STAGES[name])
        except StageError as exc:
            print(f"STAGE FAILED {name}: {exc}", file=sys.stderr)
            ok = False
            break
    return 0 if ok else 1


MIXTURE_SQL_DIR = Path("/root/medrl/configs/curation/mixtures")
MIXTURE_PHASES = ("phase1", "phase2", "phase3", "phase4", "phase5")


def run_mixture_default(
    run_id: str,
    *,
    sql_dir: Path | str = MIXTURE_SQL_DIR,
    input_dir: Path | None = None,
) -> StageManifest:
    """Execute the full curriculum phase1..phase5 in order when invoked bare.

    Phase selection is a runner-level choice. Every phase's manifest is its own
    audit record (recipe sql text + sha, per-source counts, content hashes), so
    each is sealed + saved as it completes -- returning only the last would
    leave phases 1-4 with output snapshots but no manifest, and a recipe drift
    in them would be undetectable after the fact. The last manifest is returned
    unsealed for _wrap's seal/save, exactly like a single-stage entry.
    """
    last: StageManifest | None = None
    from medrl.curation.stages import mixture as mixture_run

    for phase in MIXTURE_PHASES:
        manifest = mixture_run.run_mixture(
            run_id, sql_dir=sql_dir, out_name=phase, input_dir=input_dir
        )
        if phase != MIXTURE_PHASES[-1]:
            store.save_manifest(store.seal_manifest(manifest))
        last = manifest
    assert last is not None
    return last


def register_builtin_stages() -> None:
    """Wire stage modules into STAGES. Idempotent; called by main()."""
    from medrl.curation import registry as registry_mod
    from medrl.curation.stages import (
        answers,
        concept,
        coverage,
        decontam_ngram,
        decontam_sem,
        dedup_lex,
        difficulty,
        embed,
        judge,
        normalize,
        structural,
    )

    STAGES.setdefault("00_registry", registry_mod.run_registry)
    STAGES.setdefault("01_normalize", normalize.stage_entry)
    STAGES.setdefault("02_structural", structural.stage_entry)
    STAGES.setdefault("03_dedup", dedup_lex.stage_entry)
    STAGES.setdefault("04_decontam_ngram", decontam_ngram.stage_entry)
    STAGES.setdefault("05_embed", embed.run_embed)
    STAGES.setdefault("06_decontam_sem", decontam_sem.run_decontam_sem)
    STAGES.setdefault("08_concept", concept.stage_entry)
    STAGES.setdefault("09_answers", answers.run_answers)
    STAGES.setdefault("11_judge", judge.stage_entry)
    STAGES.setdefault("12_difficulty", difficulty.stage_entry)
    STAGES.setdefault("13_coverage", coverage.stage_entry)
    STAGES.setdefault("15_mixture", run_mixture_default)


def main(argv: list[str] | None = None) -> int:
    register_builtin_stages()

    ap = argparse.ArgumentParser(prog="medrl-curation")
    ap.add_argument("command", choices=["run", "stages"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--stage", action="append", default=[], help="repeatable; default = all")
    args = ap.parse_args(argv)

    if args.command == "stages":
        print("\n".join(sorted(STAGES)))
        return 0
    run_id = args.run_id or f"cur-{utcnow().strftime('%Y%m%d-%H%M%S')}"
    stage_names = args.stage or sorted(STAGES)
    return run(run_id, stage_names)


if __name__ == "__main__":
    raise SystemExit(main())
