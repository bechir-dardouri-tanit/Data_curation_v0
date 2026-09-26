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

from medrl.curation import store
from medrl.curation.schema import StageError, StageManifest, utcnow

STAGES: dict[str, Callable[[str], StageManifest]] = {}


def stage(name: str) -> Callable:
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
    print(json.dumps({
        "stage": name, "rows_out": manifest.rows_out, "wall_s": manifest.wall_s,
        "mirrored": [str(p) for p in copied],
    }))
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


def register_builtin_stages() -> None:
    """Wire stage modules into STAGES. Idempotent; called by main()."""
    from medrl.curation import registry as registry_mod
    from medrl.curation.stages import normalize

    STAGES.setdefault("00_registry", registry_mod.run_registry)
    STAGES.setdefault("01_normalize", normalize.stage_entry)


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
