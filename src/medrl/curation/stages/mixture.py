"""S15 -- mixture assembly: the corpus's last stage turns filters into queries.

Everything upstream is flags-not-deletes, so a training mixture is nothing but
a deterministic query over a snapshot. The recipes live as SQL files
(``configs/curation/mixtures/phase1.sql`` ... ``phase5.sql``); this module only
executes one of them against a run snapshot and materializes three artifacts:

* the composed subset as a new snapshot dir ``15_mixture_<phase>`` -- unique
  ids, because a snapshot is a *set* and the re-decontamination stages that
  consume it iterate/write sets;
* the selected instances as an experiments-side JSONL -- one line per selected
  instance, sampling-with-replacement expansions included, in mixture order.
  This file, not the snapshot, is what the training-split builder reads;
* ``mixture_spec.sql`` -- the exact executed SQL prefixed with a per-source
  row-count header, so the shipped recipe and the realized assembly are one
  document (per-phase texts also survive in each manifest's config.sql_text).

Recombination reintroduces contamination (near-duplicates and benchmark leaks
compose badly), so the manifest's notes carry ``recheck_required``: the RUNNER
re-runs S4/S6/S8 over the composed subset. This stage never invokes sibling
stages itself -- orchestration belongs to the runner.

Deliberate contract deviations, recorded here because every flagging stage
asserts the opposite:

* ``rows_out != rows_in`` -- S15 is a query, not a flagger. ``rows_out`` counts
  the unique rows of the new snapshot; the instance count (replacement
  expansions included) is ``notes["instances_selected"]``.
* ``flag_rates`` stays empty -- S15 sets no flags.

DuckDB notes (pinned 1.5.5, verified on this engine): the function is
``random()`` (there is no ``rand()``), and session seeding is
``SELECT setseed(x)`` -- the ``SET seed`` configuration parameter does not
exist on this version, so the recipes spell the mixture contract's documented
``SET seed=42`` as ``setseed(0.42)`` (seed 42 in setseed's [-1, 1] domain).
Parameterless macros cannot reference columns on this engine, so the survivor
predicate ships as a 17-argument macro -- one spelling at the top of every
recipe, with the ``flags_*`` columns passed at each call site.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from medrl.core.logging import get_logger
from medrl.curation import store
from medrl.curation.schema import CorpusItem, StageError, StageManifest, utcnow

log = get_logger(__name__)

STAGE_PREFIX = "15_mixture"
"""Snapshot-dir / manifest prefix; the phase name completes it (15_mixture_phase1)."""

OUT_NAME_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
"""A phase name selects a recipe, it is never a path: no separators, no traversal."""

SURVIVOR_CALL = (
    "survivor(flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang, "
    "flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash, flags_dup_semantic, "
    "flags_dup_concept, flags_contam_ngram, flags_contam_semantic, flags_contam_concept, "
    "flags_answer_wrong, flags_answer_right_reasoning_contradicts, flags_kg_contradicted)"
)
"""The survivor macro's call site, mirroring the recipes: this module re-uses the
macro each recipe defines to count survivors for the spec header."""

RECHECK_REQUIRED = ("04_decontam_ngram", "06_decontam_sem", "08_concept")
"""Stages the runner must re-run over the composed subset (S15 never calls them)."""

INPUT_STAGE_CANDIDATES = (
    "13_coverage",
    "12_difficulty",
    "11_judge",
    "09_answers",
    "06_decontam_sem",
    "05_embed",
    "04_decontam_ngram",
    "03_dedup",
    "02_structural",
    "01_normalize",
)
"""Newest-first: with no explicit input, the mixture runs over the newest
snapshot the run actually has. Upstream stages 07/11/12/13 are not built yet,
so a hardcoded default would name a directory that cannot exist; the resolved
choice is recorded in manifest.config.input_stage for the audit trail."""

_MIN_DUCKDB = (0, 10)
"""The mixture contract's own floor ("SET seed requires duckdb>=0.10"); the exact
spellings used here (setseed/random) were verified on the pinned 1.5.5."""

_WRITE_BUFFER = 50_000
"""Rows buffered between append-safe write_items calls (answers.py precedent)."""


def _assert_duckdb_support() -> None:
    """Refuse engines older than the session-seed era instead of silently
    unseeding the assembly: every recipe opens with setseed(), and a duckdb
    without it would either error mid-file or, worse, run an older spelling."""
    parts = re.findall(r"\d+", duckdb.__version__)
    version = tuple(int(p) for p in parts[:3]) if parts else (0,)
    if version < _MIN_DUCKDB:
        raise StageError(
            f"S15: duckdb>={_MIN_DUCKDB[0]}.{_MIN_DUCKDB[1]} required for seeded mixture "
            f"SQL; found {duckdb.__version__}"
        )


def _resolve_input(
    run_id: str,
    input_stage: str | None,
    input_dir: Path | str | None,
) -> tuple[Path, str]:
    """(snapshot dir, label) from an explicit dir, a named stage, or newest-first."""
    if input_dir is not None:
        d = Path(input_dir)
        if not d.is_dir():
            raise StageError(f"S15: input snapshot {d} does not exist")
        return d, d.name
    run_root = store.run_dir(run_id)
    if input_stage is not None:
        d = run_root / input_stage
        if not d.is_dir():
            raise StageError(f"S15: input snapshot {d} does not exist -- run {input_stage} first")
        return d, input_stage
    for cand in INPUT_STAGE_CANDIDATES:
        d = run_root / cand
        if d.is_dir() and any(d.glob("part-*.parquet")):
            return d, cand
    raise StageError(
        f"S15: no pipeline snapshot under {run_root} to mix "
        f"(looked for: {', '.join(INPUT_STAGE_CANDIDATES)})"
    )


def _sql_file_list(files: list[Path]) -> str:
    """DuckDB list literal of quoted parquet paths (single quotes doubled)."""
    return "[" + ", ".join("'" + str(f).replace("'", "''") + "'" for f in files) + "]"


def _execute_spec(
    files: list[Path], script: str
) -> tuple[list[str], list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Run the recipe against the snapshot; return (columns, selected instances,
    per-source input counts, per-source survivor counts).

    The view is named ``corpus`` so recipes never hardcode file paths -- they
    stay executable against any run's snapshot. Survivor counts reuse the
    recipe's own ``survivor`` macro; a recipe without one simply omits the
    column from the spec header rather than failing the assembly.
    """
    con = duckdb.connect()
    try:
        # One thread, deliberately: setseed() reproducibility is per-connection
        # AND per-scan. Under parallel scans the same seeded recipe over the
        # same 2M-row parquet produced a different selection on every fresh
        # connection (6/6 distinct on duckdb 1.5.5, default threads); with
        # threads=1 every run is identical. A mixture is a deterministic query
        # over a snapshot -- the training sets must not reshuffle between runs,
        # and these group-bys can afford a single core.
        con.execute("SET threads TO 1")
        con.execute(
            "CREATE OR REPLACE VIEW corpus AS "
            f"SELECT * FROM read_parquet({_sql_file_list(files)}, union_by_name = true)"
        )
        cur = con.execute(script)
        cols = [str(d[0]) for d in (cur.description or [])]
        instances = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
        input_counts = {
            str(s): int(n)
            for s, n in con.execute(
                "SELECT source, count(*) FROM corpus GROUP BY source ORDER BY source"
            ).fetchall()
        }
        try:
            survivors = {
                str(s): int(n)
                for s, n in con.execute(
                    f"SELECT source, count(*) FROM corpus WHERE {SURVIVOR_CALL} "
                    "GROUP BY source ORDER BY source"
                ).fetchall()
            }
        except duckdb.Error:
            survivors = {}
    finally:
        con.close()
    return cols, instances, input_counts, survivors


def _validate_selection(
    sql_name: str, cols: list[str], instances: list[dict[str, Any]]
) -> list[str]:
    """Unique selected ids in mixture order; fail loud on a malformed recipe.

    A recipe's final SELECT must expose ``id`` (source is expected too). An
    empty selection is a failed assembly, not an empty success: sealing a
    zero-row mixture would silently replace a training phase with nothing.
    """
    if "id" not in cols:
        raise StageError(
            f"S15: {sql_name}'s final SELECT must expose an 'id' column (got: {cols or 'no result set'})"
        )
    ids: list[str] = []
    for row in instances:
        rid = row["id"]
        if not isinstance(rid, str) or not rid:
            raise StageError(
                f"S15: {sql_name} selected a NULL/non-string id -- select corpus.id verbatim"
            )
        ids.append(rid)
    if not ids:
        raise StageError(f"S15: {sql_name} selected 0 rows -- refusing to seal an empty mixture")
    return list(dict.fromkeys(ids))


def _compose_subset(inp: Path, out: Path, unique_ids: list[str]) -> int:
    """Stream the input snapshot once; write matching rows to the output.

    Chunked write_items calls are safe here (append-safe, per-call unique ids):
    the input snapshot has unique ids by construction. Returns matched rows so
    the caller can verify the SQL's selection and the parquet agree exactly.
    """
    wanted = set(unique_ids)
    matched = 0
    buf: list[CorpusItem] = []
    for item in store.iter_items(inp):
        if item.id not in wanted:
            continue
        matched += 1
        buf.append(item)
        if len(buf) >= _WRITE_BUFFER:
            store.write_items(buf, out)
            buf.clear()
    if buf:
        store.write_items(buf, out)
    return matched


def _per_source(
    cols: list[str], instances: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, int]]:
    """(instances, unique) tallies per source; empty unless the recipe exposed
    ``source``. The unique tally counts each id's first instance only -- an
    id sampled with replacement stays one physical row of its source."""
    if "source" not in cols:
        return {}, {}
    inst = Counter(str(r["source"]) for r in instances if r.get("source") is not None)
    seen: set[str] = set()
    uniq: Counter[str] = Counter()
    for r in instances:
        src = r.get("source")
        if src is not None and r["id"] not in seen:
            seen.add(r["id"])
            uniq[str(src)] += 1
    return dict(inst), dict(uniq)


def run_mixture(
    run_id: str,
    sql_dir: Path | str,
    out_name: str,
    *,
    input_stage: str | None = None,
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    experiments_dir: Path | str | None = None,
) -> StageManifest:
    """Execute one mixture recipe over a run snapshot; return its manifest.

    ``sql_dir/out_name.sql`` is the recipe; the composed subset lands in
    ``15_mixture_<out_name>`` (override with ``output_dir``), the instance JSONL
    and ``mixture_spec.sql`` under the experiments dir (override for tests).
    ``input_dir``/``input_stage`` follow the stage-entry pattern; with neither,
    the newest snapshot the run actually has is used (see INPUT_STAGE_CANDIDATES).

    The manifest is returned unsealed -- the runner seals and saves it. Its
    ``notes["recheck_required"]`` tells the runner to re-run S4/S6/S8 on the
    subset: recombination can reintroduce contamination, and the mixture is not
    final until those stages pass over ``15_mixture_<out_name>``.
    """
    started = utcnow()
    if not OUT_NAME_OK.fullmatch(out_name):
        raise StageError(
            f"S15: out_name {out_name!r} must be a plain phase name (letters/digits/._- only)"
        )
    sql_path = Path(sql_dir) / f"{out_name}.sql"
    if not sql_path.is_file():
        raise StageError(f"S15: mixture recipe {sql_path} does not exist")
    script = sql_path.read_text(encoding="utf-8")

    inp, resolved_name = _resolve_input(run_id, input_stage, input_dir)
    files = sorted(inp.glob("part-*.parquet"))
    if not files:
        raise StageError(f"S15: input snapshot {inp} contains no part-*.parquet files")

    out = (
        Path(output_dir)
        if output_dir is not None
        else store.stage_dir(run_id, f"{STAGE_PREFIX}_{out_name}")
    )
    if out.resolve() == inp.resolve():
        raise StageError(
            "S15: output snapshot must differ from the input snapshot (reset_dir would "
            "destroy the input)"
        )
    exp = (
        Path(experiments_dir) if experiments_dir is not None else store.EXPERIMENTS_ROOT / run_id
    )

    _assert_duckdb_support()
    cols, instances, input_counts, survivors = _execute_spec(files, script)
    unique_ids = _validate_selection(sql_path.name, cols, instances)

    # a re-run REPLACES its output (the runner's resume rule); reset_dir is the
    # store's one definition of that, and the guard above keeps it off the input
    store.reset_dir(out)
    matched = _compose_subset(inp, out, unique_ids)
    if matched != len(unique_ids):
        raise StageError(
            f"S15: recipe selected {len(unique_ids)} ids but only {matched} exist in {inp} "
            "-- input snapshot changed under the assembly"
        )

    inst_counts, uniq_counts = _per_source(cols, instances)
    manifest = StageManifest(
        run_id=run_id,
        stage=f"{STAGE_PREFIX}_{out_name}",
        started_at=started,
        config={
            "out_name": out_name,
            "sql_path": str(sql_path),
            "sql_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
            "sql_text": script,
            "input_stage": resolved_name,
            "input_dir": str(inp),
            "output_dir": str(out),
            "view": "corpus",
            "input_files": len(files),
            "duckdb_version": duckdb.__version__,
        },
        thresholds={},  # no numeric knobs consumed: every mixture constant lives in the recipe
    )
    manifest.rows_in = store.count_items(inp)
    manifest.rows_out = store.count_items(out)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.notes = {
        "recheck_required": list(RECHECK_REQUIRED),
        "recheck_reason": (
            "recombination can reintroduce near-duplicates and benchmark contamination; "
            "the runner re-runs these flag stages over the composed subset (S15 never "
            "invokes sibling stages itself)"
        ),
        "instances_selected": len(instances),
        "unique_rows": len(unique_ids),
        "rows_out_is_unique_rows": True,
        "sampling_with_replacement": len(instances) > len(unique_ids),
        "per_source_input": input_counts,
        "per_source_survivors": survivors,
        "per_source_instances": inst_counts,
        "per_source_unique": uniq_counts,
        "ids_jsonl": str(exp / f"{STAGE_PREFIX}_{out_name}.ids.jsonl"),
        "spec_sql": str(exp / "mixture_spec.sql"),
    }
    if manifest.rows_out != len(unique_ids):
        # not a bare assert: under `python -O` it vanishes and a snapshot that
        # does not match the recipe's selection would publish silently
        raise StageError(
            f"S15: snapshot holds {manifest.rows_out} rows but the recipe "
            f"selected {len(unique_ids)} unique ids"
        )

    exp.mkdir(parents=True, exist_ok=True)
    ids_path = exp / f"{STAGE_PREFIX}_{out_name}.ids.jsonl"
    with ids_path.open("w", encoding="utf-8") as fh:
        for pos, row in enumerate(instances):
            rec: dict[str, Any] = {"position": pos}
            rec.update(row)
            fh.write(json.dumps(rec, default=str) + "\n")

    header = [
        "-- mixture_spec.sql -- the exact SQL executed for this assembly.",
        f"-- run_id: {run_id}    phase: {out_name}    executed_at: {started.isoformat()}",
        f"-- input snapshot: {resolved_name} ({inp})    input_sha256={manifest.input_sha256}",
        f"-- input files: {len(files)} part file(s), {manifest.rows_in} rows",
        "-- rows per source (input / survivors / selected instances / unique selected):",
    ]
    for src in sorted(set(input_counts) | set(survivors) | set(inst_counts) | set(uniq_counts)):
        header.append(
            f"--   {src:<32} {input_counts.get(src, 0):>8} / {survivors.get(src, 0):>8} / "
            f"{inst_counts.get(src, 0):>8} / {uniq_counts.get(src, 0):>8}"
        )
    if not survivors:
        header.append("--   (survivor counts unavailable: this recipe defines no `survivor` macro)")
    spec_path = exp / "mixture_spec.sql"
    spec_path.write_text("\n".join(header) + "\n\n" + script, encoding="utf-8")

    log.info(
        "S15 %s: %d instances (%d unique) from %s -> %s; recheck %s",
        out_name,
        len(instances),
        len(unique_ids),
        resolved_name,
        out,
        ", ".join(RECHECK_REQUIRED),
    )
    return manifest


__all__ = [
    "INPUT_STAGE_CANDIDATES",
    "RECHECK_REQUIRED",
    "STAGE_PREFIX",
    "SURVIVOR_CALL",
    "run_mixture",
]
