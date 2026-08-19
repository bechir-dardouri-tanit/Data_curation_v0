"""``medrl`` command line.

Subcommands are thin: they validate configuration, compute a plan, and hand off to a
runtime. Every command works in ``--dry-run`` on a CPU box, because the way a GPU-only
tool rots is by being unrunnable anywhere except the GPU box.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from medrl import __version__
from medrl.core.config import Grade
from medrl.core.logging import setup_logging

app = typer.Typer(
    name="medrl",
    no_args_is_help=True,
    help="Bilingual medical reasoning LLM post-training stack.",
    add_completion=False,
)
model_app = typer.Typer(no_args_is_help=True, help="Model inspection and surgery.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation harness.")
app.add_typer(model_app, name="model")
app.add_typer(eval_app, name="eval")

# Fixed width: a non-tty subprocess gets an 80-col terminal and rich would truncate
# registry names to 'healthben…', which breaks scripted consumption of the output.
console = Console(width=160)
Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


@app.callback()
def _main(verbose: Verbose = False) -> None:
    setup_logging(10 if verbose else 20)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"medrl {__version__}")


@app.command()
def probe(
    repo: Annotated[str, typer.Option("--repo", help="HF repo to assert invariants for.")] = "Qwen/Qwen3.5-9B",
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    config_path: Annotated[str | None, typer.Option("--config", help="Local config.json.")] = None,
) -> None:
    """Assert architecture invariants; exit non-zero on drift."""
    from medrl.model.probe import main as probe_main

    raise typer.Exit(probe_main(["--repo", repo] + (["--revision", revision] if revision else [])
                                + (["--config", config_path] if config_path else [])))


@eval_app.command("tasks")
def eval_tasks(
    grade: Annotated[str | None, typer.Option("--grade", help="decision|reporting|guardrail")] = None,
) -> None:
    """List the benchmark registry."""
    from medrl.eval.tasks.benchmarks import TASKS

    wanted = Grade(grade) if grade else None
    table = Table(title=f"benchmarks ({len(TASKS)})")
    for col in ("name", "grade", "lang", "prompt", "verify", "judge", "hf_id"):
        table.add_column(col)
    for name in TASKS:
        spec = TASKS.get(name)
        if wanted and spec.grade is not wanted:
            continue
        table.add_row(
            name, spec.grade.value, spec.language.value, spec.prompt_style.value,
            spec.verify_style.value, "yes" if spec.requires_judge else "-",
            spec.hf_id or "-",
        )
    console.print(table)


@eval_app.command("plan")
def eval_plan(
    config: Annotated[str, typer.Option("--config", "-c", help="Hydra/YAML eval config path or preset.")] = "decision_grade",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = True,
) -> None:
    """Validate config and show the serving deployment plan."""
    from medrl.eval.config_loader import load_eval_config
    from medrl.eval.serving.vllm import plan_deployment

    cfg = load_eval_config(config)
    plan = plan_deployment(cfg)
    console.print(f"[bold]config[/] {cfg.fingerprint}  ({len(cfg.benchmarks)} benchmarks)")
    console.print(plan.render())
    if not dry_run:  # pragma: no cover - runtime path needs vLLM
        from medrl.eval.runner import run_eval

        results = run_eval(cfg)
        console.print_json(json.dumps(results, default=str))


@model_app.command("convert")
def model_convert(
    source: Annotated[str, typer.Option("--source")] = "Qwen/Qwen3.5-9B",
    out: Annotated[str, typer.Option("--out")] = "/scratch/medrl/checkpoints/qwen35-9b-text",
    verify_logits: Annotated[bool, typer.Option("--verify-logits/--no-verify-logits")] = True,
) -> None:
    """Emit the text-only checkpoint (strip vision + MTP) with equivalence checking."""
    try:
        from medrl.model.surgery import convert  # requires torch + transformers

        result = convert(source, out, verify_logits=verify_logits)
    except (ImportError, RuntimeError) as exc:
        # RuntimeError is surgery's own guard: its torch import is deferred to call time.
        console.print(f"[red]training stack not available[/]: {exc}")
        console.print("install with: uv pip install -e '.[train]'")
        raise typer.Exit(2) from exc
    console.print_json(json.dumps(result))


if __name__ == "__main__":
    app()
