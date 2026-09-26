"""``medrl`` command line.

Subcommands are thin: they validate configuration, compute a plan, and hand off to a
runtime. Every command works in ``--dry-run`` on a CPU box, because the way a GPU-only
tool rots is by being unrunnable anywhere except the GPU box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from medrl import __version__
from medrl.core.config import ClusterConfig, EvalConfig, Grade, TrainStrategy
from medrl.core.logging import setup_logging

app = typer.Typer(
    name="medrl",
    no_args_is_help=True,
    help="Bilingual medical reasoning LLM post-training stack.",
    add_completion=False,
)
model_app = typer.Typer(no_args_is_help=True, help="Model inspection and surgery.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation harness.")
data_app = typer.Typer(no_args_is_help=True, help="Data preparation and processing.")
train_app = typer.Typer(no_args_is_help=True, help="Training commands (SFT, preference, RL).")
ablation_app = typer.Typer(no_args_is_help=True, help="Ablation sweep management.")
app.add_typer(model_app, name="model")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(ablation_app, name="ablation")

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
        _run_eval(cfg)


@eval_app.command("run")
def eval_run(
    config: Annotated[str, typer.Option("--config", "-c", help="Hydra/YAML eval config path or preset.")] = "decision_grade",
    fresh: Annotated[bool, typer.Option("--fresh", help="Ignore completions already on disk.")] = False,
) -> None:
    """Run the eval end to end (policy generation, judging, scoring)."""
    from medrl.eval.config_loader import load_eval_config

    _run_eval(load_eval_config(config), resume=not fresh)


def _run_eval(cfg: EvalConfig, *, resume: bool = True) -> None:
    from dataclasses import asdict

    from medrl.eval.runner import run_eval
    from medrl.eval.tasks.benchmarks import TASKS
    from medrl.eval.tasks.spec import VerifyStyle

    results = run_eval(cfg, resume=resume)
    table = Table(title=f"results ({len(results)})")
    for col in ("benchmark", "points", "95% ci", "mde", "n", "reps", "extr.fail", "think.ok"):
        table.add_column(col)
    for name, r in results.items():
        reps = str(r.n_repeats) if not r.repeats_missing else f"{r.n_repeats}(-{r.repeats_missing})"
        # 0.0 would read as "clean" on benchmarks where the rate is not measured.
        extr = f"{r.extraction_fail_rate:.1%}" if TASKS.get(name).verify_style is VerifyStyle.LETTER else "-"
        table.add_row(
            name, f"{r.points:.1f}", f"[{r.ci_low:.1f}, {r.ci_high:.1f}]", f"{r.mde_points:.1f}",
            str(r.n_items), reps, extr, f"{r.think_completion_rate:.1%}",
        )
    console.print(table)
    console.print_json(json.dumps({k: asdict(v) for k, v in results.items()}))


@model_app.command("convert")
def model_convert(
    source: Annotated[str, typer.Option("--source")] = "Qwen/Qwen3.5-9B",
    out: Annotated[str, typer.Option("--out")] = "/scratch/medrl/checkpoints/qwen35-9b-text",
    verify_logits: Annotated[bool, typer.Option("--verify-logits/--no-verify-logits")] = True,
) -> None:
    """Emit the text-only checkpoint (strip vision + MTP) with equivalence checking."""
    try:
        from medrl.model.surgery import (  # requires torch + transformers
            SurgeryVerificationError,
            convert,
        )

        result = convert(source, out, verify_logits=verify_logits)
    except SurgeryVerificationError as exc:
        # Must precede the RuntimeError arm: the checkpoint is wrong, not unavailable.
        console.print(f"[red]verification failed, nothing written[/]: {exc}")
        raise typer.Exit(1) from exc
    except (ImportError, RuntimeError) as exc:
        # RuntimeError is surgery's own guard: its torch import is deferred to call time.
        console.print(f"[red]training stack not available[/]: {exc}")
        console.print("install with: uv pip install -e '.[train]'")
        raise typer.Exit(2) from exc
    console.print_json(json.dumps(result))


# ===============================================================================================
# Data commands
# ===============================================================================================


# ===============================================================================================
# Curation commands (S0-S15): thin wrappers over medrl.curation.runner.
# ===============================================================================================


@data_app.command("curate")
def data_curate(
    run_id: Annotated[Optional[str], typer.Option(help="Run id; defaults to cur-<timestamp>.")] = None,
    stage: Annotated[Optional[list[str]], typer.Option(help="Stage(s) to run; repeatable; default all registered.")] = None,
    list_stages: Annotated[bool, typer.Option("--list", help="List registered stages and exit.")] = False,
) -> None:
    """Run curation pipeline stages. Heavy outputs -> /scratch, manifests -> experiments/ (committed)."""
    from medrl.curation import runner as curation_runner

    if list_stages:
        curation_runner.register_builtin_stages()
        for name in sorted(curation_runner.STAGES):
            print(name)
        return
    final_run_id = run_id or f"cur-{__import__('datetime').datetime.now(tz=__import__('datetime').timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    curation_runner.register_builtin_stages()
    rc = curation_runner.run(final_run_id, stage or sorted(curation_runner.STAGES))
    raise typer.Exit(rc)


@data_app.command("prepare")
def data_prepare(
    config: Annotated[str, typer.Option("--config", "-c", help="Data pipeline config path or preset.")] = "sft-default",
    output: Annotated[str, typer.Option("--output", "-o", help="Output path for processed data.")] = "data/processed",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config without running.")] = False,
) -> None:
    """Prepare training data from raw sources.

    Runs the full data pipeline: loading, deduplication, decontamination,
    and filtering. Outputs processed data in the specified format.
    """
    from medrl.data.pipeline import PipelineBuilder

    console.print(f"[bold]config[/] {config}")
    console.print(f"[bold]output[/] {output}")

    if dry_run:
        console.print("[yellow]dry-run enabled, skipping pipeline execution[/]")
        # Validate config by attempting to load it
        try:
            # Try as preset first
            from medrl.core.paths import repo_root
            preset_path = repo_root() / "configs" / "data" / f"{config}.yaml"
            if preset_path.exists():
                console.print(f"[green]preset found:[/] {preset_path}")
            else:
                console.print(f"[yellow]config path or preset:[/]{config}")
        except Exception as e:
            console.print(f"[red]config validation failed:[/] {e}")
            raise typer.Exit(1) from e
        return

    # Build and run pipeline
    try:
        # Determine builder type from config name
        if "pref" in config.lower() or "dpo" in config.lower():
            builder = PipelineBuilder.preference(name=config)
        elif "rl" in config.lower() or "rollout" in config.lower():
            builder = PipelineBuilder.rl(name=config)
        else:
            builder = PipelineBuilder.sft(name=config)

        # Load config if path provided
        config_path = Path(config)
        if config_path.suffix in {".yaml", ".yml"}:
            import yaml
            with config_path.open() as f:
                cfg = yaml.safe_load(f)
                # Apply config settings to builder
                if cfg.get("dedup_threshold"):
                    builder = builder.with_dedup_threshold(cfg["dedup_threshold"])
                if cfg.get("filter_min_length") or cfg.get("filter_languages"):
                    builder = builder.with_filter(
                        min_length=cfg.get("filter_min_length", 10),
                        languages=tuple(cfg.get("filter_languages", ("en",))),
                    )

        pipeline = builder.build()

        # Load data from configured sources
        console.print("[bold]Loading data sources...[/]")
        # This would be extended to load from config
        # For now, just show the plan

        console.print("[bold]Pipeline plan:[/]")
        console.print(f"  Sources: {pipeline.config.sources}")
        console.print(f"  Deduplication: {pipeline.config.dedup_enabled} (threshold={pipeline.config.dedup_threshold})")
        console.print(f"  Decontamination: {pipeline.config.decontam_enabled}")
        console.print(f"  Filtering: {pipeline.config.filter_enabled}")
        console.print(f"  Output format: {pipeline.config.output_format}")

        # Run pipeline
        items = pipeline.run()
        pipeline.save(output)

        console.print(f"[green]Pipeline complete:[/] {len(items)} items -> {output}")

    except (ImportError, RuntimeError) as exc:
        console.print(f"[red]pipeline failed[/]: {exc}")
        raise typer.Exit(2) from exc


@data_app.command("decontaminate")
def data_decontaminate(
    train_path: Annotated[str, typer.Option("--train-path", help="Path to training data (JSONL).")] = "",
    benchmarks: Annotated[str, typer.Option("--benchmarks", help="Comma-separated benchmark names.")] = "medqa,medmcqa",
    ngram_threshold: Annotated[float, typer.Option("--ngram-threshold", help="N-gram overlap threshold.")] = 0.8,
    embedding_threshold: Annotated[float, typer.Option("--embedding-threshold", help="Embedding similarity threshold.")] = 0.9,
    check_embeddings: Annotated[bool, typer.Option("--check-embeddings/--no-check-embeddings", help="Run embedding checks.")] = False,
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for reports.")] = "outputs/decontamination",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config without running.")] = False,
) -> None:
    """Run decontamination analysis on training data.

    Detects and removes benchmark contamination using n-gram overlap
    and optionally embedding-based semantic similarity.
    """
    from medrl.data.decontam import (
        TrainDocument,
        build_benchmark_index,
        check_contamination,
        generate_per_benchmark_reports,
        save_contamination_report,
    )

    console.print(f"[bold]train data[/] {train_path}")
    console.print(f"[bold]benchmarks[/] {benchmarks}")
    console.print(f"[bold]output[/] {output_dir}")

    if dry_run:
        console.print("[yellow]dry-run enabled, skipping decontamination[/]")
        return

    try:
        # Load benchmark index
        benchmark_names = [b.strip() for b in benchmarks.split(",")]
        console.print(f"[bold]Loading benchmarks:[/] {len(benchmark_names)} benchmarks")

        index = build_benchmark_index(
            benchmark_names,
            include_embeddings=check_embeddings,
        )

        console.print(f"[green]Benchmark index built:[/] {len(index.items)} items")

        # Load training data
        console.print("[bold]Loading training data...[/]")
        train_docs = []
        train_path_obj = Path(train_path)

        if train_path_obj.suffix == ".jsonl":
            import json
            with train_path_obj.open() as f:
                for line_no, line in enumerate(f):
                    if line.strip():
                        record = json.loads(line)
                        # Extract text from common formats
                        text = record.get("text") or record.get("content", "")
                        if not text and "messages" in record:
                            # OpenAI chat format
                            text = " ".join(msg.get("content", "") for msg in record["messages"])

                        train_docs.append(TrainDocument(
                            id=record.get("id", f"doc_{line_no}"),
                            text=text,
                            source=record.get("source", "unknown"),
                        ))

        console.print(f"[green]Loaded {len(train_docs)} training documents[/]")

        # Run contamination check
        console.print("[bold]Running contamination check...[/]")
        result = check_contamination(
            train_docs,
            index,
            ngram_threshold=ngram_threshold,
            embedding_threshold=embedding_threshold,
            check_embeddings=check_embeddings,
        )

        # Save reports
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "contamination_report.json"
        save_contamination_report(result, report_path)

        generate_per_benchmark_reports(result, output_path)

        console.print("[green]Decontamination complete[/]")
        console.print(f"  Total docs: {result.total_train_docs}")
        console.print(f"  Contaminated: {result.contaminated_docs} ({result.contamination_rate():.1%})")
        console.print(f"  Clean: {len(result.clean_docs)}")
        console.print(f"  Report saved to: {report_path}")

    except (ImportError, RuntimeError) as exc:
        console.print(f"[red]decontamination failed[/]: {exc}")
        raise typer.Exit(2) from exc


# ===============================================================================================
# Training commands
# ===============================================================================================


@train_app.command("sft")
def train_sft(
    config: Annotated[str, typer.Option("--config", "-c", help="SFT config path or preset.")] = "sft-default",
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o", help="Output directory.")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from checkpoint.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config without training.")] = False,
) -> None:
    """Run supervised fine-tuning (SFT) training.

    Supports FSDP2 distributed training, sequence packing, and thinking mode.
    """
    from medrl.train.sft.config import load_sft_config
    from medrl.train.sft.runner import run_sft

    console.print(f"[bold]config[/] {config}")
    if output_dir:
        console.print(f"[bold]output[/] {output_dir}")

    try:
        cfg = load_sft_config(config)

        if dry_run:
            console.print("[yellow]dry-run enabled, skipping training[/]")
            console.print(f"[bold]Config fingerprint:[/] {cfg.fingerprint[:12]}...")
            console.print(f"[bold]Model:[/] {cfg.model.hf_id}")
            console.print(f"[bold]Data:[/] {cfg.data.train_path}")
            console.print(f"[bold]Epochs:[/] {cfg.num_epochs}")
            console.print(f"[bold]Max steps:[/] {cfg.max_steps}")
            console.print(f"[bold]Batch size:[/] {cfg.global_batch_size}")
            console.print(f"[bold]Learning rate:[/] {cfg.optimizer.lr}")
            return

        # Override output_dir if provided
        if output_dir:
            from dataclasses import replace
            cfg = replace(cfg, output_dir=output_dir)

        console.print(f"[bold]Starting SFT training[/] (fingerprint: {cfg.fingerprint[:12]}...)")

        result = run_sft(cfg)

        console.print("[green]Training complete[/]")
        console.print(f"  Final loss: {result.get('eval_loss', 'N/A')}")

    except FileNotFoundError as exc:
        # In dry-run mode, this is acceptable
        if dry_run:
            console.print("[yellow]dry-run mode[/]: config file not found, would create:")
            console.print(f"  Config: {config}")
            console.print(f"  Output: {output_dir or 'default'}")
            return
        console.print(f"[red]config not found:[/] {config}")
        raise typer.Exit(1) from exc
    except (ImportError, RuntimeError) as exc:
        console.print(f"[red]SFT training failed[/]: {exc}")
        console.print("install with: uv pip install -e '.[train]'")
        raise typer.Exit(2) from exc


@train_app.command("pref")
def train_pref(
    config: Annotated[str, typer.Option("--config", "-c", help="Preference config path or preset.")] = "dpo-default",
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory.")] = "outputs/pref",
    resume: Annotated[bool, typer.Option("--resume", help="Resume from checkpoint.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config without training.")] = False,
) -> None:
    """Run preference optimization training (DPO/SimPO).

    Trains models on preference pairs without a separate reward model.
    Supports DPO, SimPO, and RPO methods.
    """
    from medrl.train.pref.config import PrefTrainConfig, dpo_preset, load_pref_config, simpo_preset
    from medrl.train.pref.runner import run_pref_train

    console.print(f"[bold]config[/] {config}")
    console.print(f"[bold]output[/] {output_dir}")

    try:
        from medrl.core.config import ModelConfig

        # Load config
        cfg: PrefTrainConfig
        if config == "dpo-default":
            cfg = PrefTrainConfig(
                model=ModelConfig(hf_id="Qwen/Qwen3.5-9B"),
                pref=dpo_preset(),
                output_dir=output_dir,
            )
        elif config == "simpo-default":
            cfg = PrefTrainConfig(
                model=ModelConfig(hf_id="Qwen/Qwen3.5-9B"),
                pref=simpo_preset(),
                output_dir=output_dir,
            )
        else:
            # Try loading from file
            cfg = load_pref_config(config)
            if output_dir != "outputs/pref":
                from dataclasses import replace
                cfg = replace(cfg, output_dir=output_dir)

        if dry_run:
            console.print("[yellow]dry-run enabled, skipping training[/]")
            console.print(f"[bold]Config fingerprint:[/] {cfg.fingerprint[:12]}...")
            console.print(f"[bold]Model:[/] {cfg.model.hf_id}")
            console.print(f"[bold]Method:[/] {cfg.pref.method.value}")
            console.print(f"[bold]Beta:[/] {cfg.pref.beta}")
            console.print(f"[bold]Data:[/] {cfg.pref.train_path}")
            console.print(f"[bold]Epochs:[/] {cfg.pref.epochs}")
            return

        # Use default cluster config
        cluster = ClusterConfig(
            name="default",
            num_gpus=1,
            gpu_memory_gb=80,
            train_strategy=TrainStrategy.FSDP2,
            micro_batch_size=1,
        )

        console.print(f"[bold]Starting preference training[/] (fingerprint: {cfg.fingerprint[:12]}...)")

        result = run_pref_train(cfg, cluster)

        console.print("[green]Training complete[/]")
        console.print(f"  Final step: {result.step}")

    except FileNotFoundError as exc:
        # In dry-run mode, this is acceptable
        if dry_run:
            console.print("[yellow]dry-run mode[/]: config file not found, would use preset")
            console.print(f"  Config: {config}")
            console.print(f"  Output: {output_dir}")
            return
        console.print(f"[red]config not found:[/] {config}")
        raise typer.Exit(1) from exc
    except (ImportError, RuntimeError) as exc:
        console.print(f"[red]Preference training failed[/]: {exc}")
        console.print("install with: uv pip install -e '.[train]'")
        raise typer.Exit(2) from exc


@train_app.command("rl")
def train_rl(
    config: Annotated[str, typer.Option("--config", "-c", help="RL config path or preset.")] = "rl-default",
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory.")] = "outputs/rl",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config without training.")] = False,
) -> None:
    """Run reinforcement learning training (RLHF/reward modeling).

    This command is a placeholder for future RL training implementation.
    Currently, use preference optimization (train pref) for RL-style training.
    """
    console.print(f"[bold]config[/] {config}")
    console.print(f"[bold]output[/] {output_dir}")

    if dry_run:
        console.print("[yellow]dry-run enabled, skipping training[/]")
        console.print("[yellow]RL training not yet implemented[/]")
        console.print("Use 'medrl train pref' for preference optimization (DPO/SimPO).")
        return

    console.print("[red]RL training not yet implemented[/]")
    console.print("Use 'medrl train pref' for preference optimization (DPO/SimPO).")
    raise typer.Exit(1)


# ===============================================================================================
# Ablation commands
# ===============================================================================================


@ablation_app.command("sweep")
def ablation_sweep(
    config: Annotated[str, typer.Option("--config", "-c", help="Ablation config path or preset.")] = "A1",
    baseline_config: Annotated[str | None, typer.Option("--baseline-config", help="Baseline eval config path.")] = None,
    baseline_id: Annotated[str | None, typer.Option("--baseline-id", help="Baseline run ID.")] = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o", help="Output directory.")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from previous results.")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config without running.")] = False,
) -> None:
    """Run an ablation sweep.

    Executes multiple eval runs across a parameter space to measure the impact
    of configuration changes on benchmark performance.
    """
    from medrl.ablation.config import (
        ABLATION_A1,
        ABLATION_A2,
        ABLATION_A3,
        ABLATION_A4,
        ABLATION_A5,
        ABLATION_A6,
        ABLATION_A7,
        AblationConfig,
    )
    from medrl.ablation.sweep import run_sweep

    console.print(f"[bold]config[/] {config}")

    # Load ablation config
    ablation_cfg: AblationConfig
    preset_map = {
        "A1": ABLATION_A1,
        "A2": ABLATION_A2,
        "A3": ABLATION_A3,
        "A4": ABLATION_A4,
        "A5": ABLATION_A5,
        "A6": ABLATION_A6,
        "A7": ABLATION_A7,
    }

    if config in preset_map:
        # Need baseline_id for presets
        if not baseline_id:
            console.print("[red]baseline-id required for preset ablations[/]")
            console.print("Usage: medrl ablation sweep --config A1 --baseline-id <run_id>")
            raise typer.Exit(1)
        ablation_cfg = preset_map[config](baseline_id=baseline_id)
    else:
        # Load from file
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]config not found:[/] {config}")
            raise typer.Exit(1)

        import yaml
        with config_path.open() as f:
            raw = yaml.safe_load(f)
            ablation_cfg = AblationConfig.model_validate(raw)

    console.print(f"[bold]Ablation:[/] {ablation_cfg.name}")
    console.print(f"[bold]Baseline:[/] {ablation_cfg.baseline_id}")
    console.print(f"[bold]Parameters:[/] {len(ablation_cfg.sweep.parameters)}")

    if dry_run:
        console.print("[yellow]dry-run enabled, skipping sweep[/]")
        console.print(f"[bold]Config fingerprint:[/] {ablation_cfg.fingerprint[:12]}...")
        console.print(f"[bold]Strategy:[/] {ablation_cfg.sweep.strategy.type}")

        for param in ablation_cfg.sweep.parameters:
            console.print(f"  - {param.name}: {param.type.value} with {len(param.choices)} choices")
        return

    try:
        console.print(f"[bold]Starting ablation sweep[/] (fingerprint: {ablation_cfg.fingerprint[:12]}...)")

        result = run_sweep(
            ablation_cfg,
            baseline_config_path=baseline_config,
            resume=resume,
        )

        console.print("[green]Sweep complete[/]")
        console.print(f"  Completed arms: {len(result.completed_arms)}")
        console.print(f"  Failed arms: {len(result.failed_arms)}")

        if result.promotion_decision:
            console.print(f"  Promotion decision: {result.promotion_decision}")

        # Results are saved to experiments_dir()
        console.print(f"  Results saved to: experiments/ablation/{result.sweep_id}/")

    except (ImportError, RuntimeError) as exc:
        console.print(f"[red]Ablation sweep failed[/]: {exc}")
        raise typer.Exit(2) from exc


@ablation_app.command("list-presets")
def ablation_list_presets() -> None:
    """List available ablation presets."""
    from medrl.ablation.config import (
        ABLATION_A1_TEMPLATE,
        ABLATION_A2_TEMPLATE,
        ABLATION_A3_TEMPLATE,
        ABLATION_A4_TEMPLATE,
        ABLATION_A5_TEMPLATE,
        ABLATION_A6_TEMPLATE,
        ABLATION_A7_TEMPLATE,
    )

    presets = [
        ("A1", ABLATION_A1_TEMPLATE, "Thinking mode ablation"),
        ("A2", ABLATION_A2_TEMPLATE, "Sampling strategy ablation"),
        ("A3", ABLATION_A3_TEMPLATE, "Judge strength ablation"),
        ("A4", ABLATION_A4_TEMPLATE, "Context length ablation"),
        ("A5", ABLATION_A5_TEMPLATE, "Nucleus sampling ablation"),
        ("A6", ABLATION_A6_TEMPLATE, "Accuracy vs compute Pareto"),
        ("A7", ABLATION_A7_TEMPLATE, "Guardrail sweep"),
    ]

    table = Table(title="ablation presets")
    table.add_column("name")
    table.add_column("description")
    table.add_column("parameters")
    table.add_column("tags")

    for name, preset, description in presets:
        param_names = ", ".join(p.name for p in preset.sweep.parameters)
        tags = ", ".join(preset.tags)
        table.add_row(name, description, param_names, tags)

    console.print(table)


if __name__ == "__main__":
    app()
