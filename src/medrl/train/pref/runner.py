"""Preference optimization training runner.

Orchestrates DPO/SimPO training with:
- FSDP2 compatibility
- W&B integration
- Checkpoint management
- Beta tuning
- Proper validation

This follows the same patterns as the eval runner for consistency.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch convention
)

from medrl.core.config import ClusterConfig, TrainStrategy
from medrl.core.logging import get_logger
from medrl.train.pref.config import PrefOptConfig, PrefTrainConfig
from medrl.train.pref.data import PrefBatch, PrefDataset, create_dataloader
from medrl.train.pref.loss import PrefLoss, PrefLossOutput, compute_grad_norm

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Training state
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainMetrics:
    """Metrics logged during training."""

    step: int
    epoch: int
    loss: float
    policy_loss: float
    sft_loss: float
    accuracy: float
    chosen_logps_mean: float
    rejected_logps_mean: float
    chosen_length_mean: float
    rejected_length_mean: float
    grad_norm: float
    learning_rate: float
    beta: float | None = None  # Set during beta tuning


@dataclass(frozen=True)
class TrainState:
    """Current training state.

    Attributes:
        step: Global training step.
        epoch: Current epoch (0-indexed).
        global_batch_size: Effective batch size (accounting for grad accum).
        best_val_loss: Best validation loss seen so far.
        beta: Current beta value (for beta tuning).
    """

    step: int = 0
    epoch: int = 0
    global_batch_size: int = 0
    best_val_loss: float = float("inf")
    beta: float | None = None

    def checkpoint_path(self, output_dir: Path) -> Path:
        """Get path to checkpoint file."""
        return output_dir / f"checkpoint-step-{self.step}"

    def is_best(self, val_loss: float) -> bool:
        """Check if this is the best validation loss so far."""
        return val_loss < self.best_val_loss


# --------------------------------------------------------------------------------------
# Tracking sink
# --------------------------------------------------------------------------------------


class PrefTrackingSink:
    """Tracking sink for preference training.

    Logs metrics and optionally writes to W&B.
    """

    def __init__(self, config: PrefTrainConfig) -> None:
        """Initialize the tracking sink.

        Args:
            config: Training configuration.
        """
        self.config = config
        self._backend: Any | None = None

        if config.tracking.enabled and config.tracking.backend == "wandb":
            try:
                import wandb

                self._backend = wandb
                run_name = config.run_name or f"pref-{config.pref.fingerprint[:8]}"
                self._run = wandb.init(
                    project=config.tracking.project,
                    entity=config.tracking.entity,
                    group=config.tracking.group,
                    job_type="pref_train",
                    name=run_name,
                    tags=list(config.tracking.tags),
                    config=self._flatten_config(config),
                    reinit=True,
                )
                log.info("W&B tracking enabled: %s", run_name)
            except ImportError:
                log.warning("wandb not installed; tracking disabled")

    def _flatten_config(self, config: PrefTrainConfig) -> dict[str, Any]:
        """Flatten config for logging."""
        return {
            "model_ref": config.model.ref,
            "method": config.pref.method.value,
            "beta": config.pref.beta,
            "length_norm": config.pref.length_norm.value,
            "sft_loss_weight": config.pref.sft_loss_weight,
            "learning_rate": config.pref.learning_rate,
            "max_length": config.pref.max_length,
            "train_path": config.pref.train_path,
            "config_hash": config.fingerprint,
        }

    def log_metrics(self, metrics: TrainMetrics, step: int) -> None:
        """Log training metrics."""
        log_dict = {
            "train/loss": metrics.loss,
            "train/policy_loss": metrics.policy_loss,
            "train/sft_loss": metrics.sft_loss,
            "train/accuracy": metrics.accuracy,
            "train/chosen_logps": metrics.chosen_logps_mean,
            "train/rejected_logps": metrics.rejected_logps_mean,
            "train/chosen_length": metrics.chosen_length_mean,
            "train/rejected_length": metrics.rejected_length_mean,
            "train/grad_norm": metrics.grad_norm,
            "train/learning_rate": metrics.learning_rate,
            "train/step": metrics.step,
            "train/epoch": metrics.epoch,
        }

        if metrics.beta is not None:
            log_dict["train/beta"] = metrics.beta

        # Log to console at info level
        log.info(
            "Step %d (epoch %d): loss=%.4f, policy=%.4f, sft=%.4f, acc=%.3f, grad_norm=%.2f",
            metrics.step,
            metrics.epoch,
            metrics.loss,
            metrics.policy_loss,
            metrics.sft_loss,
            metrics.accuracy,
            metrics.grad_norm,
        )

        # Log to W&B if available
        if self._backend and self._run:
            self._run.log(log_dict, step=step)

    def log_validation(
        self,
        val_loss: float,
        val_accuracy: float,
        step: int,
    ) -> None:
        """Log validation metrics."""
        log_dict = {
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
        }

        log.info("Validation at step %d: loss=%.4f, acc=%.3f", step, val_loss, val_accuracy)

        if self._backend and self._run:
            self._run.log(log_dict, step=step)

    def finish(self) -> None:
        """Finish the tracking run."""
        if self._backend and self._run:
            self._run.finish()


# --------------------------------------------------------------------------------------
# Checkpoint management
# --------------------------------------------------------------------------------------


class CheckpointManager:
    """Manages training checkpoints."""

    def __init__(self, output_dir: Path, max_to_keep: int = 3) -> None:
        """Initialize the checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints.
            max_to_keep: Maximum number of checkpoints to retain.
        """
        self.output_dir = output_dir
        self.max_to_keep = max_to_keep
        self.checkpoints: list[Path] = []

    def save(
        self,
        state: TrainState,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint.

        Args:
            state: Training state.
            model: The model (may be FSDP-wrapped).
            optimizer: The optimizer.
            scheduler: The learning rate scheduler.
            is_best: Whether this is the best checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint_path = state.checkpoint_path(self.output_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Handle FSDP checkpointing
        if isinstance(model, FSDP):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                state_dict = model.state_dict()
        else:
            state_dict = model.state_dict()

        # Save checkpoint
        checkpoint = {
            "state": {
                "step": state.step,
                "epoch": state.epoch,
                "best_val_loss": state.best_val_loss,
                "beta": state.beta,
            },
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
        }

        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path / "model.pt")

        # Track checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            log.info("Saved best model to %s", best_path)

        self.checkpoints.append(checkpoint_path)

        # Prune old checkpoints
        self._prune()

        return checkpoint_path

    def _prune(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        while len(self.checkpoints) > self.max_to_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                for file in old_path.glob("*"):
                    file.unlink()
                old_path.rmdir()
                log.debug("Pruned old checkpoint: %s", old_path)

    def load(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> TrainState:
        """Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or file.
            model: The model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.

        Returns:
            The loaded training state.
        """
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "model.pt"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model state
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch convention
        )
        from torch.distributed.fsdp import StateDictType
        if isinstance(model, FSDP):

            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        # Load optimizer state
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state
        if scheduler and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        # Return training state
        state_dict = checkpoint.get("state", {})
        return TrainState(
            step=state_dict.get("step", 0),
            epoch=state_dict.get("epoch", 0),
            best_val_loss=state_dict.get("best_val_loss", float("inf")),
            beta=state_dict.get("beta"),
        )


# --------------------------------------------------------------------------------------
# Beta tuning
# --------------------------------------------------------------------------------------


class BetaTuner:
    """Tunes the beta hyperparameter during early training.

    Strategy: Sweep beta values across a range and select the one with the
    lowest validation loss after a short tuning phase.
    """

    def __init__(
        self,
        config: PrefOptConfig,
        n_steps: int,
    ) -> None:
        """Initialize the beta tuner.

        Args:
            config: Training configuration.
            n_steps: Number of steps for tuning phase.
        """
        self.config = config
        self.n_steps = n_steps
        self.betas = self._generate_betas()
        self.current_idx = 0
        self.step_in_trial = 0
        self.losses: dict[float, list[float]] = {b: [] for b in self.betas}

    def _generate_betas(self) -> list[float]:
        """Generate beta values to try."""
        low, high = self.config.beta_tune_range
        # Try 5 values evenly spaced
        n_trials = 5
        return [low + (high - low) * i / (n_trials - 1) for i in range(n_trials)]

    @property
    def current_beta(self) -> float:
        """Current beta value to use."""
        return self.betas[self.current_idx]

    def update(self, loss: float) -> bool:
        """Update with latest loss and check if trial is complete.

        Args:
            loss: Latest training loss.

        Returns:
            True if tuning phase is complete.
        """
        self.losses[self.current_beta].append(loss)
        self.step_in_trial += 1

        if self.step_in_trial >= self.n_steps:
            # Move to next beta
            self.current_idx += 1
            self.step_in_trial = 0

            if self.current_idx >= len(self.betas):
                # Tuning complete
                return True

        return False

    def select_best_beta(self) -> float:
        """Select the best beta based on average loss."""
        best_beta = self.config.beta
        best_loss = float("inf")

        for beta, losses in self.losses.items():
            if losses:
                avg_loss = sum(losses) / len(losses)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_beta = beta

        log.info(
            "Beta tuning complete: selected beta=%.4f (avg loss %.4f)",
            best_beta,
            best_loss,
        )
        return best_beta


# --------------------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------------------


@dataclass
class TrainerComponents:
    """Components needed for training."""

    model: torch.nn.Module
    ref_model: torch.nn.Module | None
    optimizer: torch.optim.Optimizer
    scheduler: Any
    loss_fn: PrefLoss


def run_pref_train(
    config: PrefTrainConfig,
    cluster: ClusterConfig,
) -> TrainState:
    """Execute preference optimization training.

    Args:
        config: Training configuration.
        cluster: Cluster configuration for FSDP.

    Returns:
        Final training state.
    """
    # Set up output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up tracking
    tracker = PrefTrackingSink(config)

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load datasets
    train_dataset = PrefDataset(
        config.pref.train_path,
        num_examples=config.pref.num_examples,
    )

    val_dataset: PrefDataset | None = None
    if config.pref.val_path:
        val_dataset = PrefDataset(config.pref.val_path)

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model: %s", config.model.ref)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.ref,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up distributed training if needed
    world_size = 1
    if torch.cuda.is_available() and not dist.is_initialized():
        # Initialize process group if not already done
        backend = "nccl" if torch.cuda.device_count() > 1 else "cpu"
        if backend == "nccl":
            dist.init_process_group(backend="nccl")
            dist.get_rank()
            world_size = dist.get_world_size()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.ref,
        torch_dtype=torch.bfloat16,
        trust_remote_code=config.model.trust_remote_code,
    )

    # Wrap with FSDP if needed
    if cluster.train_strategy in (TrainStrategy.FSDP2, TrainStrategy.FSDP2_OFFLOAD):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch convention
        )
        from torch.distributed.fsdp import (
            MixedPrecision,
            ShardingStrategy,
        )

        sharding = ShardingStrategy.FULL_SHARD
        if cluster.train_strategy == TrainStrategy.FSDP2_OFFLOAD:
            sharding = ShardingStrategy.SHARD_GRAD_OP

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        auto_wrap_policy = None  # Could add transformer auto-wrap policy here

        model = FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
        )  # type: ignore[assignment]

    # Load reference model (for DPO, not SimPO)
    ref_model: torch.nn.Module | None = None
    if config.pref.method.value == "dpo":
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.ref,
            torch_dtype=torch.bfloat16,
            trust_remote_code=config.model.trust_remote_code,
        )
        ref_model.eval()  # type: ignore[no-untyped-call]
        if torch.cuda.is_available():
            ref_model = ref_model.cuda()  # type: ignore[call-arg]

    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pref.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Create scheduler
    total_steps = (len(train_dataset) // cluster.micro_batch_size) * config.pref.epochs
    warmup_steps = int(total_steps * config.pref.warmup_ratio)

    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    warmup = LinearLR(optimizer, start_factor=0.0, total_iters=warmup_steps)
    decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

    # Create loss function
    loss_fn = PrefLoss(config.pref)

    # Create data loaders
    effective_batch_size = cluster.micro_batch_size * world_size
    grad_accum_steps = max(1, effective_batch_size // cluster.micro_batch_size)

    train_loader = create_dataloader(
        train_dataset,
        tokenizer,
        config.pref,
        batch_size=cluster.micro_batch_size,
        shuffle=True,
    )

    val_loader: Any = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            tokenizer,
            config.pref,
            batch_size=cluster.micro_batch_size,
            shuffle=False,
        )

    # Initialize training state
    state = TrainState(global_batch_size=effective_batch_size)
    checkpoint_manager = CheckpointManager(output_dir)

    # Resume from checkpoint if requested
    if config.resume_from_checkpoint and config.checkpoint_dir:
        checkpoint_path = Path(config.checkpoint_dir)
        state = checkpoint_manager.load(checkpoint_path, model, optimizer, scheduler)
        log.info("Resumed from checkpoint: step %d", state.step)

    # Set up beta tuning
    beta_tuner: BetaTuner | None = None
    if config.pref.beta_tune:
        beta_tuner = BetaTuner(config.pref, config.pref.beta_tune_steps)
        state = dataclasses.replace(state, beta=beta_tuner.current_beta)
        log.info("Beta tuning enabled: trying betas %s", beta_tuner.betas)

    # Training loop
    components = TrainerComponents(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )

    for epoch in range(config.pref.epochs):
        state = dataclasses.replace(state, epoch=epoch)

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = _move_batch_to_device(batch, model.device)

            # Forward pass
            loss_output = _forward_batch(components, batch, config)

            # Scale loss for gradient accumulation
            loss = loss_output.loss / grad_accum_steps
            loss.backward()  # type: ignore[no-untyped-call]

            # Gradient update
            if (batch_idx + 1) % grad_accum_steps == 0:
                grad_norm = compute_grad_norm(model)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                state = dataclasses.replace(state, step=state.step + 1)

                # Log metrics
                metrics = TrainMetrics(
                    step=state.step,
                    epoch=epoch,
                    loss=loss_output.loss.item(),
                    policy_loss=loss_output.policy_loss.item(),
                    sft_loss=loss_output.sft_loss.item(),
                    accuracy=loss_output.accuracy.item(),
                    chosen_logps_mean=loss_output.chosen_logps.mean().item(),
                    rejected_logps_mean=loss_output.rejected_logps.mean().item(),
                    chosen_length_mean=loss_output.chosen_lengths.mean().item(),
                    rejected_length_mean=loss_output.rejected_lengths.mean().item(),
                    grad_norm=grad_norm,
                    learning_rate=float(scheduler.get_last_lr()[0]),
                    beta=state.beta,
                )

                tracker.log_metrics(metrics, state.step)

                # Save checkpoint
                if state.step % config.pref.save_interval == 0:
                    checkpoint_manager.save(state, model, optimizer, scheduler)

                # Validation
                if val_loader and state.step % config.pref.eval_interval == 0:
                    val_metrics = _validate(components, val_loader, config)
                    tracker.log_validation(
                        val_metrics["loss"],
                        val_metrics["accuracy"],
                        state.step,
                    )

                    if state.is_best(val_metrics["loss"]):
                        state = dataclasses.replace(state, best_val_loss=val_metrics["loss"])
                        checkpoint_manager.save(
                            state,
                            model,
                            optimizer,
                            scheduler,
                            is_best=True,
                        )

                # Update beta tuning
                if beta_tuner:
                    tuning_complete = beta_tuner.update(loss_output.loss.item())
                    if tuning_complete:
                        best_beta = beta_tuner.select_best_beta()
                        state = dataclasses.replace(state, beta=best_beta)
                        beta_tuner = None  # Stop tuning

                        # Update loss function with new beta
                        # Create a new config with the updated beta
                        updated_pref = PrefOptConfig(
                            **{**config.pref.model_dump(), "beta": best_beta}
                        )
                        components.loss_fn = PrefLoss(updated_pref)

    # Final checkpoint
    checkpoint_manager.save(state, model, optimizer, scheduler, is_best=True)
    tracker.finish()

    return state


def _move_batch_to_device(batch: PrefBatch, device: torch.device) -> PrefBatch:
    """Move batch tensors to device."""
    return PrefBatch(
        chosen_ids=batch.chosen_ids.to(device),
        rejected_ids=batch.rejected_ids.to(device),
        chosen_labels=batch.chosen_labels.to(device),
        rejected_labels=batch.rejected_labels.to(device),
        chosen_mask=batch.chosen_mask.to(device),
        rejected_mask=batch.rejected_mask.to(device),
        metadata=batch.metadata,
    )


def _forward_batch(
    components: TrainerComponents,
    batch: PrefBatch,
    config: PrefTrainConfig,
) -> PrefLossOutput:
    """Forward pass for a single batch.

    Args:
        components: Training components.
        batch: The input batch.
        config: Training configuration.

    Returns:
        Loss output.
    """
    # Forward through policy model
    policy_chosen_out = components.model(
        input_ids=batch.chosen_ids,
        attention_mask=batch.chosen_mask,
        labels=None,  # We compute loss separately
    )
    policy_rejected_out = components.model(
        input_ids=batch.rejected_ids,
        attention_mask=batch.rejected_mask,
        labels=None,
    )

    # Forward through reference model if needed
    ref_chosen_out: Any = None
    ref_rejected_out: Any = None
    if components.ref_model is not None:
        with torch.no_grad():
            ref_chosen_out = components.ref_model(
                input_ids=batch.chosen_ids,
                attention_mask=batch.chosen_mask,
            )
            ref_rejected_out = components.ref_model(
                input_ids=batch.rejected_ids,
                attention_mask=batch.rejected_mask,
            )

    # Compute loss
    loss_output: PrefLossOutput = components.loss_fn(
        policy_chosen_logits=policy_chosen_out.logits,
        policy_rejected_logits=policy_rejected_out.logits,
        ref_chosen_logits=ref_chosen_out.logits if ref_chosen_out else torch.empty(0),
        ref_rejected_logits=ref_rejected_out.logits if ref_rejected_out else torch.empty(0),
        chosen_labels=batch.chosen_labels,
        rejected_labels=batch.rejected_labels,
        chosen_mask=batch.chosen_mask,
        rejected_mask=batch.rejected_mask,
    )
    return loss_output


def _validate(
    components: TrainerComponents,
    val_loader: Any,
    config: PrefTrainConfig,
) -> dict[str, float]:
    """Run validation.

    Args:
        components: Training components.
        val_loader: Validation data loader.
        config: Training configuration.

    Returns:
        Validation metrics.
    """
    components.model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            device = next(components.model.parameters()).device
            batch = _move_batch_to_device(batch, device)

            loss_output = _forward_batch(components, batch, config)

            total_loss += loss_output.loss.item()
            total_accuracy += loss_output.accuracy.item()
            n_batches += 1

    components.model.train()

    return {
        "loss": total_loss / max(1, n_batches),
        "accuracy": total_accuracy / max(1, n_batches),
    }
