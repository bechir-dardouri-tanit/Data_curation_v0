"""Checkpoint management for SFT training.

Handles saving, loading, and managing training checkpoints with support for:
- Distributed training (FSDP2)
- Best model tracking
- Resumable training
- Checkpoint pruning (keeping only N most recent)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from medrl.core.logging import get_logger
from medrl.train.sft.config import SFTCheckpointConfig

log = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""

    step: int
    epoch: int
    global_batch: int
    loss: float
    eval_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None))
    is_best: bool = False
    model_fingerprint: str = ""  # Config fingerprint for validation


@dataclass
class CheckpointState:
    """Complete training state for resumption."""

    step: int
    epoch: int
    global_batch: int
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] = field(default_factory=dict)
    metadata: CheckpointMetadata | None = None


class CheckpointManager:
    """Manager for SFT training checkpoints.

    Handles:
    - Periodic checkpoint saving
    - Best model tracking
    - Checkpoint pruning (save_total_limit)
    - Distributed checkpoint loading/saving (FSDP2)
    - Training resumption
    """

    def __init__(
        self,
        config: SFTCheckpointConfig,
        save_dir: str | Path,
        is_best_enabled: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.save_dir = Path(save_dir)
        self.is_best_enabled = is_best_enabled
        self.rank = rank
        self.world_size = world_size

        # Tracking
        self._checkpoints: list[tuple[int, Path]] = []  # (step, path) sorted
        self._best_metric = float("inf") if not config.greater_is_better else float("-inf")
        self._best_checkpoint_path: Path | None = None

        # Create save directory
        if self.rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup
        self._setup_directory()

    def _setup_directory(self) -> None:
        """Initialize checkpoint directory structure."""
        if self.rank != 0:
            return

        # Create subdirectories
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "best").mkdir(exist_ok=True)

        # Load existing checkpoint metadata if resuming
        if self.config.auto_resume and (self.save_dir / "latest.json").exists():
            self._load_metadata()

    def _load_metadata(self) -> None:
        """Load checkpoint metadata from disk."""
        try:
            with (self.save_dir / "latest.json").open() as f:
                metadata = json.load(f)

            self._best_metric = metadata.get("best_metric", self._best_metric)
            self._best_checkpoint_path = (
                Path(metadata["best_checkpoint"]) if metadata.get("best_checkpoint") else None
            )
            log.info("Loaded checkpoint metadata from %s", self.save_dir)
        except Exception as e:
            log.warning("Failed to load checkpoint metadata: %s", e)

    def _save_metadata(self, latest_path: Path | None = None) -> None:
        """Save checkpoint metadata to disk."""
        if self.rank != 0:
            return

        metadata = {
            "best_metric": self._best_metric,
            "best_checkpoint": str(self._best_checkpoint_path) if self._best_checkpoint_path else None,
            "latest_checkpoint": str(latest_path) if latest_path else None,
            "checkpoints": [(step, str(path)) for step, path in self._checkpoints],
            "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        }

        with (self.save_dir / "latest.json").open("w") as f:
            json.dump(metadata, f, indent=2)

    def should_save(self, step: int, epoch: int) -> bool:
        """Check if a checkpoint should be saved at this step/epoch."""
        if self.config.save_strategy == "steps":
            return self.config.save_steps is not None and step % self.config.save_steps == 0
        elif self.config.save_strategy == "epochs":
            return self.config.save_epochs is not None and epoch % self.config.save_epochs == 0
        elif self.config.save_strategy == "best":
            # Handled separately in save_best
            return False
        return False

    def update_best(self, metric: float) -> bool:
        """Check if metric is better than current best and update tracking.

        Args:
            metric: Metric value to compare.

        Returns:
            True if this is a new best.
        """
        is_better = (
            metric < self._best_metric if not self.config.greater_is_better
            else metric > self._best_metric
        )

        if is_better:
            self._best_metric = metric
            return True

        return False

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,  # torch.optim.lr_scheduler._LRScheduler
        step: int,
        epoch: int,
        global_batch: int,
        loss: float,
        eval_loss: float | None = None,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
        rng_state: dict[str, Any] | None = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            model: Model to save (FSDP2-wrapped or base).
            optimizer: Optimizer state.
            scheduler: LR scheduler state.
            step: Current training step.
            epoch: Current epoch.
            global_batch: Global batch number.
            loss: Current training loss.
            eval_loss: Validation loss if available.
            metrics: Additional metrics.
            is_best: Whether this is the best model so far.
            rng_state: Random number generator states.

        Returns:
            Path to saved checkpoint.
        """
        if self.rank != 0:
            # Only rank 0 saves checkpoints
            return Path() if not is_best else self._best_checkpoint_path or Path()

        checkpoint_name = f"checkpoint-{step}"
        checkpoint_dir = self.save_dir / "checkpoints" / checkpoint_name

        # Prepare metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            global_batch=global_batch,
            loss=loss,
            eval_loss=eval_loss,
            metrics=metrics or {},
            is_best=is_best,
        )

        # For FSDP2, use distributed checkpointing
        if self._is_fsdp_model(model):
            self._save_fsdp_checkpoint(
                model, optimizer, scheduler, checkpoint_dir, metadata, rng_state
            )
        else:
            self._save_standard_checkpoint(
                model, optimizer, scheduler, checkpoint_dir, metadata, rng_state
            )

        # Track checkpoint
        self._checkpoints.append((step, checkpoint_dir))
        self._checkpoints.sort(key=lambda x: x[0])  # Keep sorted

        # Prune old checkpoints
        if self.config.save_total_limit > 0:
            self._prune_checkpoints()

        # Update latest metadata
        self._save_metadata(checkpoint_dir)

        # Save best model
        if is_best and self.is_best_enabled:
            best_path = self.save_best(
                model, optimizer, scheduler, step, epoch, global_batch,
                loss, eval_loss, metrics or {}, rng_state
            )
            self._best_checkpoint_path = best_path
            self._save_metadata(checkpoint_dir)

        return checkpoint_dir

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int,
        global_batch: int,
        loss: float,
        eval_loss: float | None = None,
        metrics: dict[str, float] | None = None,
        rng_state: dict[str, Any] | None = None,
    ) -> Path:
        """Save the best model checkpoint.

        Args:
            Same as save().

        Returns:
            Path to best model checkpoint.
        """
        if self.rank != 0:
            return self._best_checkpoint_path or Path()

        best_dir = self.save_dir / "best"

        # Prepare metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            global_batch=global_batch,
            loss=loss,
            eval_loss=eval_loss,
            metrics=metrics or {},
            is_best=True,
        )

        # Save best model
        if self._is_fsdp_model(model):
            self._save_fsdp_checkpoint(
                model, optimizer, scheduler, best_dir, metadata, rng_state
            )
        else:
            self._save_standard_checkpoint(
                model, optimizer, scheduler, best_dir, metadata, rng_state
            )

        # Save a symlink for easy access
        latest_link = self.save_dir / "best_model"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(best_dir)

        log.info("Saved best model at step %d to %s", step, best_dir)
        return best_dir

    def _is_fsdp_model(self, model: torch.nn.Module) -> bool:
        """Check if model is FSDP2-wrapped."""
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch convention
            )
            return any(isinstance(m, FSDP) for m in model.modules())
        except ImportError:
            return False

    def _save_fsdp_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: Path,
        metadata: CheckpointMetadata,
        rng_state: dict[str, Any] | None = None,
    ) -> None:
        """Save FSDP2 checkpoint with sharded state."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch convention
        )
        from torch.distributed.fsdp import StateDictType

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save full state dict for model (rank 0 only)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model_state = model.state_dict()

        # Save model state
        model_file = checkpoint_dir / "model.safetensors"
        self._save_model_safe(model_state, model_file)

        # Save optimizer state
        optimizer_file = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_file)

        # Save scheduler state
        if scheduler is not None:
            scheduler_file = checkpoint_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_file)

        # Save RNG states
        if rng_state:
            rng_file = checkpoint_dir / "rng_state.pt"
            torch.save(rng_state, rng_file)

        # Save metadata
        with (checkpoint_dir / "metadata.json").open("w") as f:
            json.dump({
                "step": metadata.step,
                "epoch": metadata.epoch,
                "global_batch": metadata.global_batch,
                "loss": metadata.loss,
                "eval_loss": metadata.eval_loss,
                "metrics": metadata.metrics,
                "is_best": metadata.is_best,
                "timestamp": metadata.timestamp.isoformat(),
            }, f, indent=2)

        log.info("Saved FSDP checkpoint to %s", checkpoint_dir)

    def _save_standard_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: Path,
        metadata: CheckpointMetadata,
        rng_state: dict[str, Any] | None = None,
    ) -> None:
        """Save standard (non-FSDP) checkpoint."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = checkpoint_dir / "model.safetensors"
        model_state = model.state_dict()
        self._save_model_safe(model_state, model_file)

        # Save optimizer
        optimizer_file = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_file)

        # Save scheduler
        if scheduler is not None:
            scheduler_file = checkpoint_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_file)

        # Save RNG states
        if rng_state:
            rng_file = checkpoint_dir / "rng_state.pt"
            torch.save(rng_state, rng_file)

        # Save metadata
        with (checkpoint_dir / "metadata.json").open("w") as f:
            json.dump({
                "step": metadata.step,
                "epoch": metadata.epoch,
                "global_batch": metadata.global_batch,
                "loss": metadata.loss,
                "eval_loss": metadata.eval_loss,
                "metrics": metadata.metrics,
                "is_best": metadata.is_best,
                "timestamp": metadata.timestamp.isoformat(),
            }, f, indent=2)

        log.info("Saved checkpoint to %s", checkpoint_dir)

    def _save_model_safe(self, state_dict: dict[str, torch.Tensor], path: Path) -> None:
        """Save model state dict safely with safetensors."""
        try:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in state_dict.items()}, path)
        except ImportError:
            # Fallback to torch.save
            torch.save(state_dict, path.with_suffix(".pt"))

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints to stay within save_total_limit."""
        if self.rank != 0:
            return

        while len(self._checkpoints) > self.config.save_total_limit:
            # Remove oldest checkpoint
            step, path = self._checkpoints.pop(0)
            if path.exists() and path != self._best_checkpoint_path:
                shutil.rmtree(path)
                log.info("Pruned checkpoint at step %d", step)

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        path: str | Path | None = None,
    ) -> CheckpointState | None:
        """Load a training checkpoint.

        Args:
            model: Model to load into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.
            path: Path to checkpoint. If None, uses resume_from_checkpoint config.

        Returns:
            CheckpointState with training state, or None if no checkpoint found.
        """
        if path is None:
            path = self.config.resume_from_checkpoint

        if path is None and self.config.auto_resume:
            # Try loading from latest
            latest_metadata = self.save_dir / "latest.json"
            if latest_metadata.exists():
                with latest_metadata.open() as f:
                    metadata = json.load(f)
                path = metadata.get("latest_checkpoint")
                if path and Path(path).exists():
                    path = Path(path)

        if path is None or not Path(path).exists():
            log.info("No checkpoint found at %s", path)
            return None

        checkpoint_dir = Path(path)

        # Load metadata
        metadata_file = checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with metadata_file.open() as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Load model state
        model_file = checkpoint_dir / "model.safetensors"
        if not model_file.exists():
            model_file = checkpoint_dir / "model.pt"

        if model_file.exists():
            state_dict = self._load_model_safe(model_file)
            if self._is_fsdp_model(model):
                self._load_fsdp_state(model, state_dict)
            else:
                model.load_state_dict(state_dict, strict=True)

        # Load optimizer state
        if optimizer is not None:
            optimizer_file = checkpoint_dir / "optimizer.pt"
            if optimizer_file.exists():
                optimizer_state = torch.load(optimizer_file, map_location="cpu")
                optimizer.load_state_dict(optimizer_state)

        # Load scheduler state
        if scheduler is not None:
            scheduler_file = checkpoint_dir / "scheduler.pt"
            if scheduler_file.exists():
                scheduler_state = torch.load(scheduler_file, map_location="cpu")
                scheduler.load_state_dict(scheduler_state)

        # Load RNG state
        rng_state = {}
        rng_file = checkpoint_dir / "rng_state.pt"
        if rng_file.exists():
            rng_state = torch.load(rng_file)

        checkpoint_state = CheckpointState(
            step=metadata.get("step", 0),
            epoch=metadata.get("epoch", 0),
            global_batch=metadata.get("global_batch", 0),
            optimizer_state=optimizer.state_dict() if optimizer else {},
            scheduler_state=scheduler.state_dict() if scheduler else None,
            rng_state=rng_state,
            metadata=CheckpointMetadata(
                step=metadata.get("step", 0),
                epoch=metadata.get("epoch", 0),
                global_batch=metadata.get("global_batch", 0),
                loss=metadata.get("loss", 0.0),
                eval_loss=metadata.get("eval_loss"),
                metrics=metadata.get("metrics", {}),
                is_best=metadata.get("is_best", False),
            ),
        )

        log.info("Loaded checkpoint from %s at step %d", checkpoint_dir, checkpoint_state.step)
        return checkpoint_state

    def _load_model_safe(self, path: Path) -> dict[str, torch.Tensor]:
        """Load model state dict safely from safetensors or pt."""
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                return load_file(path)
            except ImportError as e:
                raise ImportError("safetensors is required to load .safetensors files") from e
        else:
            result = torch.load(path, map_location="cpu")
            return result  # type: ignore[no-any-return]

    def _load_fsdp_state(self, model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
        """Load state dict into FSDP2 model."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch convention
        )
        from torch.distributed.fsdp import StateDictType

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state_dict, strict=True)

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint, if available."""
        if not self._checkpoints:
            return None
        return self._checkpoints[-1][1]

    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint, if available."""
        return self._best_checkpoint_path

    def cleanup(self) -> None:
        """Clean up temporary checkpoint artifacts."""
        if self.rank != 0:
            return

        # Remove any temporary checkpoints
        for path in (self.save_dir / "checkpoints").glob("tmp-*"):
            if path.is_dir():
                shutil.rmtree(path)
