"""Main SFT training loop.

Implements the supervised fine-tuning training loop with:
- FSDP2 distributed training
- Sequence packing
- Assistant-only loss masking
- Thinking mode support
- Gradient clipping
- W&B integration
- Checkpoint management
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from medrl.core.logging import get_logger
from medrl.core.paths import runs_dir
from medrl.tracking.wandb import TrackingSink, make_sink
from medrl.train.sft.checkpoint import CheckpointManager
from medrl.train.sft.config import SFTConfig
from medrl.train.sft.data import SFTCollator, SFTDataset, SFTExample

log = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    loss: float
    learning_rate: float
    epoch: float
    step: int
    grad_norm: float | None = None
    tokens_per_second: float | None = None
    samples_per_second: float | None = None
    gpu_memory_utilization: float | None = None


@dataclass
class EvalMetrics:
    """Metrics collected during evaluation."""

    loss: float
    perplexity: float
    step: int
    epoch: int


class SFTRunner:
    """Main SFT training runner.

    Coordinates all aspects of SFT training:
    - Model and tokenizer loading
    - Data loading and preprocessing
    - Training loop with FSDP2
    - Checkpoint management
    - Evaluation
    - Tracking (W&B)
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self._rank = 0
        self._world_size = 1
        self._is_distributed = False

        # Setup directories
        self._setup_directories()

        # Initialize tracking
        self.tracker: TrackingSink = make_sink(
            config.tracking,
            job_type="sft",
            run_name=config.run_name or self._generate_run_name(),
        )

        # Will be initialized later
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
        self.checkpoint_manager: CheckpointManager | None = None

    def _setup_directories(self) -> None:
        """Setup output directories."""
        if self.config.output_dir:
            self.output_dir = Path(self.config.output_dir)
        else:
            timestamp = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
            run_name = self.config.run_name or timestamp
            self.output_dir = runs_dir() / "sft" / run_name

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"

        # Create directories (rank 0 only in distributed)
        if not self._is_distributed or self._rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _generate_run_name(self) -> str:
        """Generate a run name from config."""
        timestamp = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model.hf_id.split("/")[-1][:20]
        return f"{model_name}_sft_{timestamp}"

    def setup_distributed(self) -> None:
        """Initialize distributed training."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self._rank = int(os.environ["RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])
            self._is_distributed = self._world_size > 1

        if self._is_distributed:
            # Initialize process group
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            torch.cuda.set_device(self._rank)

            log.info("Initialized distributed training: rank %d, world_size %d", self._rank, self._world_size)

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
        """Load model and tokenizer for training.

        Returns:
            Tuple of (model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading model: %s", self.config.model.ref)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.ref,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=True,
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.ref,
            revision=self.config.model.revision,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            attn_implementation=self.config.model.attn_implementation,
        )

        # Resize embeddings if tokenizer size changed
        if model.config.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
            log.info("Resized embeddings to %d", len(tokenizer))

        # Handle text-only models
        if self.config.model.text_only:
            # Strip vision tower if present
            if hasattr(model, "vision_model"):
                model.vision_model = None  # type: ignore[assignment]
            if hasattr(model, "visual"):
                model.visual = None  # type: ignore[assignment]

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def setup_fsdp(self, model: PreTrainedModel) -> PreTrainedModel:
        """Setup FSDP2 wrapping for the model.

        Args:
            model: Base model to wrap.

        Returns:
            FSDP-wrapped model if enabled, otherwise original model.
        """
        if self.config.cluster.train_strategy.value not in ("fsdp2", "fsdp2_offload"):
            return model


        from torch.distributed.fsdp import (
            CPUOffload,
            ShardingStrategy,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch convention
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Auto-wrap policy
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        # Use transformer_auto_wrap_policy with transformer layer class
        # The policy function expects: module, recurse, nonwrapped_numel, transformer_layer_cls
        def auto_wrap_policy(
            module: torch.nn.Module,
            recurse: bool,
            nonwrapped_numel: int,
        ) -> bool:
            return transformer_auto_wrap_policy(
                module=module,
                recurse=recurse,
                nonwrapped_numel=nonwrapped_numel,
                transformer_layer_cls={LlamaDecoderLayer},
            )

        # Offloading
        cpu_offload = CPUOffload(offload_params=True) if self.config.effective_param_offload else None

        # Wrap model
        model = FSDP(  # type: ignore[assignment]
            model,
            sharding_strategy={
                "full_shard": ShardingStrategy.FULL_SHARD,
                "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
            }[self.config.fsdp_sharding_strategy],
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            device_id=self._rank if torch.cuda.is_available() else None,
        )

        log.info("Wrapped model with FSDP2 (strategy: %s)", self.config.fsdp_sharding_strategy)
        return model

    def setup_training(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> Trainer:
        """Setup Hugging Face Trainer with our configuration.

        Args:
            model: Model to train.
            tokenizer: Tokenizer for processing.

        Returns:
            Configured Trainer instance.
        """
        # Create collator
        collator = SFTCollator(
            tokenizer=tokenizer,
            max_seq_len=self.config.data.max_seq_len,
            packing=self.config.data.packing,
            packing_separator=self.config.data.packing_separator,
            loss_config=self.config.loss,
            thinking_config=self.config.thinking,
        )

        # Create datasets
        train_dataset = SFTDataset(
            data_path=self.config.data.train_path,
            conversation_format=self.config.data.conversation_format,
            data_type=self.config.data.dataset_type,
            filter_fn=None,  # Could add custom filter
            min_response_length=self.config.data.min_response_length,
            max_response_length=self.config.data.max_response_length,
            seed=self.config.data.seed,
        )

        eval_dataset = None
        if self.config.data.validation_path:
            eval_dataset = SFTDataset(
                data_path=self.config.data.validation_path,
                conversation_format=self.config.data.conversation_format,
                data_type=self.config.data.dataset_type,
                filter_fn=None,
                min_response_length=self.config.data.min_response_length,
                max_response_length=self.config.data.max_response_length,
                seed=self.config.data.seed,
            )

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config=self.config.checkpoint,
            save_dir=self.checkpoint_dir,
            is_best_enabled=self.config.checkpoint.load_best_model_at_end,
            rank=self._rank,
            world_size=self._world_size,
        )

        # Training arguments
        total_steps = self.config.max_steps or self.config.num_epochs * len(train_dataset)
        warmup_steps = int(total_steps * self.config.scheduler.warmup_ratio)
        if self.config.scheduler.warmup_steps is not None:
            warmup_steps = self.config.scheduler.warmup_steps

        grad_accum = self.config.cluster.grad_accum_steps(self.config.data.max_seq_len)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.config.run_name or self._generate_run_name(),

            # Batch size
            per_device_train_batch_size=self.config.cluster.micro_batch_size,
            per_device_eval_batch_size=self.config.cluster.micro_batch_size,
            gradient_accumulation_steps=grad_accum,

            # Training duration
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps or -1,

            # Optimization
            learning_rate=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.config.scheduler.type.value,
            optim="adamw_torch" if self.config.optimizer.type.value == "adamw" else self.config.optimizer.type.value,
            max_grad_norm=self.config.max_grad_norm,

            # Precision
            bf16=self.config.bf16,
            fp16=not self.config.bf16,

            # Checkpointing
            save_strategy="steps" if self.config.checkpoint.should_save_steps else "epoch",
            save_steps=self.config.checkpoint.save_steps or 0,
            save_total_limit=self.config.checkpoint.save_total_limit,
            save_on_each_node=False,

            # Evaluation
            eval_strategy="steps" if self.config.checkpoint.eval_steps else "no",
            eval_steps=self.config.checkpoint.eval_steps,
            eval_accumulation_steps=1,

            # Logging
            logging_steps=10,
            logging_first_step=True,

            # Distributed
            ddp_find_unused_parameters=False,
            fsdp="full_shard auto_wrap" if self.config.cluster.train_strategy.value == "fsdp2" else None,

            # Misc
            seed=self.config.seed,
            dataloader_num_workers=self.config.data.num_workers,
            tf32=True,
            report_to=["wandb"] if self.config.tracking.enabled and self.config.tracking.backend == "wandb" else [],
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )

        # Log config to tracker
        self._log_config()

        return trainer

    def _log_config(self) -> None:
        """Log training configuration to tracking backend."""
        config_dict = {
            "model_ref": self.config.model.ref,
            "model_dtype": self.config.model.dtype,
            "train_strategy": self.config.cluster.train_strategy.value,
            "global_batch_size": self.config.global_batch_size,
            "micro_batch_size": self.config.cluster.micro_batch_size,
            "gradient_accumulation_steps": self.config.cluster.grad_accum_steps(self.config.data.max_seq_len),
            "learning_rate": self.config.optimizer.lr,
            "weight_decay": self.config.optimizer.weight_decay,
            "warmup_ratio": self.config.scheduler.warmup_ratio,
            "max_grad_norm": self.config.max_grad_norm,
            "max_seq_len": self.config.data.max_seq_len,
            "packing_mode": self.config.data.packing.value,
            "num_epochs": self.config.num_epochs,
            "max_steps": self.config.max_steps,
            "bf16": self.config.bf16,
            "thinking_mode": self.config.thinking.mode.value if self.config.thinking else "disabled",
            "config_hash": self.config.fingerprint,
        }

        if self.config.tracking.enabled:
            self.tracker.log_config(config_dict)

    def train(self, trainer: Trainer) -> dict[str, Any]:
        """Run the training loop.

        Args:
            trainer: Configured Trainer instance.

        Returns:
            Training history with metrics.
        """
        log.info("Starting SFT training...")
        log.info("  Model: %s", self.config.model.ref)
        log.info("  Data: %s", self.config.data.train_path)
        log.info("  Epochs: %d, Max steps: %s", self.config.num_epochs, self.config.max_steps)
        log.info("  Batch size: global=%d, micro=%d, grad_accum=%d",
                self.config.global_batch_size,
                self.config.cluster.micro_batch_size,
                self.config.cluster.grad_accum_steps(self.config.data.max_seq_len))

        # Resume from checkpoint if configured
        resume_path = None
        if self.config.checkpoint.resume_from_checkpoint:
            resume_path = self.config.checkpoint.resume_from_checkpoint
        elif self.config.checkpoint.auto_resume and self.checkpoint_manager:
            latest = self.checkpoint_manager.get_latest_checkpoint()
            if latest:
                resume_path = str(latest)
                log.info("Auto-resuming from checkpoint: %s", resume_path)

        # Train
        result = trainer.train(resume_from_checkpoint=resume_path)

        log.info("Training completed. Final metrics: %s", result.metrics)

        # Finish tracking
        if self.config.tracking.enabled:
            self.tracker.finish()

        return result  # type: ignore[return-value]

    def evaluate(self, trainer: Trainer) -> dict[str, Any]:
        """Run evaluation on the validation set.

        Args:
            trainer: Trainer instance.

        Returns:
            Evaluation metrics.
        """
        if trainer.eval_dataset is None:
            log.warning("No validation dataset configured, skipping evaluation")
            return {}

        log.info("Running evaluation...")
        metrics = trainer.evaluate()

        # Log to tracker
        if self.config.tracking.enabled:
            eval_metrics = {
                f"eval_{k}": v for k, v in metrics.items()
            }
            step = metrics.get("step")
            self.tracker.log_metrics(eval_metrics, step=int(step) if step is not None else None)

        return metrics

    def save_model(self, trainer: Trainer, path: str | Path | None = None) -> Path:
        """Save the final trained model.

        Args:
            trainer: Trainer instance.
            path: Optional path to save to. Defaults to output_dir/final_model.

        Returns:
            Path to saved model.
        """
        path = self.output_dir / "final_model" if path is None else Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        trainer.save_model(str(path))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(path))

        log.info("Saved final model to %s", path)
        return path


def run_sft(config: SFTConfig) -> dict[str, Any]:
    """Convenience function to run SFT training end-to-end.

    Args:
        config: SFT configuration.

    Returns:
        Training result with metrics.
    """
    runner = SFTRunner(config)

    # Setup
    runner.setup_distributed()
    model, tokenizer = runner.load_model_and_tokenizer()

    # Wrap with FSDP if needed
    if config.cluster.train_strategy.value in ("fsdp2", "fsdp2_offload"):
        model = runner.setup_fsdp(model)

    # Setup training
    trainer = runner.setup_training(model, tokenizer)

    # Train
    result = runner.train(trainer)

    # Evaluate
    if config.data.validation_path:
        eval_result = runner.evaluate(trainer)
        result["eval"] = eval_result

    # Save final model
    runner.save_model(trainer)

    return result


# Custom data iterator for streaming large datasets
class StreamingDataIterator(Iterator[dict[str, torch.Tensor]]):
    """Iterator for streaming SFT data without loading everything into memory."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        collator: SFTCollator,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.collator = collator
        self.batch_size = batch_size
        self.seed = seed
        self._rng = random.Random(seed)

        # Detect file type
        if self.data_path.suffix == ".jsonl":
            self._file_type = "jsonl"
        elif self.data_path.suffix == ".json":
            self._file_type = "json"
        else:
            raise ValueError(f"Unsupported file type: {self.data_path.suffix}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Stream batches from the data file."""
        buffer: list[SFTExample] = []

        with self.data_path.open() as f:
            if self._file_type == "jsonl":
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        example = self._parse_record(record)
                        if example:
                            buffer.append(example)

                            if len(buffer) >= self.batch_size:
                                yield self.collator(buffer)
                                buffer = []
            else:  # json
                records = json.load(f)
                for record in records:
                    example = self._parse_record(record)
                    if example:
                        buffer.append(example)

                        if len(buffer) >= self.batch_size:
                            yield self.collator(buffer)
                            buffer = []

        # Final batch
        if buffer:
            yield self.collator(buffer)

    def _parse_record(self, record: dict[str, Any]) -> SFTExample | None:
        """Parse a record into an SFT example."""
        # Simple OpenAI format parsing
        messages = record.get("messages", [])
        response = record.get("response", "")

        if not messages or not response:
            return None

        return SFTExample(
            messages=messages,
            response=response,
            source=record.get("source", "streaming"),
            meta=record.get("meta", {}),
        )
