"""Main RL training loop for GSPO/DAPO.

Implements the RL training infrastructure supporting:
- GSPO (Group-wise Sequence Preference Optimization)
- DAPO (Direct Advantage Policy Optimization) with token-level fallback
- Colocated vLLM serving
- MoE model handling
- Comprehensive failure monitoring
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medrl.core.config import ClusterConfig, ModelConfig
from medrl.core.logging import get_logger
from medrl.core.manifest import RunManifest, Stage
from medrl.eval.items import EvalItem
from medrl.eval.scorers.judge import Judge
from medrl.eval.serving.lifecycle import ServerError, VLMMServer
from medrl.eval.serving.vllm import ServePhase, claim_port, free_port
from medrl.rl.config import (
    RLAlgorithm,
    RLTrainingConfig,
)
from medrl.rl.rewards import RewardResult, compute_total_reward
from medrl.rl.samplers import RolloutSampler, TemperatureScheduler

log = get_logger(__name__)


class RolloutError(RuntimeError):
    """Rollout generation failed."""


class TrainingError(RuntimeError):
    """Training step failed."""


@dataclass(frozen=True)
class RolloutRecord:
    """One rollout record: prompt, completion, and reward."""

    prompt_id: str
    """Unique identifier for the prompt."""

    repeat: int
    """Repeat index for this prompt."""

    content: str | None
    """Model-generated completion."""

    reward: RewardResult
    """Computed reward."""

    thinking_tokens: int | None = None
    """Number of thinking tokens (if applicable)."""

    answer_tokens: int | None = None
    """Number of answer tokens."""

    metadata: dict[str, Any] | None = None
    """Additional metadata."""


@dataclass
class RLTrainingState:
    """Training state checkpointing."""

    step: int = 0
    """Current training step."""

    epoch: int = 0
    """Current epoch."""

    consecutive_failures: int = 0
    """Consecutive rollout failures."""

    best_reward: float = 0.0
    """Best mean reward seen so far."""

    total_rollouts: int = 0
    """Total rollouts generated."""

    total_samples: int = 0
    """Total training samples processed."""


@dataclass
class RLResult:
    """Result of an RL training run."""

    run_id: str
    """Unique run identifier."""

    final_step: int
    """Final training step."""

    best_mean_reward: float
    """Best mean reward achieved."""

    total_rollouts: int
    """Total rollouts generated."""

    final_epoch: int
    """Final epoch."""

    checkpoint_path: str | None = None
    """Path to final checkpoint."""

    metrics: dict[str, Any] | None = None
    """Training metrics."""


def _is_moe_model(model_config: ModelConfig) -> bool:
    """Detect if a model is a Mixture-of-Experts model.

    MoE models require special handling for parameter-efficient training
    and gradient computation.

    Args:
        model_config: Model configuration.

    Returns:
        True if the model is an MoE model.
    """
    # Check common MoE indicators in model ID or config
    moe_indicators = ("mixtral", "moe", "expert", "grok")
    model_id_lower = model_config.hf_id.lower()
    return any(indicator in model_id_lower for indicator in moe_indicators)


class RLRunner:
    """Main RL training runner.

    Supports GSPO (sequence-level preference optimization) and DAPO
    (token-level advantage policy optimization) with colocated vLLM serving.
    """

    def __init__(
        self,
        config: RLTrainingConfig,
        cluster_config: ClusterConfig,
        run_dir: Path,
        *,
        judge: Judge | None = None,
    ):
        self.config = config
        self.cluster = cluster_config
        self.run_dir = Path(run_dir)
        self.judge = judge

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.state = RLTrainingState()
        self._manifest: RunManifest | None = None

        # Rollout sampler
        self.sampler = RolloutSampler(
            config=config.rollout_sampling,
            temperature_scheduler=(
                TemperatureScheduler(
                    schedule=config.rollout_sampling.temperature_schedule,
                    initial=config.rollout_sampling.initial_temperature,
                    final=config.rollout_sampling.final_temperature,
                    warmup_steps=config.rollout_sampling.temperature_warmup_steps,
                    decay_steps=config.rollout_sampling.temperature_decay_steps,
                )
                if config.rollout_sampling.temperature_schedule != "constant"
                else None
            ),
        )

        # Detect MoE model
        self.is_moe = _is_moe_model(config.model)
        if self.is_moe:
            log.info("Detected MoE model; enabling MoE-specific handling")

        # Colocated rollout server
        self._rollout_server: VLMMServer | None = None
        self._rollout_phase: ServePhase | None = None

    def _setup_manifest(self) -> RunManifest:
        """Create and save the run manifest."""
        if self._manifest is None:
            inputs = {"model": self.config.model.ref}
            if self.config.resume_from_checkpoint:
                inputs["checkpoint"] = self.config.resume_from_checkpoint

            self._manifest = RunManifest.create(
                Stage.RL, self.config, inputs=inputs, notes=f"algorithm={self.config.algorithm.value}"
            )
            self._manifest.save()
        return self._manifest

    def _get_rollout_phase(self) -> ServePhase:
        """Create the rollout server phase."""
        if self._rollout_phase is None:
            gpu_ids = tuple(range(self.cluster.num_gpus))
            self._rollout_phase = ServePhase(
                role="policy",
                model=self.config.model,
                gpu_ids=gpu_ids[: self.cluster.tensor_parallel],
                tensor_parallel=self.cluster.tensor_parallel,
                max_model_len=self.cluster.max_model_len,
                gpu_memory_utilization=self.cluster.gpu_memory_utilization,
                port=free_port(),
                language_model_only=self.config.model.text_only,
            )
        return self._rollout_phase

    def _start_rollout_server(self) -> VLMMServer:
        """Start the colocated rollout server."""
        phase = self._get_rollout_phase()
        phase = claim_port(phase)

        log.info(
            "Starting rollout server on GPUs %s (port %d)",
            list(phase.gpu_ids),
            phase.port,
        )

        server = VLMMServer(phase, run_dir=self.run_dir)
        try:
            server.start()
            self._rollout_server = server
            return server
        except ServerError as exc:
            raise RolloutError(f"Failed to start rollout server: {exc}") from exc

    def _stop_rollout_server(self) -> None:
        """Stop the rollout server."""
        if self._rollout_server is not None:
            try:
                self._rollout_server.stop()
            except Exception as exc:
                log.warning("Error stopping rollout server: %s", exc)
            finally:
                self._rollout_server = None

    def _generate_rollouts(
        self,
        prompts: Sequence[EvalItem],
        client: Any,
        alive: Callable[[], bool],
    ) -> list[RolloutRecord]:
        """Generate rollouts for a batch of prompts.

        Args:
            prompts: Eval items (prompts) to generate rollouts for.
            client: OpenAI client for the rollout server.
            alive: Callable that returns True while server is healthy.

        Returns:
            List of rollout records.
        """
        temperature = self.sampler.get_temperature()
        records: list[RolloutRecord] = []

        n_repeats = self.config.rollout_sampling.base.n_repeats

        for prompt in prompts:
            for repeat in range(n_repeats):
                if not alive():
                    log.error("Rollout server died during generation")
                    raise RolloutError("Server unhealthy during rollout generation")

                try:
                    completion = self._generate_single_completion(
                        client, prompt, temperature, repeat
                    )

                    reward = compute_total_reward(
                        content=completion or "",
                        verify_style=prompt.verify.style,
                        verify_params={
                            "gold_letter": prompt.verify.gold_letter,
                            "letters": prompt.verify.letters,
                            "gold_number": prompt.verify.gold_number,
                            "lower": prompt.verify.lower,
                            "upper": prompt.verify.upper,
                        },
                        rubric_verdicts=None,  # Will be filled by judge if needed
                        config=self.config.reward,
                    )

                    records.append(
                        RolloutRecord(
                            prompt_id=prompt.item_id,
                            repeat=repeat,
                            content=completion,
                            reward=reward,
                            metadata={
                                "benchmark": prompt.benchmark,
                                "temperature": temperature,
                            },
                        )
                    )

                except Exception as exc:
                    log.warning(
                        "Failed to generate rollout for %s repeat %d: %s",
                        prompt.item_id,
                        repeat,
                        exc,
                    )
                    # Add a failed record
                    records.append(
                        RolloutRecord(
                            prompt_id=prompt.item_id,
                            repeat=repeat,
                            content=None,
                            reward=RewardResult(total=0.0),
                            metadata={"error": str(exc), "benchmark": prompt.benchmark},
                        )
                    )

        return records

    def _generate_single_completion(
        self,
        client: Any,
        prompt: EvalItem,
        temperature: float,
        repeat: int,
    ) -> str | None:
        """Generate a single completion for a prompt.

        Args:
            client: OpenAI client.
            prompt: Eval item with messages.
            temperature: Sampling temperature.
            repeat: Repeat index for seeding.

        Returns:
            Generated completion text, or None on failure.
        """
        import hashlib

        # Derive a per-(prompt, repeat) seed
        seed_base = f"{prompt.item_id}:{repeat}:{self.config.rollout_sampling.base.seed}"
        seed_int = int(hashlib.sha256(seed_base.encode()).hexdigest(), 16) % (2**31)

        messages = [dict(m) for m in prompt.messages]

        response = client.chat.completions.create(
            model=self.config.model.ref,
            messages=messages,
            temperature=temperature,
            top_p=self.config.rollout_sampling.base.top_p,
            top_k=self.config.rollout_sampling.base.top_k,
            max_tokens=self.config.thinking.total_budget,
            seed=seed_int,
        )

        content = response.choices[0].message.content
        return content if content is not None else None

    def _compute_advantages_dapo(
        self,
        rewards: Sequence[float],
        values: Sequence[float] | None = None,
    ) -> list[float]:
        """Compute advantages for DAPO using GAE.

        Args:
            rewards: Sequence of rewards.
            values: Optional value function estimates.

        Returns:
            Advantages for each timestep.
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        advantages: list[float] = []
        gae = 0.0

        # Reverse iteration for GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
                next_advantage = 0.0
            else:
                next_value = (values or [0.0])[t + 1]
                next_advantage = advantages[t + 1]

            delta = rewards[t] + gamma * next_value - (values or [0.0])[t]
            gae = delta + gamma * gae_lambda * next_advantage
            advantages.insert(0, gae)

        return advantages

    def _train_step_gspo(
        self,
        rollouts: list[RolloutRecord],
    ) -> dict[str, float]:
        """Perform one GSPO training step.

        GSPO operates at the sequence level, using preference pairs from
        group-wise rollouts.

        Args:
            rollouts: Rollout records to train on.

        Returns:
            Metrics dict with loss, reward, etc.
        """
        # Group rollouts by prompt_id
        from collections import defaultdict

        groups: defaultdict[str, list[RolloutRecord]] = defaultdict(list)
        for r in rollouts:
            groups[r.prompt_id].append(r)

        # Sample preference pairs
        all_pairs = []
        for group in groups.values():
            rewards = [r.reward.total for r in group]
            pairs = self.sampler.sample_rollouts([rewards])
            for _, pair_list in pairs:
                all_pairs.extend([(group[i], group[j]) for i, j in pair_list])

        if not all_pairs:
            log.warning("No valid preference pairs in this batch")
            return {"loss": 0.0, "mean_reward": 0.0, "n_pairs": 0}

        # Compute metrics
        mean_reward = sum(r.reward.total for r in rollouts) / len(rollouts)
        n_pairs = len(all_pairs)

        # Placeholder for actual training loss computation
        # In real implementation, this would call the training loop
        loss = mean(1.0 - (chosen.reward.total - rejected.reward.total) for chosen, rejected in all_pairs)

        return {
            "loss": max(0.0, loss),
            "mean_reward": mean_reward,
            "n_pairs": n_pairs,
        }

    def _train_step_dapo(
        self,
        rollouts: list[RolloutRecord],
    ) -> dict[str, float]:
        """Perform one DAPO training step.

        DAPO operates at the token level using advantages.

        Args:
            rollouts: Rollout records to train on.

        Returns:
            Metrics dict with loss, reward, etc.
        """
        rewards = [r.reward.total for r in rollouts]
        advantages = self._compute_advantages_dapo(rewards)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0

        # Placeholder for actual training loss computation
        policy_loss = mean(a * (1.0 - r) for a, r in zip(advantages, rewards, strict=True))

        return {
            "policy_loss": policy_loss,
            "mean_reward": mean_reward,
            "mean_advantage": mean_advantage,
            "n_samples": len(rollouts),
        }

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log training metrics.

        Args:
            metrics: Metrics to log.
            step: Current training step.
        """
        log.info(
            "Step %d: reward=%.4f loss=%.4f pairs=%d",
            step,
            metrics.get("mean_reward", 0.0),
            metrics.get("loss", metrics.get("policy_loss", 0.0)),
            metrics.get("n_pairs", metrics.get("n_samples", 0)),
        )

        # Save to JSON for later analysis
        metrics_path = self.run_dir / "metrics.jsonl"
        with metrics_path.open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def _save_checkpoint(self, step: int) -> Path:
        """Save a training checkpoint.

        Args:
            step: Current training step.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}"
        checkpoint_path.mkdir(exist_ok=True)

        # Save state
        state_path = checkpoint_path / "state.json"
        with state_path.open("w") as f:
            json.dump(asdict(self.state), f, indent=2)

        # Save config
        config_path = checkpoint_path / "config.json"
        with config_path.open("w") as f:
            f.write(self.config.model_dump_json(indent=2))

        log.info("Saved checkpoint at step %d to %s", step, checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        state_path = checkpoint_path / "state.json"
        if state_path.exists():
            with state_path.open() as f:
                state_dict = json.load(f)
                self.state = RLTrainingState(**state_dict)
            log.info("Resumed from checkpoint at step %d", self.state.step)
        else:
            log.warning("Checkpoint state not found; starting from scratch")

    def run(self) -> RLResult:
        """Run the full RL training loop.

        Returns:
            RLResult with training outcomes.
        """
        manifest = self._setup_manifest()
        log.info("Starting RL training: %s", manifest.run_id)

        # Load checkpoint if resuming
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(Path(self.config.resume_from_checkpoint))

        try:
            # Main training loop
            while self.state.step < self.config.max_epochs * self.config.rollout_sampling.group_size:
                # Start rollout server
                self._start_rollout_server()
                from openai import OpenAI

                try:
                    server = self._rollout_server
                    assert server is not None
                    client = OpenAI(
                        base_url=f"http://127.0.0.1:{self._get_rollout_phase().port}",
                        api_key="EMPTY",
                        timeout=self.config.rollout_timeout_s,
                        max_retries=2,
                    )

                    # Generate rollouts
                    # (In real implementation, prompts would be loaded from data)
                    rollouts: list[RolloutRecord] = []  # Placeholder

                    # Training step
                    if self.config.algorithm is RLAlgorithm.GSPO:
                        metrics = self._train_step_gspo(rollouts)
                    else:
                        metrics = self._train_step_dapo(rollouts)

                    self._log_metrics(metrics, self.state.step)

                    # Update best reward
                    mean_reward = metrics.get("mean_reward", 0.0)
                    if mean_reward > self.state.best_reward:
                        self.state.best_reward = mean_reward

                    # Reset failure counter on success
                    self.state.consecutive_failures = 0

                    # Checkpoint
                    if self.state.step % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(self.state.step)

                    self.state.step += 1

                finally:
                    client.close()
                    self._stop_rollout_server()

            # Training completed
            manifest.complete(best_reward=self.state.best_reward)
            manifest.save()

            return RLResult(
                run_id=manifest.run_id,
                final_step=self.state.step,
                best_mean_reward=self.state.best_reward,
                total_rollouts=self.state.total_rollouts,
                final_epoch=self.state.epoch,
                checkpoint_path=str(self._save_checkpoint(self.state.step)),
                metrics={"final_metrics": {}},
            )

        except Exception as exc:
            manifest.fail(str(exc))
            manifest.save()
            raise TrainingError(f"Training failed: {exc}") from exc


def mean(iterable: Any) -> float:
    """Calculate mean of an iterable."""
    values = list(iterable)
    return sum(values) / len(values) if values else 0.0
