"""Typed configuration schema.

Hydra composes YAML; Pydantic validates it. Every config object is frozen and hashable,
so ``cfg.fingerprint`` identifies a run's inputs exactly and the artifact store can treat
it as a cache key.

The important invariant is that nothing here hardcodes a world size. Scale portability
(2 GPUs today, 8+ later) lives entirely in :class:`ClusterConfig`, and every other stage
config is written in terms of *global* quantities (global batch in tokens, total repeats)
that the cluster preset resolves into per-device numbers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from medrl.core.hashing import hash_obj


class Frozen(BaseModel):
    """Immutable, strictly-validated config base."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    @property
    def fingerprint(self) -> str:
        """Stable content hash of this config subtree."""
        return hash_obj(self.model_dump(mode="json"))


# --------------------------------------------------------------------------------------
# Cluster / topology
# --------------------------------------------------------------------------------------


class TrainStrategy(StrEnum):
    """How to shard a training job.

    ``FSDP2`` is full-parameter training; on 2x80GB it does not fit a 9B (params+grads+
    AdamW alone is ~160 GB), hence ``LORA`` and ``FSDP2_OFFLOAD`` as the small-topology
    escape hatches.
    """

    FSDP2 = "fsdp2"
    FSDP2_OFFLOAD = "fsdp2_offload"
    LORA = "lora"


class RolloutPlacement(StrEnum):
    """Where the RL rollout engine lives relative to the trainer."""

    COLOCATE = "colocate"  # share GPUs, time-sliced (needs headroom)
    SEQUENTIAL = "sequential"  # sleep/wake the engine between phases (small topologies)


class ClusterConfig(Frozen):
    """The single axis that changes between hardware topologies."""

    name: str
    num_gpus: Annotated[int, Field(ge=1)]
    gpu_memory_gb: Annotated[int, Field(ge=1)] = 80
    num_nodes: Annotated[int, Field(ge=1)] = 1

    # training
    train_strategy: TrainStrategy = TrainStrategy.FSDP2
    activation_checkpointing: bool = True
    optimizer_offload: bool = False
    param_offload: bool = False
    micro_batch_size: Annotated[int, Field(ge=1)] = 1
    # Global batch is expressed in tokens so it is topology-invariant; grad-accum is derived.
    global_batch_tokens: Annotated[int, Field(ge=1024)] = 262_144

    # inference / rollout
    tensor_parallel: Annotated[int, Field(ge=1)] = 1
    gpu_memory_utilization: Annotated[float, Field(gt=0, le=1)] = 0.85
    rollout_placement: RolloutPlacement = RolloutPlacement.SEQUENTIAL
    max_model_len: Annotated[int, Field(ge=2048)] = 40_960
    max_num_seqs: Annotated[int, Field(ge=1)] | None = None  # For Mamba SSM models with limited cache blocks

    @model_validator(mode="after")
    def _check_topology(self) -> Self:
        if self.tensor_parallel > self.num_gpus:
            raise ValueError(
                f"tensor_parallel={self.tensor_parallel} exceeds num_gpus={self.num_gpus}"
            )
        if self.num_gpus % self.tensor_parallel:
            raise ValueError(
                f"num_gpus={self.num_gpus} must be divisible by "
                f"tensor_parallel={self.tensor_parallel}"
            )
        if self.rollout_placement is RolloutPlacement.COLOCATE and self.num_gpus < 4:
            raise ValueError(
                "colocated rollout needs >=4 GPUs; use rollout_placement=sequential "
                "(vLLM sleep/wake) on small topologies"
            )
        return self

    @property
    def total_gpus(self) -> int:
        return self.num_gpus * self.num_nodes

    def grad_accum_steps(self, seq_len: int) -> int:
        """Derive gradient accumulation from the topology-invariant global token budget."""
        per_step = self.micro_batch_size * seq_len * self.total_gpus
        return max(1, round(self.global_batch_tokens / per_step))


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class ModelConfig(Frozen):
    """A model to load, either from the Hub or a local converted checkpoint."""

    hf_id: str
    revision: str | None = None
    path: str | None = None  # local path wins over hf_id when set
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    # True once the vision tower and MTP head have been stripped (Qwen3_5ForCausalLM).
    text_only: bool = False
    trust_remote_code: bool = False

    @property
    def ref(self) -> str:
        """What to hand a loader."""
        return self.path or self.hf_id


# --------------------------------------------------------------------------------------
# Decoding / thinking
# --------------------------------------------------------------------------------------


class SamplingConfig(Frozen):
    """Decoding parameters.

    Greedy decoding is deliberately not the default: for reasoning models it
    underestimates capability and produces pathological ties. We sample and report a mean
    over repeats with bootstrap CIs instead.
    """

    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.6
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 0.95
    top_k: Annotated[int, Field(ge=-1)] = 40
    n_repeats: Annotated[int, Field(ge=1)] = 8
    seed: int = 0
    max_tokens: Annotated[int, Field(ge=1)] = 10_240
    presence_penalty: float = 0.0

    @model_validator(mode="after")
    def _greedy_needs_one_repeat(self) -> Self:
        if self.temperature == 0.0 and self.n_repeats > 1:
            raise ValueError(
                "temperature=0 with n_repeats>1 wastes compute: every sample is identical. "
                "Use temperature>0 for repeats, or n_repeats=1 for greedy."
            )
        return self


def _yaml_bool_to_mode(value: object) -> object:
    """YAML 1.1 spells ``on``/``off`` as booleans; accept both spellings.

    Without this, ``thinking: {mode: on}`` silently arrives as ``True`` and validation
    fails with a confusing message. Every enum used in YAML gets the same hazard, so it
    is fixed at the field, not at each author.
    """
    if value is True:
        return "on"
    if value is False:
        return "off"
    return value


class ThinkingMode(StrEnum):
    ON = "on"
    OFF = "off"


class ThinkingConfig(Frozen):
    """Reasoning-budget control.

    Budgets are fixed and identical across every compared model so that length differences
    cannot masquerade as accuracy differences.
    """

    mode: Annotated[ThinkingMode, BeforeValidator(_yaml_bool_to_mode)] = ThinkingMode.ON
    think_budget: Annotated[int, Field(ge=0)] = 8192
    answer_budget: Annotated[int, Field(ge=1)] = 2048
    # Qwen only voluntarily emits the opening <think> tag a minority of the time; prefill it.
    prefill_think: bool = True
    # Count a response that never closed </think> as wrong, and report the rate separately.
    strict_incomplete: bool = True

    @model_validator(mode="after")
    def _off_means_no_budget(self) -> Self:
        if self.mode is ThinkingMode.OFF and self.think_budget:
            raise ValueError("thinking.mode=off requires think_budget=0")
        return self

    @property
    def total_budget(self) -> int:
        return self.think_budget + self.answer_budget


# --------------------------------------------------------------------------------------
# Benchmarks / judging
# --------------------------------------------------------------------------------------


class Grade(StrEnum):
    """What a benchmark is allowed to be used for.

    ``DECISION`` benchmarks gate promotion; ``GUARDRAIL`` benchmarks may only block it
    (catching catastrophic forgetting); ``REPORTING`` benchmarks are published but never
    steer a decision, because their minimum detectable effect is too large to be trusted.
    """

    DECISION = "decision"
    REPORTING = "reporting"
    GUARDRAIL = "guardrail"


class Language(StrEnum):
    EN = "en"
    FR = "fr"
    MULTI = "multi"


class BenchmarkConfig(Frozen):
    name: str
    loader: str  # registry key in medrl.eval.tasks
    grade: Grade = Grade.REPORTING
    language: Language = Language.EN
    hf_id: str | None = None
    split: str = "test"
    subset: str | None = None
    limit: int | None = None  # cap items (fast ablation subsets)
    requires_judge: bool = False
    # Overrides; fall back to the run-level defaults when unset.
    sampling: SamplingConfig | None = None
    thinking: ThinkingConfig | None = None


class JudgeTier(StrEnum):
    """Judges are tiered because grader strength systematically shifts rubric scores.

    A weak grader inflates HealthBench-Hard, so ``FAST`` is for ranking ablation arms only
    and headline numbers must come from ``STRONG``.
    """

    FAST = "fast"
    STRONG = "strong"


class JudgeConfig(Frozen):
    tier: JudgeTier = JudgeTier.FAST
    model: ModelConfig
    backend: Literal["vllm", "openai"] = "vllm"
    base_url: str | None = None
    max_concurrency: Annotated[int, Field(ge=1)] = 64
    # Rubric-grading robustness knobs.
    n_consistency: Annotated[int, Field(ge=1)] = 1
    position_swap: bool = True
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0
    # Bounds the judge's generation: the verdict is a tiny JSON object, and an
    # unbounded request lets a thinking-capable judge reason until truncation
    # instead of answering (observed live: every rubric call failed on it).
    max_tokens: Annotated[int, Field(ge=16)] = 512

    @model_validator(mode="after")
    def _reporting_needs_strong(self) -> Self:
        if self.tier is JudgeTier.STRONG and self.n_consistency < 3:
            raise ValueError(
                "strong-tier judging is used for reported numbers and requires "
                "n_consistency>=3 for a stable met-rate"
            )
        return self


# --------------------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------------------


class TrackingConfig(Frozen):
    enabled: bool = True
    backend: Literal["wandb", "none"] = "wandb"
    project: str = "medrl"
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    log_samples: bool = True


# --------------------------------------------------------------------------------------
# Top-level eval config
# --------------------------------------------------------------------------------------


class ServingPattern(StrEnum):
    """How policy and judge share the GPUs.

    ``SEQUENTIAL`` generates every completion to disk, tears the policy down, then grades.
    It is the default because it removes contention entirely and makes a run resumable at
    the grading boundary.
    """

    SEQUENTIAL = "sequential"
    SPLIT_GPU = "split_gpu"
    SLEEP_WAKE = "sleep_wake"


class EvalConfig(Frozen):
    model: ModelConfig
    cluster: ClusterConfig
    benchmarks: tuple[BenchmarkConfig, ...]
    sampling: SamplingConfig = SamplingConfig()
    thinking: ThinkingConfig = ThinkingConfig()
    judge: JudgeConfig | None = None
    serving: ServingPattern = ServingPattern.SEQUENTIAL
    tracking: TrackingConfig = TrackingConfig()
    output_dir: str | None = None
    # Fail the run if MCQA answer extraction fails more often than this. A high rate is a
    # harness bug, not a model result.
    max_extraction_fail_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02

    @model_validator(mode="after")
    def _judge_present_when_needed(self) -> Self:
        needs = [b.name for b in self.benchmarks if b.requires_judge]
        if needs and self.judge is None:
            raise ValueError(f"benchmarks {needs} require a judge but judge is unset")
        return self

    @model_validator(mode="after")
    def _unique_benchmarks(self) -> Self:
        names = [b.name for b in self.benchmarks]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate benchmark names: {sorted(dupes)}")
        return self

    def resolved_sampling(self, bench: BenchmarkConfig) -> SamplingConfig:
        return bench.sampling or self.sampling

    def resolved_thinking(self, bench: BenchmarkConfig) -> ThinkingConfig:
        return bench.thinking or self.thinking


def validate_dict(cls: type[BaseModel], raw: Any) -> Any:
    """Validate a plain dict / OmegaConf container into a config model.

    Kept as a helper so the Hydra boundary is one line and every entrypoint validates the
    same way.
    """
    if hasattr(raw, "_content"):  # OmegaConf container
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(raw, resolve=True)
    return cls.model_validate(raw)
