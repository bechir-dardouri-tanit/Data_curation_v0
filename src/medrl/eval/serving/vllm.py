"""Serving deployment plans.

Policy and judge compete for the same two GPUs, so which process owns them *when* is a
first-class planning problem, not a launch detail. A plan is computed as pure data and
rendered to a ``vllm serve`` command line; nothing here imports vLLM, so the deployment
arithmetic (GPU budget, phases, ports) is unit-testable on a CPU box.

Three patterns, in order of preference:

``SEQUENTIAL``  one process at a time: the policy generates everything to disk, exits, the
                judge loads and grades. No contention, resumable at the grading boundary,
                and the judge tier can be swapped without regenerating anything.
``SPLIT_GPU``   policy and judge each own one GPU. Fastest iteration, at the cost of tp=1
                (which caps ``max_model_len``) on the policy.
``SLEEP_WAKE``  one vLLM process, weights offloaded between phases; cheapest swap when
                alternating many small rounds.
"""

from __future__ import annotations

import socket
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medrl.core.config import EvalConfig, JudgeConfig, ModelConfig, ServingPattern

Role = Literal["policy", "judge"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServePhase(BaseModel):
    """One process occupancy of the GPUs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    model: ModelConfig
    gpu_ids: tuple[int, ...]
    tensor_parallel: int
    max_model_len: int
    gpu_memory_utilization: float = Field(ge=0.1, le=1.0)
    port: int = Field(ge=1024, le=65535)
    # Text-only checkpoints need no vision profiling; keeps KV cache budget for text.
    language_model_only: bool = True
    reasoning_parser: str | None = "qwen3"

    @model_validator(mode="after")
    def _tp_matches_gpus(self) -> Self:
        if self.tensor_parallel != len(self.gpu_ids):
            raise ValueError(
                f"{self.role}: tensor_parallel={self.tensor_parallel} but "
                f"gpu_ids={self.gpu_ids}"
            )
        return self

    def command(self) -> list[str]:
        """Render the ``vllm serve`` argv. Kept explicit: it IS the deployment."""
        argv = [
            "vllm", "serve", self.model.ref,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.model.dtype if self.model.dtype != "auto" else "auto",
        ]
        if self.language_model_only:
            argv.append("--language-model-only")
        if self.reasoning_parser:
            argv.extend(["--reasoning-parser", self.reasoning_parser])
        return argv

    def env(self) -> dict[str, str]:
        """GPU placement for the subprocess. This is the *only* placement mechanism:
        ``command()`` deliberately emits no device-selection flag, so a runner applying
        both could not double-restrict an already-visible set."""
        return {"CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpu_ids))}


class DeploymentPlan(BaseModel):
    """All phases of one eval run, in order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: ServingPattern
    phases: tuple[ServePhase, ...]

    @property
    def policy(self) -> ServePhase:
        return next(p for p in self.phases if p.role == "policy")

    def render(self) -> str:
        lines = [f"# deployment: {self.pattern.value}"]
        for i, phase in enumerate(self.phases, 1):
            lines.append(f"# phase {i}: {phase.role} on GPUs {list(phase.gpu_ids)}")
            lines.append(" ".join(phase.command()))
        return "\n".join(lines)


def plan_deployment(
    config: EvalConfig,
    *,
    policy_port: int | None = None,
    judge_port: int | None = None,
) -> DeploymentPlan:
    """Compute the serving plan for an eval config from the cluster preset alone."""
    cluster = config.cluster
    all_gpus = tuple(range(cluster.num_gpus))
    policy_len = min(cluster.max_model_len, config.thinking.total_budget + 8192)

    policy = ServePhase(
        role="policy",
        model=config.model,
        gpu_ids=all_gpus if cluster.tensor_parallel > 1 else all_gpus[:1],
        tensor_parallel=max(1, cluster.tensor_parallel if cluster.num_gpus > 1 else 1),
        max_model_len=policy_len,
        gpu_memory_utilization=cluster.gpu_memory_utilization,
        port=policy_port or free_port(),
    )

    judge_cfg: JudgeConfig | None = config.judge
    if judge_cfg is None:
        return DeploymentPlan(pattern=ServingPattern.SEQUENTIAL, phases=(policy,))

    judge = ServePhase(
        role="judge",
        model=judge_cfg.model,
        gpu_ids=all_gpus[:1],
        tensor_parallel=1,
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        port=judge_port or free_port(),
        reasoning_parser=None,
    )

    if config.serving is ServingPattern.SPLIT_GPU:
        # Policy concedes to one GPU so the judge can hold the other for the whole run.
        policy = policy.model_copy(
            update={
                "gpu_ids": all_gpus[-1:],
                "tensor_parallel": 1,
                "max_model_len": min(policy.max_model_len, 24576),
            }
        )
        return DeploymentPlan(pattern=config.serving, phases=(policy, judge))

    # SEQUENTIAL and SLEEP_WAKE share the same phase list; the difference is process
    # lifetime, handled by the runner (restart vs sleep/wake), not the command line.
    return DeploymentPlan(pattern=config.serving, phases=(policy, judge))
