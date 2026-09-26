"""Data schemas for the medrl pipeline.

Canonical schemas for supervised fine-tuning, preference pairs, RL prompts, and
data provenance. All schemas are frozen pydantic dataclasses for immutability and
hashability -- they can be used as dict keys and cached.

Typical usage:
    >>> from medrl.data import SFTItem, PrefPair
    >>> item = SFTItem(messages=[{"role": "user", "content": "What is ibuprofen?"}],
    ...                response="Ibuprofen is a NSAID...")
    >>> pair = PrefPair(chosen=item, rejected=lower_quality_item)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Base Message Types
# =============================================================================


@dataclass(frozen=True)
class Message:
    """A single chat message.

    Attributes:
        role: The message role (system, user, assistant).
        content: The message text content.
        meta: Optional metadata for provenance tracking.

    Examples:
        >>> msg = Message(role="user", content="What is aspirin?")
        >>> msg
        Message(role='user', content='What is aspirin?', meta={})
    """

    role: Literal["system", "user", "assistant"]
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to OpenAI-style dict for API calls."""
        return {"role": self.role, "content": self.content}


# =============================================================================
# Supervised Fine-Tuning (SFT)
# =============================================================================


@dataclass(frozen=True)
class SFTItem:
    """A single supervised fine-tuning example.

    SFT trains a model to generate responses given conversation history. Each item
    contains the full conversation (messages) and the target response.

    Attributes:
        messages: Conversation history as Message objects.
        response: The target response the model should learn to generate.
        source: Data source identifier for provenance.
        meta: Optional metadata (topic, difficulty, language, etc.).

    Examples:
        >>> item = SFTItem(
        ...     messages=[Message(role="user", content="Explain NSAIDs.")],
        ...     response="NSAIDs are non-steroidal anti-inflammatory drugs...",
        ...     source="medqa"
        ... )
    """

    messages: tuple[Message, ...] = ()
    response: str = ""
    source: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        """Convert messages plus response to OpenAI-style training format."""
        msgs = [m.to_dict() for m in self.messages]
        msgs.append({"role": "assistant", "content": self.response})
        return msgs


# =============================================================================
# Preference (RLHF / DPO)
# =============================================================================


@dataclass(frozen=True)
class PrefPair:
    """A preference pair for RLHF / DPO training.

    Contains two responses to the same prompt: a chosen (better) response and a
    rejected (worse) response. The model learns to prefer chosen over rejected.

    Attributes:
        prompt: The input prompt (conversation history).
        chosen: The preferred response.
        rejected: The dispreferred response.
        source: Data source identifier.
        meta: Optional metadata.

    Examples:
        >>> pair = PrefPair(
        ...     prompt=[Message(role="user", content="What is dosage?")],
        ...     chosen="Consult label for dosage.",
        ...     rejected="Take 1000mg hourly.",
        ...     source="medmcqa"
        ... )
    """

    prompt: tuple[Message, ...] = ()
    chosen: str = ""
    rejected: str = ""
    source: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI-style preference format."""
        return {
            "messages": [m.to_dict() for m in self.prompt],
            "chosen": {"role": "assistant", "content": self.chosen},
            "rejected": {"role": "assistant", "content": self.rejected},
        }


# =============================================================================
# Reinforcement Learning (RL)
# =============================================================================


@dataclass(frozen=True)
class RLPrompt:
    """A prompt for RL rollout generation.

    RL prompts are conversation contexts that the policy model completes. The
    completion is then scored by a reward model.

    Attributes:
        messages: The conversation history up to (but not including) completion.
        source: Data source identifier.
        meta: Optional metadata for reward computation.

    Examples:
        >>> rl_prompt = RLPrompt(
        ...     messages=[Message(role="user", content="Diagnose: fever, cough")],
        ...     source="medcalc"
        ... )
    """

    messages: tuple[Message, ...] = ()
    source: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to OpenAI-style message list."""
        return [m.to_dict() for m in self.messages]


@dataclass(frozen=True)
class RLRollout:
    """A single RL rollout: prompt, completion, and reward.

    Produced by the policy during rollouts, consumed by the RL trainer.

    Attributes:
        prompt: The original prompt (as RLPrompt).
        completion: The model-generated completion.
        reward: Scalar reward signal.
        meta: Optional metadata (timestamp, seed, etc.).

    Examples:
        >>> rollout = RLRollout(
        ...     prompt=RLPrompt(messages=(Message(role="user", content="..."),)),
        ...     completion="Based on symptoms...",
        ...     reward=0.85
        ... )
    """

    prompt: RLPrompt
    completion: str
    reward: float
    meta: Mapping[str, Any] = field(default_factory=dict)


# =============================================================================
# Provenance & Metadata
# =============================================================================


class DataSource(BaseModel):
    """Provenance metadata for a data source.

    Tracks where data came from, when it was fetched, and any transformation
    history. Used for reproducibility and debugging.

    Attributes:
        name: Source identifier (e.g., "medqa", "stanford_alpaca").
        version: Version or commit hash of the source.
        fetched_at: When the data was fetched.
        count: Number of items from this source.
        url: Optional source URL or path.
        checksum: Content hash for verification.

    Examples:
        >>> source = DataSource(name="medqa", version="v1", count=10000)
        >>> source.model_dump()
        {'name': 'medqa', 'version': 'v1', 'fetched_at': ..., 'count': 10000, ...}
    """

    name: str = Field(description="Source identifier")
    version: str = Field(default="unknown", description="Version or commit hash")
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC).replace(tzinfo=None),
        description="Fetch timestamp"
    )
    count: int = Field(default=0, description="Number of items")
    url: str | None = Field(default=None, description="Source URL or path")
    checksum: str | None = Field(default=None, description="Content hash")

    model_config = {"frozen": True}


class DataStats(BaseModel):
    """Statistics about a dataset or pipeline stage.

    Used for logging and monitoring data quality through the pipeline.

    Attributes:
        total_items: Total number of items.
        deduped: Number of items removed by deduplication.
        decontaminated: Number of items removed by decontamination.
        filtered: Number of items removed by quality filters.
        avg_length: Average response length in characters.
        language_dist: Language distribution if applicable.

    Examples:
        >>> stats = DataStats(total_items=10000, deduped=500, filtered=200)
        >>> stats.retained
        9300
    """

    total_items: int = Field(default=0, description="Total input items")
    deduped: int = Field(default=0, description="Items removed by deduplication")
    decontaminated: int = Field(default=0, description="Items removed by decontamination")
    filtered: int = Field(default=0, description="Items removed by quality filters")
    avg_length: float = Field(default=0.0, description="Average response length")
    language_dist: dict[str, int] = Field(default_factory=dict, description="Language counts")

    @property
    def retained(self) -> int:
        """Number of items retained after all filtering."""
        return self.total_items - self.deduped - self.decontaminated - self.filtered

    @property
    def retention_rate(self) -> float:
        """Fraction of items retained (0-1)."""
        if self.total_items == 0:
            return 0.0
        return self.retained / self.total_items

    model_config = {"frozen": True}


# =============================================================================
# Pipeline State
# =============================================================================


class PipelineState(BaseModel):
    """State of a data pipeline run.

    Encapsulates all outputs and statistics from a pipeline execution for
    reproducibility and caching.

    Attributes:
        items: The processed data items.
        stats: Pipeline statistics.
        sources: Provenance metadata for all sources.
        config_hash: Hash of the pipeline configuration.

    Examples:
        >>> state = PipelineState(items=[item], stats=DataStats(total_items=1))
        >>> state.save_json("pipeline_state.json")
    """

    items: list[dict[str, Any]] = Field(default_factory=list, description="Processed items")
    stats: DataStats = Field(default_factory=DataStats, description="Pipeline statistics")
    sources: tuple[DataSource, ...] = Field(default=(), description="Source provenance")
    config_hash: str = Field(default="", description="Config hash for caching")

    def save_json(self, path: str) -> None:
        """Save state to a JSON file."""
        import json
        from pathlib import Path as StdLibPath

        from medrl.core.hashing import canonical_json

        with StdLibPath(path).open("w") as f:
            json.dump(canonical_json(self.model_dump()), f, indent=2)

    model_config = {"frozen": True}
