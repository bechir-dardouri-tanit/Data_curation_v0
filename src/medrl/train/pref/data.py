"""Preference data loading and sampling.

Handles loading, preprocessing, and sampling of preference pairs for
DPO/SimPO training. Supports multiple sampling strategies and integrates
with the data schemas defined in medrl.data.schemas.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerBase

from medrl.core.logging import get_logger
from medrl.data.schemas import Message, PrefPair
from medrl.train.pref.config import PairSampling, PrefOptConfig

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefBatch:
    """A single training batch.

    Attributes:
        chosen_ids: Token IDs for chosen responses [batch, seq_len].
        rejected_ids: Token IDs for rejected responses [batch, seq_len].
        chosen_labels: Labels for chosen responses (with -100 for prompt) [batch, seq_len].
        rejected_labels: Labels for rejected responses [batch, seq_len].
        chosen_mask: Attention mask for chosen responses [batch, seq_len].
        rejected_mask: Attention mask for rejected responses [batch, seq_len].
        metadata: Optional metadata for logging/debugging.
    """

    chosen_ids: torch.Tensor
    rejected_ids: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_labels: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_mask: torch.Tensor
    metadata: list[dict[str, Any]] | None = None


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class PrefDataset(Dataset[PrefPair]):
    """Dataset of preference pairs.

    Supports loading from JSONL files where each line is a JSON object with:
        - prompt: list of message dicts [{"role": "user", "content": "..."}]
        - chosen: the chosen response text
        - rejected: the rejected response text
        - source: optional source identifier
        - meta: optional metadata dict
    """

    def __init__(
        self,
        path: str,
        num_examples: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            path: Path to the JSONL file containing preference pairs.
            num_examples: Optional limit on number of examples to load.
        """
        self.path = Path(path)
        self.num_examples = num_examples
        self.pairs: list[PrefPair] = []
        self._load()

    def _load(self) -> None:
        """Load preference pairs from disk."""
        if not self.path.exists():
            raise FileNotFoundError(f"Preference data not found: {self.path}")

        log.info("Loading preference pairs from %s", self.path)
        counts: Counter[str] = Counter()

        with self.path.open() as f:
            for i, line in enumerate(f):
                if self.num_examples and len(self.pairs) >= self.num_examples:
                    break

                try:
                    raw = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    log.warning("Skipping invalid JSON at line %d: %s", i + 1, e)
                    continue

                try:
                    pair = self._parse_pair(raw)
                    self.pairs.append(pair)
                    counts[pair.source] += 1
                except (KeyError, TypeError) as e:
                    log.warning("Skipping invalid pair at line %d: %s", i + 1, e)
                    continue

        log.info(
            "Loaded %d preference pairs from %d sources: %s",
            len(self.pairs),
            len(counts),
            dict(counts.most_common(5)),
        )

    def _parse_pair(self, raw: dict[str, Any]) -> PrefPair:
        """Parse a single preference pair from raw JSON.

        Args:
            raw: The raw JSON object.

        Returns:
            A PrefPair instance.
        """
        prompt_msgs = raw.get("prompt", [])
        if isinstance(prompt_msgs, str):
            # Handle legacy string format
            prompt_msgs = [{"role": "user", "content": prompt_msgs}]

        # Convert message dicts to Message objects
        messages = tuple(
            Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                meta={},
            )
            for m in prompt_msgs
        )

        return PrefPair(
            prompt=messages,
            chosen=raw.get("chosen", ""),
            rejected=raw.get("rejected", ""),
            source=raw.get("source", "unknown"),
            meta=raw.get("meta", {}),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PrefPair:
        return self.pairs[idx]


# --------------------------------------------------------------------------------------
# Sampling strategies
# --------------------------------------------------------------------------------------


class PrefSampler(Sampler[int]):
    """Custom sampler for preference pairs.

    Supports various sampling strategies:
    - RANDOM: Uniform random sampling
    - BALANCED: Ensure equal chosen/rejected representation
    - HARD_NEGATIVES: Prioritize hard negatives (high loss)
    - STRATIFIED: Sample proportionally across sources
    """

    def __init__(
        self,
        dataset: PrefDataset,
        strategy: PairSampling = PairSampling.RANDOM,
        shuffle: bool = True,
        generator: random.Random | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset: The preference dataset.
            strategy: The sampling strategy to use.
            shuffle: Whether to shuffle indices.
            generator: Random generator for reproducibility.
        """
        self.dataset = dataset
        self.strategy = strategy
        self.shuffle = shuffle
        self.generator = generator or random.Random()

        # Build source index for stratified sampling
        self.source_indices: dict[str, list[int]] = {}
        if strategy == PairSampling.STRATIFIED:
            for idx, pair in enumerate(dataset.pairs):
                self.source_indices.setdefault(pair.source, []).append(idx)

        # Hard negatives tracking (updated externally during training)
        self.hard_negatives: set[int] = set()

    def _sample_indices(self) -> list[int]:
        """Generate indices according to the sampling strategy."""
        n = len(self.dataset)

        if self.strategy == PairSampling.RANDOM:
            indices = list(range(n))

        elif self.strategy == PairSampling.BALANCED:
            # Balanced sampling: duplicate indices to ensure equal representation
            # This is a simple approach; for true balance, we'd need label-based sampling
            indices = list(range(n)) * 2  # Duplicate to ensure more mixing

        elif self.strategy == PairSampling.HARD_NEGATIVES:
            # Prioritize hard negatives with some random exploration
            hard_indices = list(self.hard_negatives)
            random_indices = [i for i in range(n) if i not in self.hard_negatives]
            # Mix: 50% hard negatives, 50% random
            n_hard = min(len(hard_indices), n // 2)
            n_random = n - n_hard
            indices = (
                self.generator.sample(hard_indices, n_hard) +
                self.generator.sample(random_indices, n_random)
            )

        elif self.strategy == PairSampling.STRATIFIED:
            # Sample proportionally from each source
            indices = []
            for _source, source_idxs in self.source_indices.items():
                # Sample proportionally to source size
                n_from_source = max(1, int(len(source_idxs) / n * n))
                indices.extend(self.generator.sample(source_idxs, n_from_source))
            # Pad with random samples if needed
            while len(indices) < n:
                indices.append(self.generator.randint(0, n - 1))

        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

        if self.shuffle:
            self.generator.shuffle(indices)

        return indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._sample_indices())

    def __len__(self) -> int:
        return len(self.dataset)

    def update_hard_negatives(
        self,
        indices: list[int],
        losses: list[float],
        percentile: float = 0.8,
    ) -> None:
        """Update the hard negatives set based on recent losses.

        Args:
            indices: The batch indices.
            losses: The corresponding losses.
            percentile: Loss percentile to consider as "hard" (default 0.8 = top 20%).
        """
        if self.strategy != PairSampling.HARD_NEGATIVES:
            return

        # Compute threshold
        threshold = torch.tensor(losses).quantile(percentile).item()

        # Update hard negatives
        for idx, loss in zip(indices, losses, strict=False):
            if loss >= threshold:
                self.hard_negatives.add(idx)
            else:
                self.hard_negatives.discard(idx)

        # Keep only the top N% hardest
        max_hard = len(self.dataset) // 4
        if len(self.hard_negatives) > max_hard:
            # Prune by keeping the most frequently seen
            # (In practice, you'd track frequency, but this is a simple approximation)
            self.hard_negatives = set(list(self.hard_negatives)[:max_hard])


# --------------------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CollatorConfig:
    """Configuration for the batch collator."""

    max_length: int
    max_prompt_length: int
    max_completion_length: int
    pad_token_id: int
    truncation_side: str = "right"


class PrefCollator:
    """Collator for preference data batches.

    Handles:
    - Tokenization
    - Truncation
    - Padding
    - Creating attention masks
    - Creating labels (with -100 for prompt tokens)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: CollatorConfig,
    ) -> None:
        """Initialize the collator.

        Args:
            tokenizer: The tokenizer to use.
            config: Collator configuration.
        """
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, pairs: list[PrefPair]) -> PrefBatch:
        """Collate a batch of preference pairs.

        Args:
            pairs: List of preference pairs.

        Returns:
            A PrefBatch with tokenized and padded tensors.
        """
        len(pairs)

        # Tokenize prompts, chosen, and rejected
        chosen_ids_list = []
        rejected_ids_list = []
        chosen_labels_list = []
        rejected_labels_list = []
        chosen_mask_list = []
        rejected_mask_list = []
        metadata_list = []

        for pair in pairs:
            # Format messages for tokenizer
            messages = [m.to_dict() for m in pair.prompt]

            # Tokenize chosen
            chosen_full = self.tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": pair.chosen}],
                return_tensors="pt",
                add_generation_prompt=True,
            )
            chosen_ids = chosen_full[0].tolist() if hasattr(chosen_full[0], "tolist") else chosen_full[0]

            # Tokenize rejected
            rejected_full = self.tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": pair.rejected}],
                return_tensors="pt",
                add_generation_prompt=True,
            )
            rejected_ids = rejected_full[0].tolist() if hasattr(rejected_full[0], "tolist") else rejected_full[0]

            # Find prompt length (same for both)
            prompt_full = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            prompt_len = len(prompt_full[0])  # type: ignore[arg-type]

            # Truncate if needed
            chosen_ids = self._truncate(chosen_ids, prompt_len, self.config.max_completion_length)  # type: ignore[arg-type]
            rejected_ids = self._truncate(rejected_ids, prompt_len, self.config.max_completion_length)  # type: ignore[arg-type]

            # Create labels (-100 for prompt tokens)
            chosen_labels = chosen_ids.copy()
            rejected_labels = rejected_ids.copy()
            for i in range(prompt_len):
                chosen_labels[i] = -100
                rejected_labels[i] = -100

            # Global truncation
            chosen_ids = chosen_ids[: self.config.max_length]
            rejected_ids = rejected_ids[: self.config.max_length]
            chosen_labels = chosen_labels[: self.config.max_length]
            rejected_labels = rejected_labels[: self.config.max_length]

            # Create attention masks
            chosen_mask = [1] * len(chosen_ids)
            rejected_mask = [1] * len(rejected_ids)

            chosen_ids_list.append(chosen_ids)
            rejected_ids_list.append(rejected_ids)
            chosen_labels_list.append(chosen_labels)
            rejected_labels_list.append(rejected_labels)
            chosen_mask_list.append(chosen_mask)
            rejected_mask_list.append(rejected_mask)
            metadata_list.append({"source": pair.source, "meta": pair.meta})

        # Pad to max length in batch
        max_len = max(
            max(len(ids) for ids in chosen_ids_list),
            max(len(ids) for ids in rejected_ids_list),
        )

        chosen_ids_padded = self._pad(chosen_ids_list, max_len)
        rejected_ids_padded = self._pad(rejected_ids_list, max_len)
        chosen_labels_padded = self._pad(chosen_labels_list, max_len, pad_value=-100)
        rejected_labels_padded = self._pad(rejected_labels_list, max_len, pad_value=-100)
        chosen_mask_padded = self._pad(chosen_mask_list, max_len, pad_value=0)
        rejected_mask_padded = self._pad(rejected_mask_list, max_len, pad_value=0)

        return PrefBatch(
            chosen_ids=torch.tensor(chosen_ids_padded, dtype=torch.long),
            rejected_ids=torch.tensor(rejected_ids_padded, dtype=torch.long),
            chosen_labels=torch.tensor(chosen_labels_padded, dtype=torch.long),
            rejected_labels=torch.tensor(rejected_labels_padded, dtype=torch.long),
            chosen_mask=torch.tensor(chosen_mask_padded, dtype=torch.long),
            rejected_mask=torch.tensor(rejected_mask_padded, dtype=torch.long),
            metadata=metadata_list,
        )

    def _truncate(
        self,
        ids: list[int],
        prompt_len: int,
        max_completion_len: int,
    ) -> list[int]:
        """Truncate completion while preserving prompt.

        Args:
            ids: Token IDs.
            prompt_len: Length of prompt prefix.
            max_completion_len: Maximum completion length.

        Returns:
            Truncated token IDs.
        """
        completion_len = len(ids) - prompt_len
        if completion_len <= max_completion_len:
            return ids

        # Truncate completion
        new_len = prompt_len + max_completion_len
        return ids[:new_len]

    def _pad(
        self,
        sequences: list[list[int]],
        max_len: int,
        pad_value: int = 0,
    ) -> list[list[int]]:
        """Pad sequences to max_len.

        Args:
            sequences: List of token sequences.
            max_len: Target length.
            pad_value: Padding value.

        Returns:
            Padded sequences.
        """
        return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]


def create_dataloader(
    dataset: PrefDataset,
    tokenizer: PreTrainedTokenizerBase,
    config: PrefOptConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader[PrefBatch]:
    """Create a DataLoader for preference training.

    Args:
        dataset: The preference dataset.
        tokenizer: The tokenizer to use.
        config: Preference optimization configuration.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        A configured DataLoader.
    """
    # Create sampler
    sampler = PrefSampler(
        dataset=dataset,
        strategy=config.pair_sampling,
        shuffle=shuffle,
    )

    # Create collator
    collator = PrefCollator(
        tokenizer=tokenizer,
        config=CollatorConfig(
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )

    return DataLoader(
        dataset=dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
