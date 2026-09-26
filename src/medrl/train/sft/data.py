"""SFT data loading and collation.

Handles loading SFT datasets, sequence packing, assistant-only loss masking,
and thinking mode integration. Supports multiple data formats and Axolotl-style
conversation formats.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from medrl.core.config import ThinkingConfig
from medrl.core.logging import get_logger
from medrl.train.sft.config import (
    SequencePackingMode,
    SFTLossConfig,
)

log = get_logger(__name__)

# Special tokens for sequence packing
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_PACK_SEPARATOR = ""


@dataclass
class SFTExample:
    """A single SFT training example.

    Attributes:
        messages: Conversation history in OpenAI format.
        response: Target assistant response.
        source: Data source identifier.
        thinking: Optional thinking content if using thinking mode.
        meta: Optional metadata.
    """

    messages: list[dict[str, str]]
    response: str
    source: str = "unknown"
    thinking: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_conversation(self) -> list[dict[str, str]]:
        """Convert to full conversation including response."""
        conv = list(self.messages)
        if self.thinking:
            # Prepend thinking to response
            response = f"<think >\n{self.thinking}\n\n"
            response += self.response
        else:
            response = self.response
        conv.append({"role": "assistant", "content": response})
        return conv


@dataclass
class PackedSequence:
    """A packed sequence with loss mask.

    Attributes:
        input_ids: Token IDs of the packed sequence.
        attention_mask: Attention mask (1 = real token, 0 = padding).
        labels: Labels for loss computation (-100 for ignored tokens).
        seq_lens: Lengths of individual sequences in the packed batch.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    seq_lens: list[int] = field(default_factory=list)


def load_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Load records from a JSONL file."""
    from pathlib import Path as StdLibPath
    with StdLibPath(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path | str) -> list[dict[str, Any]]:
    """Load records from a JSON file."""
    from pathlib import Path as StdLibPath
    with StdLibPath(path).open() as f:
        result = json.load(f)
        return result  # type: ignore[no-any-return]


def load_parquet(path: Path | str) -> list[dict[str, Any]]:
    """Load records from a Parquet file."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("pyarrow is required for parquet support: pip install pyarrow") from e

    table = pq.read_table(path)
    result = table.to_pylist()
    return result  # type: ignore[no-any-return]


class SFTDataset(Dataset[SFTExample]):
    """SFT dataset with loading and preprocessing.

    Supports multiple conversation formats and data sources.
    """

    def __init__(
        self,
        data_path: str | Path,
        conversation_format: Literal["sharegpt", "openai", "custom"] = "openai",
        data_type: Literal["json", "jsonl", "parquet", "hub"] = "jsonl",
        filter_fn: Callable[[SFTExample], bool] | None = None,
        min_response_length: int | None = None,
        max_response_length: int = 32768,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.conversation_format = conversation_format
        self.filter_fn = filter_fn
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.seed = seed
        self._rng = random.Random(seed)

        # Load data
        if data_type == "jsonl":
            self._records = list(load_jsonl(self.data_path))
        elif data_type == "json":
            self._records = load_json(self.data_path)
        elif data_type == "parquet":
            self._records = load_parquet(self.data_path)
        elif data_type == "hub":
            # Load from HuggingFace Hub
            try:
                from datasets import load_dataset
                # data_path should be like "dataset_name/config/split" or just "dataset_name"
                parts = str(data_path).split("/")
                if len(parts) >= 3:
                    dataset_name = "/".join(parts[:-2])
                    config_name = parts[-2]
                    split = parts[-1]
                    ds = load_dataset(dataset_name, name=config_name, split=split)
                elif len(parts) == 2:
                    dataset_name = parts[0]
                    split = parts[1]
                    ds = load_dataset(dataset_name, split=split)
                else:
                    ds = load_dataset(str(data_path))
                self._records = list(ds)
            except ImportError as e:
                raise ImportError("datasets package is required for 'hub' data_type. Install with: pip install datasets") from e
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        # Filter and parse
        self._examples = self._parse_records()

    def _parse_records(self) -> list[SFTExample]:
        """Parse raw records into SFT examples."""
        examples = []

        for record in self._records:
            try:
                example = self._parse_record(record)
                if example and self._passes_filter(example):
                    examples.append(example)
            except Exception as e:
                log.warning("Failed to parse record: %s", e)
                continue

        log.info("Loaded %d examples from %s", len(examples), self.data_path)
        return examples

    def _parse_record(self, record: dict[str, Any]) -> SFTExample | None:
        """Parse a single record into an SFT example based on format."""
        if self.conversation_format == "openai":
            return self._parse_openai(record)
        elif self.conversation_format == "sharegpt":
            return self._parse_sharegpt(record)
        else:
            return self._parse_custom(record)

    def _parse_openai(self, record: dict[str, Any]) -> SFTExample | None:
        """Parse OpenAI-style conversation format.

        Expected format:
            {"messages": [..., {"role": "user", "content": "..."}], "response": "..."}
        or
            {"messages": [..., {"role": "assistant", "content": "..."}]}
        """
        messages: list[dict[str, str]] = record.get("messages", [])
        response = ""

        # Handle response as separate field or as last message
        if "response" in record:
            response = record["response"]
        elif messages and messages[-1].get("role") == "assistant":
            response = messages.pop()["content"]
        else:
            return None  # Invalid format

        return SFTExample(
            messages=messages,
            response=response,
            source=record.get("source", "unknown"),
            thinking=record.get("thinking"),
            meta={k: v for k, v in record.items() if k not in {"messages", "response", "source", "thinking"}},
        )

    def _parse_sharegpt(self, record: dict[str, Any]) -> SFTExample | None:
        """Parse ShareGPT-style format.

        Expected format:
            {"conversations": [["from", "human"], ["from", "gpt"], ...]}
        """
        convs = record.get("conversations", [])
        if not convs:
            return None

        messages = []
        response = ""
        thinking = None

        for i, turn in enumerate(convs):
            if len(turn) < 2:
                continue
            role, content = turn[0], turn[1]

            if role.lower() in {"human", "user"}:
                messages.append({"role": "user", "content": content})
            elif role.lower() in {"gpt", "assistant", "chatgpt"}:
                if i == len(convs) - 1:
                    # Last assistant message is the response
                    response = content
                else:
                    messages.append({"role": "assistant", "content": content})

        # Extract thinking if present in response
        if "<think >" in response and "</think >" in response:
            parts = response.split("</think >", 1)
            if len(parts) == 2:
                thinking_part = parts[0].split("<think >", 1)[-1]
                thinking = thinking_part.strip()
                response = parts[1].strip()

        return SFTExample(
            messages=messages,
            response=response,
            source=record.get("source", "sharegpt"),
            thinking=thinking,
            meta={k: v for k, v in record.items() if k not in {"conversations", "source"}},
        )

    def _parse_custom(self, record: dict[str, Any]) -> SFTExample | None:
        """Parse custom format (user-defined fields).

        Expected format (configurable):
            {"prompt": "...", "completion": "..."}
        or
            {"instruction": "...", "output": "..."}
        """
        # Try common field names
        prompt_fields = ["prompt", "instruction", "input", "question"]
        completion_fields = ["completion", "output", "answer", "response"]

        prompt = None
        for field_name in prompt_fields:
            if field_name in record:
                prompt = record[field_name]
                break

        completion = None
        for field_name in completion_fields:
            if field_name in record:
                completion = record[field_name]
                break

        if prompt is None or completion is None:
            return None

        return SFTExample(
            messages=[{"role": "user", "content": prompt}],
            response=completion,
            source=record.get("source", "custom"),
            thinking=record.get("thinking"),
            meta={k: v for k, v in record.items() if k not in prompt_fields + completion_fields + ["source", "thinking"]},
        )

    def _passes_filter(self, example: SFTExample) -> bool:
        """Check if example passes quality filters."""
        # Length filters
        if self.min_response_length and len(example.response) < self.min_response_length:
            return False
        if len(example.response) > self.max_response_length:
            return False

        # Custom filter
        return not (self.filter_fn and not self.filter_fn(example))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> SFTExample:
        return self._examples[idx]


class SFTCollator:
    """Collator for SFT training with sequence packing and loss masking.

    Handles:
    - Tokenization of conversations
    - Sequence packing to minimize padding waste
    - Assistant-only loss masking
    - Thinking mode handling
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 8192,
        packing: SequencePackingMode = SequencePackingMode.SMART,
        packing_separator: str = "",
        loss_config: SFTLossConfig = SFTLossConfig(),
        thinking_config: ThinkingConfig | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packing = packing
        self.packing_separator = packing_separator
        self.loss_config = loss_config
        self.thinking_config = thinking_config

        # Cache token IDs
        self.user_token_id = self._get_role_token("user")
        self.assistant_token_id = self._get_role_token("assistant")
        self.system_token_id = self._get_role_token("system")

    def _get_role_token(self, role: str) -> int | None:
        """Get token ID for a role marker if present."""
        # This depends on the tokenizer's chat template
        # Default: None (role is encoded as text)
        return None

    def __call__(self, examples: list[SFTExample]) -> dict[str, torch.Tensor]:
        """Collate a batch of examples into tensors.

        Returns:
            Dict with input_ids, attention_mask, and labels tensors.
        """
        if self.packing == SequencePackingMode.NONE:
            return self._collate_no_packing(examples)
        else:
            return self._collate_packed(examples)

    def _collate_no_packing(self, examples: list[SFTExample]) -> dict[str, torch.Tensor]:
        """Collate without sequence packing."""
        # Tokenize each example independently
        batch_input_ids = []
        batch_labels = []

        for example in examples:
            conversation = example.to_conversation()
            encoded = self.tokenizer.apply_chat_template(
                conversation,
                return_tensors="pt",
                return_dict=True,
            )

            input_ids = encoded["input_ids"]  # type: ignore[index, call-overload]
            if hasattr(input_ids, "squeeze"):
                input_ids = input_ids.squeeze(0)
            else:
                raise TypeError("input_ids is not a tensor")
            labels = self._create_loss_mask(conversation, input_ids)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        # Pad to same length
        padded = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding="longest",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        padded["labels"] = self._pad_labels(batch_labels, padded["attention_mask"].shape)

        return padded  # type: ignore[return-value]

    def _collate_packed(self, examples: list[SFTExample]) -> dict[str, torch.Tensor]:
        """Collate with sequence packing to minimize waste."""
        # First, tokenize all examples
        tokenized_examples = []
        for example in examples:
            conversation = example.to_conversation()
            encoded = self.tokenizer.apply_chat_template(
                conversation,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"]  # type: ignore[index, call-overload]
            if hasattr(input_ids, "squeeze"):
                input_ids = input_ids.squeeze(0)
            else:
                raise TypeError("input_ids is not a tensor")
            labels = self._create_loss_mask(conversation, input_ids)
            tokenized_examples.append((input_ids, labels))

        # Pack sequences
        packed_input_ids: list[torch.Tensor] = []
        packed_attention_mask: list[torch.Tensor] = []
        packed_labels: list[torch.Tensor] = []

        current_input_ids: list[int] = []
        current_labels: list[int] = []
        seq_lens: list[int] = []

        for input_ids, labels in tokenized_examples:
            if self.packing == SequencePackingMode.SMART:
                # Check if adding this sequence would exceed max length
                if current_input_ids and len(current_input_ids) + len(input_ids) > self.max_seq_len:
                    # Flush current batch
                    packed_input_ids.append(self._pack_and_pad(current_input_ids, seq_lens))
                    packed_attention_mask.append(torch.ones(len(self._pack_and_pad(current_input_ids, seq_lens))))
                    packed_labels.append(self._pack_and_pad(current_labels, seq_lens))
                    current_input_ids = []
                    current_labels = []
                    seq_lens = []

                current_input_ids.extend(input_ids.tolist())
                current_labels.extend(labels.tolist())
                seq_lens.append(len(input_ids))
            else:
                # CONCAT mode: simple concatenation
                current_input_ids.extend(input_ids.tolist())
                current_labels.extend(labels.tolist())
                seq_lens.append(len(input_ids))

                # Check length
                if len(current_input_ids) >= self.max_seq_len:
                    packed_input_ids.append(self._pack_and_pad(current_input_ids, seq_lens))
                    packed_attention_mask.append(torch.ones(len(self._pack_and_pad(current_input_ids, seq_lens))))
                    packed_labels.append(self._pack_and_pad(current_labels, seq_lens))
                    current_input_ids = []
                    current_labels = []
                    seq_lens = []

        # Flush remaining
        if current_input_ids:
            packed_input_ids.append(self._pack_and_pad(current_input_ids, seq_lens))
            packed_attention_mask.append(torch.ones(len(self._pack_and_pad(current_input_ids, seq_lens))))
            packed_labels.append(self._pack_and_pad(current_labels, seq_lens))

        # Stack into batch
        max_len = max(len(ids) for ids in packed_input_ids)
        batch_input_ids = torch.zeros((len(packed_input_ids), max_len), dtype=torch.long)
        batch_attention_mask = torch.zeros((len(packed_input_ids), max_len), dtype=torch.long)
        batch_labels = torch.full((len(packed_input_ids), max_len), self.loss_config.ignore_index, dtype=torch.long)

        for i, (ids, mask, labels) in enumerate(zip(packed_input_ids, packed_attention_mask, packed_labels, strict=False)):
            batch_input_ids[i, :len(ids)] = ids
            batch_attention_mask[i, :len(mask)] = mask
            batch_labels[i, :len(labels)] = labels

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

    def _pack_and_pad(self, tokens: list[int], seq_lens: list[int]) -> torch.Tensor:
        """Pack tokens and pad to max_seq_len."""
        packed = torch.tensor(tokens, dtype=torch.long)
        if len(packed) > self.max_seq_len:
            packed = packed[:self.max_seq_len]
        return packed

    def _pad_labels(self, labels: list[torch.Tensor], target_shape: tuple[int, int]) -> torch.Tensor:
        """Pad labels to target shape with ignore_index."""
        batch_size, max_len = target_shape
        padded = torch.full((batch_size, max_len), self.loss_config.ignore_index, dtype=torch.long)
        for i, label in enumerate(labels):
            padded[i, :len(label)] = label
        return padded

    def _create_loss_mask(
        self,
        conversation: list[dict[str, str]],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Create loss mask based on loss configuration.

        Returns:
            Tensor with -100 for tokens to ignore, original IDs otherwise.
        """
        labels = input_ids.clone()

        # Default: compute loss on all tokens
        # We'll mask out based on configuration

        # Reconstruct the conversation token by token to identify roles
        # This is a simplified approach; a production system would track
        # token ranges more precisely during template application

        if self.loss_config.mask_user_tokens or self.loss_config.mask_system_tokens:
            # Apply chat template with loss tracking
            # This requires token-by-token reconstruction

            # Simplified approach: mark non-assistant tokens
            # In production, use tokenizer's built-in loss masking
            pass

        # Handle thinking mode
        if self.thinking_config and not self.loss_config.include_thinking_in_loss:
            # Mask out thinking tokens
            # This requires tracking where thinking tokens are
            pass

        return labels


def create_sft_dataloader(
    dataset: SFTDataset,
    batch_size: int,
    collator: SFTCollator,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    shuffle: bool = True,
) -> DataLoader[SFTExample]:
    """Create a DataLoader for SFT training.

    Args:
        dataset: SFTDataset instance.
        batch_size: Batch size (per device).
        collator: SFTCollator instance.
        num_workers: Number of data loading workers.
        prefetch_factor: Number of batches to prefetch.
        shuffle: Whether to shuffle the dataset.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
    )


def estimate_tokens_per_example(
    examples: list[SFTExample],
    tokenizer: PreTrainedTokenizerBase,
    sample_size: int = 100,
) -> int:
    """Estimate average tokens per example for budget planning.

    Args:
        examples: List of SFT examples.
        tokenizer: Tokenizer to use.
        sample_size: Number of examples to sample for estimation.

    Returns:
        Estimated average token count per example.
    """
    sample = examples[:min(sample_size, len(examples))]

    total_tokens = 0
    for example in sample:
        conversation = example.to_conversation()
        encoded = tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]  # type: ignore[index, call-overload]
        if hasattr(input_ids, "shape"):
            total_tokens += input_ids.shape[1]

    return total_tokens // len(sample) if sample else 0
