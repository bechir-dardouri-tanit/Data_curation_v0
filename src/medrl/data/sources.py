"""Data source loaders and registries for the medrl pipeline.

Provides loaders for common medical dataset formats (JSON, JSONL, Parquet, HuggingFace)
and a registry system for registering custom loaders. Handles discovery and
loading of training data from various sources.

Typical usage:
    >>> from medrl.data.sources import load_jsonl, load_huggingface
    >>> items = load_jsonl("data/medqa.jsonl")
    >>> dataset = load_huggingface("medqa", split="train")
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger
from medrl.core.registry import Registry

logger = get_logger(__name__)


# =============================================================================
# Data Source Registry
# =============================================================================


class LoaderRegistry(Registry[Callable[[str], Sequence[dict[str, Any]]]]):
    """Registry for data source loaders.

    Allows custom loaders to be registered and looked up by name.

    Examples:
        >>> registry = LoaderRegistry("loader")
        >>> @registry.register("custom")
        ... def my_loader(path: str) -> list[dict]:
        ...     return [...]
        >>> loader = registry.get("custom")
    """

    def __init__(self) -> None:
        super().__init__(kind="loader")


# Global loader registry
loaders = LoaderRegistry()


# =============================================================================
# File Format Loaders
# =============================================================================


@loaders.register("jsonl")  # type: ignore[untyped-decorator,arg-type]
def load_jsonl(path: str, text_key: str = "text") -> list[dict[str, Any]]:
    """Load data from JSONL file.

    Args:
        path: Path to JSONL file.
        text_key: Key to use as text field if not present.

    Returns:
        List of items from the file.

    Examples:
        >>> items = load_jsonl("data/medqa.jsonl")
        >>> len(items) > 0
        True
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path_obj}")

    items = []
    with path_obj.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Ensure text key exists
                if "text" not in item and text_key in item:
                    item["text"] = item[text_key]
                items.append(item)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse line {line_num}: {e}")

    logger.info(f"Loaded {len(items)} items from JSONL: {path_obj}")
    return items


@loaders.register("json")  # type: ignore[untyped-decorator,arg-type]
def load_json(path: str, text_key: str = "text") -> list[dict[str, Any]]:
    """Load data from JSON file.

    Expects a JSON array of objects or a dict with a "data" key.

    Args:
        path: Path to JSON file.
        text_key: Key to use as text field if not present.

    Returns:
        List of items from the file.

    Examples:
        >>> items = load_json("data/medqa.json")
        >>> len(items) > 0
        True
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path_obj}")

    with path_obj.open("r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        items: list[dict[str, Any]] = data
    elif isinstance(data, dict):
        items = data.get("data", data.get("items", []))  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported JSON format in {path_obj}")

    # Ensure text key exists
    for item in items:
        if "text" not in item and text_key in item:
            item["text"] = item[text_key]

    logger.info(f"Loaded {len(items)} items from JSON: {path_obj}")
    return items


@loaders.register("parquet")  # type: ignore[untyped-decorator,arg-type]
def load_parquet(path: str, text_key: str = "text") -> list[dict[str, Any]]:
    """Load data from Parquet file.

    Args:
        path: Path to Parquet file.
        text_key: Key to use as text field if not present.

    Returns:
        List of items from the file.

    Raises:
        ImportError: If pyarrow is not installed.

    Examples:
        >>> items = load_parquet("data/medqa.parquet")
        >>> len(items) > 0
        True
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for Parquet loading. "
            "Install with: pip install pyarrow"
        ) from e

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path_obj}")

    table = pq.read_table(path_obj)
    items: list[dict[str, Any]] = table.to_pylist()

    # Ensure text key exists
    for item in items:
        if "text" not in item and text_key in item:
            item["text"] = item[text_key]

    logger.info(f"Loaded {len(items)} items from Parquet: {path_obj}")
    return items


@loaders.register("txt")  # type: ignore[untyped-decorator,arg-type]
def load_txt(path: str, delimiter: str = "\n\n") -> list[dict[str, Any]]:
    """Load data from plain text file.

    Each document/entry is separated by delimiter (default: double newline).

    Args:
        path: Path to text file.
        delimiter: Entry separator.

    Returns:
        List of items with "text" field.

    Examples:
        >>> items = load_txt("data/documents.txt")
        >>> items[0]["text"]
        'First document...'
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path_obj}")

    with path_obj.open("r") as f:
        content = f.read()

    texts = content.split(delimiter)
    items = [{"text": text.strip()} for text in texts if text.strip()]

    logger.info(f"Loaded {len(items)} items from TXT: {path_obj}")
    return items


# =============================================================================
# HuggingFace Datasets Loader
# =============================================================================


@loaders.register("huggingface")  # type: ignore[untyped-decorator,arg-type]
def load_huggingface(
    path: str,
    split: str = "train",
    text_field: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Load data from HuggingFace Datasets.

    Args:
        path: Dataset path (name or local path).
        split: Dataset split to load.
        text_field: Field to use as "text" (auto-detected if None).
        **kwargs: Additional arguments for load_dataset.

    Returns:
        List of items from the dataset.

    Raises:
        ImportError: If datasets is not installed.

    Examples:
        >>> items = load_huggingface("medqa", split="train")
        >>> len(items) > 0
        True
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets is required for HuggingFace loading. "
            "Install with: pip install datasets"
        ) from e

    logger.info(f"Loading HuggingFace dataset: {path}, split={split}")

    dataset = load_dataset(path, split=split, **kwargs)

    # Convert to list of dicts
    items = [dict(row) for row in dataset]

    # Auto-detect text field if not specified
    if text_field is None:
        # Common field names to try
        candidates = ["text", "question", "prompt", "content", "document"]
        for candidate in candidates:
            if items and candidate in items[0]:
                text_field = candidate
                break

        # If still None, use first string field
        if text_field is None and items:
            for key, value in items[0].items():
                if isinstance(value, str):
                    text_field = key
                    break

    # Ensure "text" key exists
    if text_field and text_field != "text":
        for item in items:
            if "text" not in item and text_field in item:
                item["text"] = item[text_field]

    logger.info(f"Loaded {len(items)} items from HuggingFace: {path}")
    return items


# =============================================================================
# Directory Loader
# =============================================================================


@loaders.register("directory")  # type: ignore[untyped-decorator,arg-type]
def load_directory(
    path: str,
    pattern: str = "*.jsonl",
    recursive: bool = True,
    loader: str | None = None,
) -> list[dict[str, Any]]:
    """Load data from all matching files in a directory.

    Args:
        path: Directory path.
        pattern: Glob pattern for files.
        recursive: Whether to search subdirectories.
        loader: Loader name (auto-detected from extension if None).

    Returns:
        Combined list of items from all files.

    Examples:
        >>> items = load_directory("data/", pattern="*.jsonl")
        >>> len(items) > 0
        True
    """
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    files = list(dir_path.glob(pattern)) if not recursive else list(dir_path.rglob(pattern))

    if not files:
        logger.warning(f"No files found matching {pattern} in {path}")
        return []

    all_items: list[dict[str, Any]] = []
    for file_path in files:
        try:
            # Auto-detect loader from extension
            if loader is None:
                ext = file_path.suffix.lstrip(".")
                loader_map: dict[str, str] = {
                    "jsonl": "jsonl",
                    "json": "json",
                    "parquet": "parquet",
                    "txt": "txt",
                }
                loader_name = loader_map.get(ext, "jsonl")

            # Load file
            load_func = loaders.get(loader_name or "jsonl")
            items = load_func(str(file_path))
            all_items.extend(items)

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

    logger.info(f"Loaded {len(all_items)} items from {len(files)} files in {path}")
    return all_items


# =============================================================================
# Streaming Loader (for large datasets)
# =============================================================================


def stream_jsonl(path: str, batch_size: int = 1000) -> Iterator[list[dict[str, Any]]]:
    """Stream JSONL file in batches.

    Useful for processing large datasets that don't fit in memory.

    Args:
        path: Path to JSONL file.
        batch_size: Items per batch.

    Yields:
        Batches of items.

    Examples:
        >>> for batch in stream_jsonl("large.jsonl", batch_size=100):
        ...     process_batch(batch)
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path_obj}")

    batch: list[dict[str, Any]] = []
    with path_obj.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                batch.append(item)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse line: {e}")

    if batch:
        yield batch


# =============================================================================
# Source Metadata
# =============================================================================


@dataclass(frozen=True)
class SourceInfo:
    """Metadata about a data source.

    Attributes:
        path: Source path or identifier.
        format: Format name (jsonl, parquet, etc.).
        count: Number of items.
        size_bytes: File size in bytes.
        checksum: Content hash.
    """

    path: str
    format: str = "unknown"
    count: int = 0
    size_bytes: int = 0
    checksum: str = ""

    @classmethod
    def from_file(cls, path: str, loader: str = "jsonl") -> SourceInfo:
        """Create source info by inspecting a file.

        Args:
            path: File path.
            loader: Loader name.

        Returns:
            SourceInfo instance.
        """
        path_obj = Path(path)
        size_bytes = path_obj.stat().st_size if path_obj.exists() else 0

        # Try to get count
        count = 0
        if path_obj.exists():
            try:
                load_func = loaders.get(loader)
                items = load_func(str(path_obj))
                count = len(items)
            except Exception:
                pass

        return cls(path=str(path_obj), format=loader, count=count, size_bytes=size_bytes)


def inspect_source(path: str) -> SourceInfo:
    """Inspect a data source without loading it.

    Args:
        path: Source path.

    Returns:
        SourceInfo with metadata.

    Examples:
        >>> info = inspect_source("data/medqa.jsonl")
        >>> info.count
        10000
    """
    path_obj = Path(path)

    if not path_obj.exists():
        return SourceInfo(path=str(path_obj))

    # Detect format from extension
    ext_map = {
        ".jsonl": "jsonl",
        ".json": "json",
        ".parquet": "parquet",
        ".txt": "txt",
    }

    format_name = ext_map.get(path_obj.suffix.lower(), "unknown")
    size_bytes = path_obj.stat().st_size

    # Try to get item count
    count = 0
    if format_name == "jsonl":
        with path_obj.open("r") as f:
            count = sum(1 for line in f if line.strip())

    return SourceInfo(path=str(path_obj), format=format_name, count=count, size_bytes=size_bytes)


# =============================================================================
# Universal Load Function
# =============================================================================


def load_source(
    path: str,
    format: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Load data from a source with auto-detection.

    Args:
        path: Source path or identifier.
        format: Format name (auto-detected if None).
        **kwargs: Additional arguments for the loader.

    Returns:
        List of items.

    Examples:
        >>> items = load_source("data/medqa.jsonl")
        >>> items = load_source("medqa", format="huggingface")
    """
    path_obj = Path(path)

    # Auto-detect format if not specified
    if format is None:
        if path_obj.exists():
            ext_map = {
                ".jsonl": "jsonl",
                ".json": "json",
                ".parquet": "parquet",
                ".txt": "txt",
            }
            format = ext_map.get(path_obj.suffix.lower())
        else:
            # Assume HuggingFace dataset
            format = "huggingface"

    if format is None:
        raise ValueError(f"Could not detect format for: {path}")

    # Get loader and load
    load_func = loaders.get(format)
    result = load_func(str(path_obj), **kwargs)
    return list(result)
