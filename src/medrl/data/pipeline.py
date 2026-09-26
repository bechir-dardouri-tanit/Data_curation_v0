"""Main data processing pipeline for medrl.

Orchestrates the full data pipeline: loading, deduplication, decontamination,
filtering, and output generation. Provides a unified API for building datasets
from multiple sources with caching and reproducibility.

Typical usage:
    >>> from medrl.data import DataPipeline
    >>> pipeline = DataPipeline(name="sft-medqa")
    >>> pipeline.add_source("medqa", path="data/medqa.jsonl")
    >>> pipeline.dedup(minhash_threshold=0.9)
    >>> pipeline.decontam(benchmarks=["medqa", "medmcqa"])
    >>> items = pipeline.run()
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medrl.core.hashing import hash_obj
from medrl.core.logging import get_logger
from medrl.data.decontam import ngram_decontam
from medrl.data.dedup import hybrid_dedup
from medrl.data.filters import FilterPipeline, LanguageFilter, LengthFilter
from medrl.data.schemas import DataStats, PipelineState

logger = get_logger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a data pipeline run.

    All parameters are frozen for reproducibility. The config hash is used for
    caching pipeline outputs.

    Attributes:
        name: Pipeline identifier.
        sources: List of (name, path) tuples for data sources.
        dedup_enabled: Whether to run deduplication.
        dedup_threshold: MinHash similarity threshold.
        decontam_enabled: Whether to run decontamination.
        decontam_benchmarks: List of benchmark names for decontamination.
        decontam_ngram_n: N-gram size for decontamination.
        filter_enabled: Whether to run quality filters.
        filter_min_length: Minimum text length.
        filter_languages: Allowed language codes.
        cache_dir: Directory for caching pipeline outputs.
        output_format: Output format ("jsonl", "parquet", "json").

    Examples:
        >>> config = PipelineConfig(
        ...     name="sft-medqa",
        ...     sources=[("medqa", "data/medqa.jsonl")],
        ...     dedup_threshold=0.9
        ... )
        >>> config.hash
        'a1b2c3d4...'
    """

    name: str = "pipeline"
    sources: tuple[tuple[str, str], ...] = ()
    dedup_enabled: bool = True
    dedup_threshold: float = 0.9
    decontam_enabled: bool = True
    decontam_benchmarks: tuple[str, ...] = ()
    decontam_ngram_n: int = 13
    filter_enabled: bool = True
    filter_min_length: int = 10
    filter_languages: tuple[str, ...] = ("en",)
    cache_dir: str | None = None
    output_format: str = "jsonl"

    @property
    def hash(self) -> str:
        """Compute hash of this config for caching."""
        return hash_obj(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "sources": list(self.sources),
            "dedup_enabled": self.dedup_enabled,
            "dedup_threshold": self.dedup_threshold,
            "decontam_enabled": self.decontam_enabled,
            "decontam_benchmarks": list(self.decontam_benchmarks),
            "decontam_ngram_n": self.decontam_ngram_n,
            "filter_enabled": self.filter_enabled,
            "filter_min_length": self.filter_min_length,
            "filter_languages": list(self.filter_languages),
            "cache_dir": self.cache_dir,
            "output_format": self.output_format,
        }


# =============================================================================
# Main Pipeline Class
# =============================================================================


class DataPipeline:
    """Main data processing pipeline.

    Orchestrates loading, deduplication, decontamination, filtering, and
    output generation. Supports caching for reproducibility.

    Attributes:
        config: Pipeline configuration.
        state: Current pipeline state (items, stats, etc.).

    Examples:
        >>> pipeline = DataPipeline(name="sft-medqa")
        >>> pipeline.add_source("medqa", path="data/medqa.jsonl")
        >>> pipeline.run()
        [...]
    """

    def __init__(self, name: str = "pipeline", config: PipelineConfig | None = None) -> None:
        """Initialize a new data pipeline.

        Args:
            name: Pipeline identifier.
            config: Optional pipeline configuration.
        """
        self.config = config or PipelineConfig(name=name)
        self.state = PipelineState(
            items=[],
            stats=DataStats(),
            sources=(),
            config_hash=self.config.hash,
        )
        self._eval_texts: dict[str, list[str]] = {}  # benchmark -> texts

    def add_source(self, name: str, path: str, version: str = "unknown") -> DataPipeline:
        """Add a data source to the pipeline.

        Args:
            name: Source identifier.
            path: Path to the data file.
            version: Optional version string.

        Returns:
            Self for chaining.
        """
        sources = [*list(self.config.sources), (name, path)]
        object.__setattr__(self.config, "sources", tuple(sources))
        return self

    def add_benchmark(self, name: str, texts: Sequence[str]) -> DataPipeline:
        """Add a benchmark for decontamination.

        Args:
            name: Benchmark identifier.
            texts: Benchmark texts.

        Returns:
            Self for chaining.
        """
        self._eval_texts[name] = list(texts)
        benchmarks = list(self.config.decontam_benchmarks)
        if name not in benchmarks:
            benchmarks.append(name)
            object.__setattr__(self.config, "decontam_benchmarks", tuple(benchmarks))
        return self

    def load(self, items: Iterable[dict[str, Any]]) -> DataPipeline:
        """Load items into the pipeline.

        Args:
            items: Items to load.

        Returns:
            Self for chaining.
        """
        items_list = list(items)
        # Create new state with loaded items
        new_stats = DataStats(total_items=len(items_list))
        object.__setattr__(
            self, "state", PipelineState(items=items_list, stats=new_stats, sources=self.state.sources, config_hash=self.config.hash)
        )
        logger.info(f"Loaded {len(items_list)} items into pipeline")
        return self

    def dedup(self, threshold: float | None = None) -> DataPipeline:
        """Apply deduplication to the current items.

        Args:
            threshold: MinHash threshold (overrides config).

        Returns:
            Self for chaining.
        """
        if not self.config.dedup_enabled:
            logger.info("Deduplication disabled, skipping")
            return self

        threshold = threshold or self.config.dedup_threshold

        before = len(self.state.items)
        deduped, _meta = hybrid_dedup(
            self.state.items,
            text_key="text",
            minhash_threshold=threshold,
            semantic_enabled=False,
        )

        deduped_count = before - len(deduped)

        # Create new stats with updated dedup count
        new_stats = DataStats(
            total_items=self.state.stats.total_items,
            deduped=self.state.stats.deduped + deduped_count,
            decontaminated=self.state.stats.decontaminated,
            filtered=self.state.stats.filtered,
            avg_length=self.state.stats.avg_length,
            language_dist=self.state.stats.language_dist,
        )

        # Create new state
        object.__setattr__(
            self, "state", PipelineState(items=deduped, stats=new_stats, sources=self.state.sources, config_hash=self.config.hash)
        )

        logger.info(f"Deduplication: {before} -> {len(deduped)} items ({deduped_count} removed)")

        return self

    def decontam(self, benchmarks: Sequence[str] | None = None) -> DataPipeline:
        """Apply decontamination against benchmarks.

        Args:
            benchmarks: Benchmarks to check (overrides config).

        Returns:
            Self for chaining.
        """
        if not self.config.decontam_enabled:
            logger.info("Decontamination disabled, skipping")
            return self

        benchmarks = benchmarks or self.config.decontam_benchmarks

        if not benchmarks:
            logger.warning("No benchmarks specified for decontamination")
            return self

        # Collect all eval texts
        all_eval_texts: list[str] = []
        for bench in benchmarks:
            if bench in self._eval_texts:
                all_eval_texts.extend(self._eval_texts[bench])
            else:
                logger.warning(f"Benchmark '{bench}' not loaded, skipping")

        if not all_eval_texts:
            logger.warning("No eval texts available for decontamination")
            return self

        from medrl.data.decontam import TrainDocument

        before = len(self.state.items)
        # Convert dict items to TrainDocument objects
        train_docs = [
            TrainDocument(
                id=str(i),
                text=item.get("text", ""),
                source=item.get("source", "unknown"),
            )
            for i, item in enumerate(self.state.items)
        ]
        clean_docs, _contaminated = ngram_decontam(
            train_docs,
            all_eval_texts,
            threshold=0.8,  # Default contamination threshold
        )

        decontam_count = before - len(clean_docs)

        # Create new stats with updated decontam count
        new_stats = DataStats(
            total_items=self.state.stats.total_items,
            deduped=self.state.stats.deduped,
            decontaminated=self.state.stats.decontaminated + decontam_count,
            filtered=self.state.stats.filtered,
            avg_length=self.state.stats.avg_length,
            language_dist=self.state.stats.language_dist,
        )

        # Convert TrainDocument back to dict items
        clean_items = [self.state.items[int(doc.id)] for doc in clean_docs]

        # Create new state
        object.__setattr__(
            self, "state", PipelineState(items=clean_items, stats=new_stats, sources=self.state.sources, config_hash=self.config.hash)
        )

        logger.info(
            f"Decontamination: {before} -> {len(clean_docs)} items ({decontam_count} removed)"
        )

        return self

    def filter(self, pipeline: FilterPipeline | None = None) -> DataPipeline:
        """Apply quality filters.

        Args:
            pipeline: Filter pipeline (uses default if None).

        Returns:
            Self for chaining.
        """
        if not self.config.filter_enabled:
            logger.info("Filtering disabled, skipping")
            return self

        if pipeline is None:
            pipeline = FilterPipeline()
            pipeline.add("length", LengthFilter(min_length=self.config.filter_min_length))
            if self.config.filter_languages:
                pipeline.add(
                    "language", LanguageFilter(allowed=tuple(self.config.filter_languages))
                )

        before = len(self.state.items)
        filtered = pipeline.filter(self.state.items)
        filtered_count = before - len(filtered)

        # Create new stats with updated filter count
        new_stats = DataStats(
            total_items=self.state.stats.total_items,
            deduped=self.state.stats.deduped,
            decontaminated=self.state.stats.decontaminated,
            filtered=self.state.stats.filtered + filtered_count,
            avg_length=self.state.stats.avg_length,
            language_dist=self.state.stats.language_dist,
        )

        # Create new state
        object.__setattr__(
            self, "state", PipelineState(items=filtered, stats=new_stats, sources=self.state.sources, config_hash=self.config.hash)
        )

        logger.info(f"Filtering: {before} -> {len(filtered)} items ({filtered_count} removed)")

        return self

    def run(self) -> list[dict[str, Any]]:
        """Run the complete pipeline.

        Args:
            ...

        Returns:
            Processed items.
        """
        logger.info(f"Running pipeline '{self.config.name}'")
        logger.info(f"Config: {self.config.to_dict()}")

        # Apply pipeline stages
        self.dedup()
        self.decontam()
        self.filter()

        final_count = len(self.state.items)
        logger.info(f"Pipeline complete: {final_count} items retained")

        return self.state.items

    def save(self, path: str) -> None:
        """Save pipeline output to file.

        Args:
            path: Output path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        format = self.config.output_format

        if format == "jsonl":
            import json

            with path_obj.open("w") as f:
                for item in self.state.items:
                    f.write(json.dumps(item) + "\n")

        elif format == "json":
            import json

            with path_obj.open("w") as f:
                json.dump(self.state.items, f, indent=2)

        elif format == "parquet":
            try:
                import pyarrow as pa  # type: ignore[import-untyped]
                import pyarrow.parquet as pq  # type: ignore[import-untyped]
            except ImportError as e:
                raise ImportError(
                    "pyarrow is required for Parquet output. "
                    "Install with: pip install pyarrow"
                ) from e

            table = pa.table(self.state.items)
            pq.write_table(table, path_obj)

        else:
            raise ValueError(f"Unknown output format: {format}")

        logger.info(f"Saved {len(self.state.items)} items to {path}")

    def get_stats(self) -> DataStats:
        """Get pipeline statistics."""
        return self.state.stats

    def get_report(self) -> dict[str, Any]:
        """Get a detailed pipeline report.

        Returns:
            Report dictionary with stats and config.
        """
        return {
            "config": self.config.to_dict(),
            "stats": {
                "total_items": self.state.stats.total_items,
                "deduped": self.state.stats.deduped,
                "decontaminated": self.state.stats.decontaminated,
                "filtered": self.state.stats.filtered,
                "retained": self.state.stats.retained,
                "retention_rate": self.state.stats.retention_rate,
            },
            "config_hash": self.config.hash,
        }


# =============================================================================
# Pipeline Builder
# =============================================================================


class PipelineBuilder:
    """Builder for creating configured pipelines.

    Provides a fluent interface for constructing pipelines with common presets.

    Examples:
        >>> builder = PipelineBuilder.sft(name="medqa-sft")
        >>> builder.add_source("medqa", "data/medqa.jsonl")
        >>> pipeline = builder.build()
    """

    @classmethod
    def sft(cls, name: str = "sft-pipeline") -> PipelineBuilder:
        """Create a builder for SFT data pipeline.

        Args:
            name: Pipeline name.

        Returns:
            PipelineBuilder instance.
        """
        config = PipelineConfig(
            name=name,
            dedup_enabled=True,
            dedup_threshold=0.9,
            decontam_enabled=True,
            filter_enabled=True,
            filter_min_length=10,
        )
        return cls(config)

    @classmethod
    def preference(cls, name: str = "pref-pipeline") -> PipelineBuilder:
        """Create a builder for preference (RLHF/DPO) pipeline.

        Args:
            name: Pipeline name.

        Returns:
            PipelineBuilder instance.
        """
        config = PipelineConfig(
            name=name,
            dedup_enabled=True,
            dedup_threshold=0.85,  # More strict for preference data
            decontam_enabled=True,
            filter_enabled=True,
            filter_min_length=20,  # Longer minimum for preferences
        )
        return cls(config)

    @classmethod
    def rl(cls, name: str = "rl-pipeline") -> PipelineBuilder:
        """Create a builder for RL rollout pipeline.

        Args:
            name: Pipeline name.

        Returns:
            PipelineBuilder instance.
        """
        config = PipelineConfig(
            name=name,
            dedup_enabled=False,  # No dedup for RL prompts
            decontam_enabled=True,
            filter_enabled=True,
            filter_min_length=5,
        )
        return cls(config)

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize builder with config.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self._sources: list[tuple[str, str]] = []
        self._benchmarks: dict[str, list[str]] = {}

    def add_source(self, name: str, path: str) -> PipelineBuilder:
        """Add a data source.

        Args:
            name: Source name.
            path: Source path.

        Returns:
            Self for chaining.
        """
        self._sources.append((name, path))
        return self

    def add_benchmark(self, name: str, texts: Sequence[str]) -> PipelineBuilder:
        """Add a benchmark for decontamination.

        Args:
            name: Benchmark name.
            texts: Benchmark texts.

        Returns:
            Self for chaining.
        """
        self._benchmarks[name] = list(texts)
        return self

    def with_dedup_threshold(self, threshold: float) -> PipelineBuilder:
        """Set deduplication threshold.

        Args:
            threshold: MinHash threshold.

        Returns:
            Self for chaining.
        """
        object.__setattr__(self.config, "dedup_threshold", threshold)
        return self

    def with_filter(self, min_length: int, languages: tuple[str, ...] = ("en",)) -> PipelineBuilder:
        """Set filter parameters.

        Args:
            min_length: Minimum text length.
            languages: Allowed languages.

        Returns:
            Self for chaining.
        """
        object.__setattr__(self.config, "filter_min_length", min_length)
        object.__setattr__(self.config, "filter_languages", languages)
        return self

    def build(self) -> DataPipeline:
        """Build the pipeline.

        Returns:
            Configured DataPipeline instance.
        """
        # Update config with sources
        object.__setattr__(self.config, "sources", tuple(self._sources))

        pipeline = DataPipeline(config=self.config)

        # Add benchmarks
        for name, texts in self._benchmarks.items():
            pipeline.add_benchmark(name, texts)

        return pipeline
