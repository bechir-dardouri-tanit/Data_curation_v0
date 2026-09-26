"""Main data pipeline exports for medrl.

This module provides the complete data pipeline infrastructure including:

- Schemas: Core data structures (SFTItem, PrefPair, RLPrompt, etc.)
- Deduplication: MinHash and semantic duplicate removal
- Decontamination: Benchmark contamination detection
- Filters: Quality filters and validators
- Pipeline: Main data processing orchestration
- Generation: Teacher model generation and distillation
- Sources: Data source loaders and registries

Quick start:
    >>> from medrl.data import DataPipeline, load_source
    >>>
    >>> # Load and process data
    >>> items = load_source("data/medqa.jsonl")
    >>> pipeline = DataPipeline(name="medqa-sft")
    >>> pipeline.load(items)
    >>> pipeline.dedup(threshold=0.9)
    >>> clean_items = pipeline.run()

For SFT data:
    >>> from medrl.data import SFTItem
    >>> item = SFTItem(
    ...     messages=[Message(role="user", content="What is aspirin?")],
    ...     response="Aspirin is a NSAID...",
    ...     source="medqa"
    ... )

For preference (RLHF/DPO) data:
    >>> from medrl.data import PrefPair
    >>> pair = PrefPair(
    ...     prompt=[Message(role="user", content="Explain dosage.")],
    ...     chosen="Consult label...",
    ...     rejected="Take 1000mg hourly...",
    ...     source="medmcqa"
    ... )

For deduplication:
    >>> from medrl.data import minhash_dedup
    >>> deduped, hashes = minhash_dedup(items, threshold=0.9)

For decontamination:
    >>> from medrl.data import ngram_decontam
    >>> clean, dirty = ngram_decontam(train_items, eval_texts)

For quality filtering:
    >>> from medrl.data import FilterPipeline, LengthFilter
    >>> pipeline = FilterPipeline()
    >>> pipeline.add("length", LengthFilter(min_length=10))
    >>> filtered = pipeline.filter(items)
"""

from __future__ import annotations

# =============================================================================
# Decontamination
# =============================================================================
from medrl.data.decontam import (
    BenchmarkDecontaminator,
    exact_decontam,
    generate_decontam_report,
    ngram_decontam,
    semantic_decontam,
)

# =============================================================================
# Deduplication
# =============================================================================
from medrl.data.dedup import (
    cluster_dedup,
    exact_dedup,
    hybrid_dedup,
    minhash_dedup,
    ngram_dedup,
    semantic_dedup,
)

# =============================================================================
# Filters
# =============================================================================
from medrl.data.filters import (
    BoilerplateFilter,
    Filter,
    FilterPipeline,
    LanguageFilter,
    LengthFilter,
    MedicalTerminologyFilter,
    PIIFilter,
    RatioFilter,
    RegexFilter,
    WordCountFilter,
    validate_item,
    validate_message_format,
    validate_response_text,
)

# =============================================================================
# Generation
# =============================================================================
from medrl.data.generation import (
    GenerationConfig,
    PromptTemplate,
    TeacherGenerator,
    augment_dataset,
    extract_rationales,
    generate_preference_pairs,
    generate_sft_data,
)

# =============================================================================
# Pipeline
# =============================================================================
from medrl.data.pipeline import (
    DataPipeline,
    PipelineBuilder,
    PipelineConfig,
)

# =============================================================================
# Core schemas
# =============================================================================
from medrl.data.schemas import (
    DataSource,
    DataStats,
    Message,
    PipelineState,
    PrefPair,
    RLPrompt,
    RLRollout,
    SFTItem,
)

# =============================================================================
# Sources
# =============================================================================
from medrl.data.sources import (
    SourceInfo,
    inspect_source,
    load_directory,
    load_huggingface,
    load_json,
    load_jsonl,
    load_parquet,
    load_source,
    load_txt,
    loaders,
    stream_jsonl,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "BenchmarkDecontaminator",
    "BoilerplateFilter",
    # Pipeline
    "DataPipeline",
    "DataSource",
    "DataStats",
    # Filters
    "Filter",
    "FilterPipeline",
    "GenerationConfig",
    "LanguageFilter",
    "LengthFilter",
    "MedicalTerminologyFilter",
    # Schemas
    "Message",
    "PIIFilter",
    "PipelineBuilder",
    "PipelineConfig",
    "PipelineState",
    "PrefPair",
    "PromptTemplate",
    "RLPrompt",
    "RLRollout",
    "RatioFilter",
    "RegexFilter",
    "SFTItem",
    "SourceInfo",
    # Generation
    "TeacherGenerator",
    "WordCountFilter",
    "augment_dataset",
    "cluster_dedup",
    "exact_decontam",
    # Deduplication
    "exact_dedup",
    "extract_rationales",
    "generate_decontam_report",
    "generate_preference_pairs",
    "generate_sft_data",
    "hybrid_dedup",
    "inspect_source",
    "load_directory",
    "load_huggingface",
    "load_json",
    # Sources
    "load_jsonl",
    "load_parquet",
    "load_source",
    "load_txt",
    "loaders",
    "minhash_dedup",
    # Decontamination
    "ngram_decontam",
    "ngram_dedup",
    "semantic_decontam",
    "semantic_dedup",
    "stream_jsonl",
    "validate_item",
    "validate_message_format",
    "validate_response_text",
]
