"""Curation pipeline: S0-S15 corpus build for the medrl training program.

Public surface: the schema contracts, the storage layer, and the stage runner.
Stage modules live in :mod:`medrl.curation.stages` and are imported by the
runner, never by each other -- a stage reads a snapshot directory and writes one.
"""

from medrl.curation.schema import (
    AnswerType,
    CorpusItem,
    DifficultyBand,
    Flags,
    Message,
    SourceRecord,
    StageError,
    StageManifest,
)
from medrl.curation.thresholds import THRESHOLDS, Thresholds, snapshot

__all__ = [
    "THRESHOLDS",
    "AnswerType",
    "CorpusItem",
    "DifficultyBand",
    "Flags",
    "Message",
    "SourceRecord",
    "StageError",
    "StageManifest",
    "Thresholds",
    "snapshot",
]
