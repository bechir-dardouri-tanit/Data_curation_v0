"""Contamination detection for evaluation benchmarks.

Pre-eval contamination scan using n-gram overlap analysis between training data
and evaluation benchmarks. Implements 13-gram overlap detection as recommended in
the literature for identifying potential memorization or data leakage.

This module provides:
- Pre-eval contamination scan
- 13-gram overlap detector (configurable n-gram size)
- Negative control generation
- Per-benchmark reports
- Integration with eval runner as pre-flight check
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger
from medrl.eval.items import EvalItem

log = get_logger(__name__)


class ContaminationSeverity(StrEnum):
    """Severity levels for contamination findings."""
    CRITICAL = "critical"  # >50% overlap or long exact matches
    HIGH = "high"  # 20-50% overlap
    MEDIUM = "medium"  # 5-20% overlap
    LOW = "low"  # 1-5% overlap
    NONE = "none"  # <1% overlap


@dataclass(frozen=True)
class NGramStats:
    """N-gram overlap statistics for a single benchmark item."""
    item_id: str
    n_gram_size: int
    # Number of unique n-grams in the evaluation item
    eval_ngrams: int
    # Number of n-grams that appear in training data
    overlapping_ngrams: int
    # Longest exact substring match (in characters)
    longest_match: int
    # Fraction of n-grams from eval that appear in training
    overlap_fraction: float

    @property
    def severity(self) -> ContaminationSeverity:
        """Determine severity based on overlap statistics."""
        if self.overlap_fraction > 0.5 or self.longest_match > 200:
            return ContaminationSeverity.CRITICAL
        if self.overlap_fraction > 0.2 or self.longest_match > 100:
            return ContaminationSeverity.HIGH
        if self.overlap_fraction > 0.05 or self.longest_match > 50:
            return ContaminationSeverity.MEDIUM
        if self.overlap_fraction > 0.01:
            return ContaminationSeverity.LOW
        return ContaminationSeverity.NONE


@dataclass(frozen=True)
class BenchmarkContaminationReport:
    """Contamination report for a single benchmark."""
    benchmark: str
    # Total items scanned
    total_items: int
    # Items with any overlap
    items_with_overlap: int
    # Severity breakdown
    severity_counts: dict[str, int] = field(default_factory=dict)
    # Per-item statistics
    item_stats: list[NGramStats] = field(default_factory=list)
    # Training data sources checked
    training_sources: list[str] = field(default_factory=list)
    # N-gram size used
    n_gram_size: int = 13

    @property
    def contamination_rate(self) -> float:
        """Fraction of items with any contamination."""
        if self.total_items == 0:
            return 0.0
        return self.items_with_overlap / self.total_items

    @property
    def has_critical_contamination(self) -> bool:
        """Whether any items have critical contamination."""
        return self.severity_counts.get("critical", 0) > 0


@dataclass
class NegativeControl:
    """A negative control item for testing memorization."""
    control_id: str
    original_benchmark: str
    original_item_id: str
    messages: tuple[dict[str, str], ...]
    # The perturbation applied
    perturbation: str
    # Expected answer (different from original)
    expected_diff: str


@dataclass
class DecontaminationConfig:
    """Configuration for contamination detection."""
    # N-gram size for overlap detection (13 is standard in literature)
    n_gram_size: int = 13
    # Minimum overlap length to report (in characters)
    min_match_length: int = 20
    # Thresholds for severity
    critical_threshold: float = 0.5
    high_threshold: float = 0.2
    medium_threshold: float = 0.05
    low_threshold: float = 0.01
    # Whether to generate negative controls
    generate_negative_controls: bool = True
    # Number of negative controls per contaminated item
    negative_controls_per_item: int = 1
    # Whether to fail the run on critical contamination
    fail_on_critical: bool = True
    # Paths to training data to check against
    training_data_paths: list[str] = field(default_factory=list)


class NGramOverlapDetector:
    """Detects n-gram overlap between evaluation items and training data.

    Uses a sliding window approach to extract n-grams and builds an index
    for efficient lookup. The 13-gram size is based on research showing that
    shorter n-grams are too common while longer ones miss subtle overlaps.
    """

    def __init__(self, n: int = 13, min_match_length: int = 20):
        """Initialize the detector.

        Args:
            n: N-gram size (default 13, based on literature)
            min_match_length: Minimum character length for exact match reporting
        """
        self.n = n
        self.min_match_length = min_match_length
        self._training_ngrams: set[str] = set()
        self._training_texts: list[str] = []
        self._sources: list[str] = []

    def index_training_data(self, texts: list[str], source: str = "unknown") -> None:
        """Index training texts for n-gram lookup.

        Args:
            texts: List of training text samples (prompts, questions, etc.)
            source: Identifier for the data source
        """
        log.info("Indexing %d training samples from %s", len(texts), source)
        self._sources.append(source)
        self._training_texts.extend(texts)

        for text in texts:
            normalized = self._normalize(text)
            ngrams = self._extract_ngrams(normalized)
            self._training_ngrams.update(ngrams)

        log.info(
            "Training index now has %d unique %d-grams from %d samples",
            len(self._training_ngrams), self.n, len(self._training_texts)
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase and collapse whitespace
        return " ".join(text.lower().split())

    def _extract_ngrams(self, text: str) -> set[str]:
        """Extract all n-grams from normalized text.

        Uses word-level n-grams for texts with enough words, falling back to
        character-level n-grams for short texts. This ensures detection works
        even for very short items.
        """
        words = text.split()
        if len(words) >= self.n:
            return {" ".join(words[i:i + self.n]) for i in range(len(words) - self.n + 1)}
        # Fallback: use character n-grams for short texts
        return {text[i:i + self.n] for i in range(len(text) - self.n + 1)}

    def _find_longest_match(self, eval_text: str, training_text: str) -> int:
        """Find the longest common substring using dynamic programming."""
        eval_norm = self._normalize(eval_text)
        train_norm = self._normalize(training_text)

        # Use a more efficient approach for long texts
        if len(eval_norm) * len(train_norm) > 100_000:
            # For long texts, use a sliding window approach
            return self._find_longest_match_window(eval_norm, train_norm)

        # DP approach for shorter texts
        m = [[0] * (len(train_norm) + 1) for _ in range(len(eval_norm) + 1)]
        longest = 0

        for i in range(1, len(eval_norm) + 1):
            for j in range(1, len(train_norm) + 1):
                if eval_norm[i - 1] == train_norm[j - 1]:
                    m[i][j] = m[i - 1][j - 1] + 1
                    longest = max(longest, m[i][j])
                else:
                    m[i][j] = 0

        return longest

    def _find_longest_match_window(self, eval_text: str, training_text: str) -> int:
        """Find longest match using sliding window for efficiency."""
        max_match = 0
        # Try windows from eval text in training text
        for window_size in [200, 100, 50, 30, 20]:
            if window_size > len(eval_text):
                continue
            for i in range(len(eval_text) - window_size + 1):
                window = eval_text[i:i + window_size]
                if window in training_text:
                    # Expand to find full match
                    start = i
                    end = i + window_size
                    while start > 0 and eval_text[start - 1] in training_text:
                        start -= 1
                    while end < len(eval_text) and eval_text[end] in training_text:
                        end += 1
                    max_match = max(max_match, end - start)
                    if max_match >= 200:
                        return max_match
        return max_match

    def check_item(self, item: EvalItem) -> NGramStats:
        """Check a single evaluation item for contamination.

        Args:
            item: The evaluation item to check

        Returns:
            NGramStats with overlap information
        """
        # Extract text from messages
        item_text = self._extract_text_from_item(item)

        if not item_text:
            return NGramStats(
                item_id=item.item_id,
                n_gram_size=self.n,
                eval_ngrams=0,
                overlapping_ngrams=0,
                longest_match=0,
                overlap_fraction=0.0,
            )

        normalized = self._normalize(item_text)
        eval_ngrams = self._extract_ngrams(normalized)

        # Find overlapping n-grams
        overlapping = eval_ngrams & self._training_ngrams

        # Find longest exact match
        longest_match = 0
        for training_text in self._training_texts:
            match = self._find_longest_match(item_text, training_text)
            longest_match = max(longest_match, match)
            if longest_match > 200:  # Early exit for very long matches
                break

        overlap_fraction = len(overlapping) / len(eval_ngrams) if eval_ngrams else 0.0

        return NGramStats(
            item_id=item.item_id,
            n_gram_size=self.n,
            eval_ngrams=len(eval_ngrams),
            overlapping_ngrams=len(overlapping),
            longest_match=longest_match,
            overlap_fraction=overlap_fraction,
        )

    def _extract_text_from_item(self, item: EvalItem) -> str:
        """Extract all text content from an evaluation item."""
        texts = []
        for msg in item.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
        return " ".join(texts)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the training index."""
        return {
            "n_gram_size": self.n,
            "unique_ngrams": len(self._training_ngrams),
            "training_samples": len(self._training_texts),
            "sources": list(set(self._sources)),
        }


def load_training_texts(
    paths: list[str] | list[Path],
    max_samples: int | None = None,
    text_fields: list[str] | None = None,
) -> list[str]:
    """Load training texts from various data sources.

    Args:
        paths: List of paths to training data (JSONL, JSON, or directories)
        max_samples: Maximum number of samples to load (None = unlimited)
        text_fields: Field names to extract text from (for JSONL/JSON)

    Returns:
        List of text samples from training data
    """
    texts = []
    seen = 0

    default_text_fields = ["prompt", "question", "instruction", "text", "content"]

    for path in paths:
        path = Path(path)
        if not path.exists():
            log.warning("Training data path does not exist: %s", path)
            continue

        if path.is_file():
            texts.extend(_load_texts_from_file(
                path, max_samples, text_fields or default_text_fields
            ))
        elif path.is_dir():
            for file in path.glob("*.jsonl"):
                texts.extend(_load_texts_from_file(
                    file, max_samples, text_fields or default_text_fields
                ))
                if max_samples and len(texts) >= max_samples:
                    break
            if max_samples and len(texts) >= max_samples:
                break

        seen = len(texts)
        if max_samples and seen >= max_samples:
            log.info("Reached max_samples limit (%d)", max_samples)
            texts = texts[:max_samples]
            break

    log.info("Loaded %d training texts", len(texts))
    return texts


def _load_texts_from_file(
    path: Path,
    max_samples: int | None,
    text_fields: list[str],
) -> list[str]:
    """Load texts from a single file."""
    texts: list[str] = []

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if max_samples and len(texts) >= max_samples:
                    break
                try:
                    record = json.loads(line)
                    text = _extract_text_from_record(record, text_fields)
                    if text:
                        texts.append(text)
                except (json.JSONDecodeError, KeyError):
                    continue

    elif path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            if max_samples and len(texts) >= max_samples:
                break
            text = _extract_text_from_record(record, text_fields)
            if text:
                texts.append(text)

    return texts


def _extract_text_from_record(record: dict[str, Any], text_fields: list[str]) -> str | None:
    """Extract text content from a data record."""
    # Try each field in order
    for field_name in text_fields:
        if field_name in record:
            value = record[field_name]
            if isinstance(value, str) and value.strip():
                return value

    # Handle nested message structures
    if "messages" in record:
        messages = record["messages"]
        if isinstance(messages, list):
            texts = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, str):
                        texts.append(content)
            if texts:
                return " ".join(texts)

    # Handle conversation structures
    if "conversations" in record:
        convs = record["conversations"]
        if isinstance(convs, list):
            texts = []
            for turn in convs:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    texts.append(str(turn[1]))
            if texts:
                return " ".join(texts)

    return None


def scan_benchmark(
    items: list[EvalItem],
    detector: NGramOverlapDetector,
    config: DecontaminationConfig,
) -> BenchmarkContaminationReport:
    """Scan a benchmark for contamination.

    Args:
        items: Evaluation items to scan
        detector: Configured n-gram overlap detector
        config: Decontamination configuration

    Returns:
        BenchmarkContaminationReport with findings
    """
    benchmark_name = items[0].benchmark if items else "unknown"

    log.info("Scanning benchmark %s with %d items", benchmark_name, len(items))

    item_stats: list[NGramStats] = []
    severity_counts: dict[str, int] = defaultdict(int)

    for item in items:
        stats = detector.check_item(item)
        item_stats.append(stats)

        if stats.overlap_fraction > 0:
            severity_counts[stats.severity.value] += 1

    report = BenchmarkContaminationReport(
        benchmark=benchmark_name,
        total_items=len(items),
        items_with_overlap=sum(
            1 for s in item_stats if s.overlap_fraction > 0
        ),
        severity_counts=dict(severity_counts),
        item_stats=[s for s in item_stats if s.overlap_fraction > 0],
        training_sources=detector.get_stats()["sources"],
        n_gram_size=config.n_gram_size,
    )

    log.info(
        "Benchmark %s: %d/%d items with overlap (%.1f%%), critical=%d",
        benchmark_name,
        report.items_with_overlap,
        report.total_items,
        report.contamination_rate * 100,
        report.severity_counts.get("critical", 0),
    )

    return report


def generate_negative_controls(
    contaminated_items: list[EvalItem],
    stats: list[NGramStats],
    controls_per_item: int = 1,
) -> list[NegativeControl]:
    """Generate negative controls for contaminated items.

    Creates perturbed versions of contaminated items to test whether
    the model truly knows the material or is just memorizing.

    Args:
        contaminated_items: Items with detected contamination
        stats: Corresponding n-gram statistics
        controls_per_item: Number of controls to generate per item

    Returns:
        List of negative control items
    """
    controls = []

    for item, stat in zip(contaminated_items, stats, strict=False):
        if stat.severity == ContaminationSeverity.NONE:
            continue

        # Generate different perturbations
        perturbations = _generate_perturbations(item, controls_per_item)

        for i, (perturbed_msg, perturbation_desc) in enumerate(perturbations):
            # For MCQA, the expected answer should change
            expected_diff = _get_perturbed_expected_answer(item, i)

            control = NegativeControl(
                control_id=f"{item.benchmark}::{item.item_id}::negctrl::{i}",
                original_benchmark=item.benchmark,
                original_item_id=item.item_id,
                messages=tuple(perturbed_msg),
                perturbation=perturbation_desc,
                expected_diff=expected_diff,
            )
            controls.append(control)

    log.info("Generated %d negative controls", len(controls))
    return controls


def _generate_perturbations(item: EvalItem, n: int) -> list[tuple[list[dict[str, str]], str]]:
    """Generate perturbed versions of an item."""
    perturbations: list[tuple[list[dict[str, str]], str]] = []
    messages = [dict(m) for m in item.messages]

    # Perturbation 1: Add distractor context
    if n >= 1 and messages:
        perturbed = messages.copy()
        last_msg = perturbed[-1].copy()
        if "content" in last_msg:
            last_msg["content"] = (
                "CONTEXT: Consider this additional information: "
                "Recent studies have updated clinical guidelines. "
                "\n\n" + last_msg["content"]
            )
            perturbed[-1] = last_msg
            perturbations.append((perturbed, "distractor_context"))

    # Perturbation 2: Reverse the question
    if n >= 2 and len(messages) >= 2:
        perturbed = messages.copy()
        last_msg = perturbed[-1].copy()
        if "content" in last_msg:
            content = last_msg["content"]
            # Add a reverse framing
            last_msg["content"] = (
                "INVERTED: What if the opposite were true? "
                + content
            )
            perturbed[-1] = last_msg
            perturbations.append((perturbed, "reverse_framing"))

    # Perturbation 3: Add noise
    if n >= 3 and messages:
        perturbed = messages.copy()
        last_msg = perturbed[-1].copy()
        if "content" in last_msg:
            last_msg["content"] = (
                "NOISE_TEST: " + last_msg["content"] + " "
                "(Ignore this repeated text: noise test noise test)"
            )
            perturbed[-1] = last_msg
            perturbations.append((perturbed, "noise_injection"))

    return perturbations[:n]


def _get_perturbed_expected_answer(item: EvalItem, perturbation_idx: int) -> str:
    """Get the expected answer for a perturbed item."""
    # For negative controls, we expect the model to handle the perturbation
    # This is simplified; in practice, you'd need more sophisticated logic
    return "perturbed"


def run_decontamination_check(
    benchmarks: dict[str, list[EvalItem]],
    training_paths: list[str] | None = None,
    config: DecontaminationConfig | None = None,
) -> dict[str, BenchmarkContaminationReport]:
    """Run full decontamination check across all benchmarks.

    This is the main entry point for contamination detection. It loads
    training data, builds an n-gram index, and scans all benchmarks.

    Args:
        benchmarks: Dict mapping benchmark names to their items
        training_paths: Paths to training data (if not in config)
        config: Decontamination configuration

    Returns:
        Dict mapping benchmark names to contamination reports
    """
    if config is None:
        config = DecontaminationConfig()

    if training_paths:
        config = DecontaminationConfig(
            **{**asdict(config), "training_data_paths": training_paths}
        )

    log.info("Starting decontamination check with config: %s", config)

    # Load training data
    training_texts = load_training_texts(
        config.training_data_paths,
        max_samples=None,  # Load all available training data
    )

    if not training_texts:
        log.warning("No training data loaded; contamination check will be empty")
        return {
            name: BenchmarkContaminationReport(
                benchmark=name,
                total_items=len(items),
                items_with_overlap=0,
                training_sources=[],
                n_gram_size=config.n_gram_size,
            )
            for name, items in benchmarks.items()
        }

    # Build detector
    detector = NGramOverlapDetector(
        n=config.n_gram_size,
        min_match_length=config.min_match_length,
    )
    detector.index_training_data(training_texts, source="training_data")

    # Scan each benchmark
    reports: dict[str, BenchmarkContaminationReport] = {}
    critical_found = False

    for name, items in benchmarks.items():
        report = scan_benchmark(items, detector, config)
        reports[name] = report

        if report.has_critical_contamination:
            critical_found = True
            log.error(
                "CRITICAL contamination found in %s: %d items exceed threshold",
                name,
                report.severity_counts.get("critical", 0),
            )

    # Check if we should fail
    if critical_found and config.fail_on_critical:
        critical_benchmarks = [
            name for name, r in reports.items()
            if r.has_critical_contamination
        ]
        raise RuntimeError(
            f"Critical contamination detected in {critical_benchmarks}. "
            "This indicates potential data leakage between training and evaluation. "
            "Review the contamination reports and either remove the overlapping "
            "training data or exclude the affected benchmark items. "
            "To proceed anyway, set fail_on_critical=False in the decontamination config."
        )

    return reports


def save_decontamination_report(
    reports: dict[str, BenchmarkContaminationReport],
    output_path: Path | str,
) -> None:
    """Save contamination reports to disk.

    Args:
        reports: Contamination reports by benchmark
        output_path: Path to save the report (JSON format)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        name: {
            **asdict(report),
            "item_stats": [asdict(s) for s in report.item_stats],
        }
        for name, report in reports.items()
    }

    output_path.write_text(json.dumps(serializable, indent=2))
    log.info("Saved contamination report to %s", output_path)


def summarize_for_human(reports: dict[str, BenchmarkContaminationReport]) -> str:
    """Generate a human-readable summary of contamination findings.

    Args:
        reports: Contamination reports by benchmark

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 80,
        "CONTAMINATION SCAN SUMMARY",
        "=" * 80,
        "",
    ]

    total_items = sum(r.total_items for r in reports.values())
    total_contaminated = sum(r.items_with_overlap for r in reports.values())

    lines.extend([
        f"Total items scanned: {total_items}",
        f"Items with any overlap: {total_contaminated} ({total_contaminated/total_items*100:.1f}%)",
        "",
        "-" * 80,
        "Per-benchmark breakdown:",
        "-" * 80,
        "",
    ])

    for name, report in sorted(reports.items()):
        lines.extend([
            f"{name}:",
            f"  Items: {report.items_with_overlap}/{report.total_items} "
            f"({report.contamination_rate*100:.1f}%)",
            f"  Severity: {dict(report.severity_counts)}",
            "",
        ])

    lines.extend([
        "-" * 80,
        "",
        "Legend:",
        "  critical: >50% n-gram overlap or >200 char exact match",
        "  high: 20-50% overlap or >100 char exact match",
        "  medium: 5-20% overlap or >50 char exact match",
        "  low: 1-5% overlap",
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


__all__ = [
    "BenchmarkContaminationReport",
    "ContaminationSeverity",
    "DecontaminationConfig",
    "NGramOverlapDetector",
    "NGramStats",
    "NegativeControl",
    "generate_negative_controls",
    "load_training_texts",
    "run_decontamination_check",
    "save_decontamination_report",
    "scan_benchmark",
    "summarize_for_human",
]
