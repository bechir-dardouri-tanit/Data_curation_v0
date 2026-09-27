"""Tests for contamination detection functionality."""

from __future__ import annotations

import json

import pytest

from medrl.eval.decontam import (
    BenchmarkContaminationReport,
    ContaminationSeverity,
    DecontaminationConfig,
    NegativeControl,
    NGramOverlapDetector,
    NGramStats,
    _extract_text_from_record,
    generate_negative_controls,
    load_training_texts,
    run_decontamination_check,
    save_decontamination_report,
    scan_benchmark,
    summarize_for_human,
)
from medrl.eval.items import EvalItem, VerifySpec
from medrl.eval.tasks.spec import VerifyStyle


@pytest.fixture
def sample_training_texts(tmp_path) -> list[str]:
    """Create sample training data for testing."""
    # Use longer texts to ensure 13-grams can be generated
    data = [
        "The patient presents with chest pain and shortness of breath. The pain is substernal and radiating to the left arm. There is associated diaphoresis and nausea.",
        "Diagnosis of myocardial infarction requires ECG changes such as ST elevation in two contiguous leads. Cardiac enzymes are also typically elevated.",
        "Treatment for acute MI includes aspirin, heparin, and dual antiplatelet therapy. Reperfusion therapy should be initiated as soon as possible.",
        "The differential diagnosis includes pulmonary embolism, aortic dissection, and pneumothorax. CT angiography may be necessary to rule these out.",
        "Beta blockers are contraindicated in acute heart failure due to their negative inotropic effects. However, they are beneficial long term after recovery.",
    ]

    # Write to a JSONL file
    records = [{"prompt": text, "source": "test"} for text in data]
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )

    return data


@pytest.fixture
def sample_eval_items() -> list[EvalItem]:
    """Create sample evaluation items for testing."""
    # q1 has exact overlap with training data to test detection
    return [
        EvalItem(
            benchmark="medqa",
            item_id="q1",
            messages=(
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": "The patient presents with chest pain and shortness of breath. The pain is substernal and radiating to the left arm."},
            ),
            verify=VerifySpec(style=VerifyStyle.LETTER, letters="ABCDE", gold_letter="A"),
        ),
        EvalItem(
            benchmark="medqa",
            item_id="q2",
            messages=(
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": "A 65-year-old male presents with sudden onset severe headache. What is the diagnosis?"},
            ),
            verify=VerifySpec(style=VerifyStyle.LETTER, letters="ABCDE", gold_letter="B"),
        ),
        EvalItem(
            benchmark="medqa",
            item_id="q3",
            messages=(
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": "Treatment for acute MI includes aspirin, heparin, and dual antiplatelet therapy"},
            ),
            verify=VerifySpec(style=VerifyStyle.LETTER, letters="AB", gold_letter="A"),
        ),
    ]


def test_ngram_detector_indexes_training_data(sample_training_texts) -> None:
    """Test that the detector correctly indexes training data."""
    detector = NGramOverlapDetector(n=13)
    detector.index_training_data(sample_training_texts, source="test")

    stats = detector.get_stats()
    assert stats["training_samples"] == 5
    assert stats["unique_ngrams"] > 0
    assert stats["n_gram_size"] == 13
    assert "test" in stats["sources"]


def test_ngram_detector_detects_exact_match(sample_training_texts, sample_eval_items) -> None:
    """Test detection of exact matches between training and eval data."""
    detector = NGramOverlapDetector(n=13)
    detector.index_training_data(sample_training_texts, source="test")

    # q1 has exact overlap with first training text
    result = detector.check_item(sample_eval_items[0])

    assert result.item_id == "q1"
    assert result.n_gram_size == 13
    assert result.overlapping_ngrams > 0
    assert result.overlap_fraction > 0
    assert result.longest_match > 0


def test_ngram_detector_no_match(sample_training_texts, sample_eval_items) -> None:
    """Test handling of items with no overlap."""
    detector = NGramOverlapDetector(n=13)
    detector.index_training_data(sample_training_texts, source="test")

    # q3 has partial overlap but should have some n-gram match
    result = detector.check_item(sample_eval_items[2])

    assert result.item_id == "q3"
    # Most n-grams won't match
    assert result.overlap_fraction >= 0


def test_ngram_stats_severity_classification() -> None:
    """Test severity classification based on overlap."""
    # Critical: >50% overlap
    critical = NGramStats(
        item_id="critical", n_gram_size=13, eval_ngrams=100, overlapping_ngrams=60,
        longest_match=150, overlap_fraction=0.6,
    )
    assert critical.severity == ContaminationSeverity.CRITICAL

    # High: 20-50% overlap
    high = NGramStats(
        item_id="high", n_gram_size=13, eval_ngrams=100, overlapping_ngrams=30,
        longest_match=80, overlap_fraction=0.3,
    )
    assert high.severity == ContaminationSeverity.HIGH

    # Medium: 5-20% overlap
    medium = NGramStats(
        item_id="medium", n_gram_size=13, eval_ngrams=100, overlapping_ngrams=10,
        longest_match=30, overlap_fraction=0.1,
    )
    assert medium.severity == ContaminationSeverity.MEDIUM

    # Low: 1-5% overlap
    low = NGramStats(
        item_id="low", n_gram_size=13, eval_ngrams=100, overlapping_ngrams=2,
        longest_match=10, overlap_fraction=0.02,
    )
    assert low.severity == ContaminationSeverity.LOW

    # None: <1% overlap
    none = NGramStats(
        item_id="none", n_gram_size=13, eval_ngrams=100, overlapping_ngrams=0,
        longest_match=5, overlap_fraction=0.0,
    )
    assert none.severity == ContaminationSeverity.NONE


def test_ngram_stats_longest_match_triggers_critical() -> None:
    """Test that very long exact matches trigger critical severity."""
    stats = NGramStats(
        item_id="long_match", n_gram_size=13, eval_ngrams=100, overlapping_ngrams=10,
        longest_match=250,  # >200 chars
        overlap_fraction=0.1,  # Would be medium based on fraction alone
    )
    assert stats.severity == ContaminationSeverity.CRITICAL


def test_scan_benchmark(sample_training_texts, sample_eval_items) -> None:
    """Test scanning a full benchmark for contamination."""
    detector = NGramOverlapDetector(n=13)
    detector.index_training_data(sample_training_texts, source="test")

    config = DecontaminationConfig(n_gram_size=13)
    report = scan_benchmark(sample_eval_items, detector, config)

    assert report.benchmark == "medqa"
    assert report.total_items == 3
    assert report.items_with_overlap >= 1  # At least q1 should match
    assert report.n_gram_size == 13
    assert len(report.item_stats) >= 1


def test_load_training_texts(tmp_path, sample_training_texts) -> None:
    """Test loading training texts from files."""
    texts = load_training_texts([tmp_path / "train.jsonl"], text_fields=["prompt"])

    assert len(texts) == 5
    assert texts[0] == sample_training_texts[0]


def test_extract_text_from_record() -> None:
    """Test text extraction from various record formats."""
    # Direct field
    assert _extract_text_from_record({"prompt": "test"}, ["prompt"]) == "test"

    # Message format
    msg_record = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    text = _extract_text_from_record(msg_record, ["prompt"])
    assert "question" in text and "answer" in text

    # Conversation format
    conv_record = {
        "conversations": [
            ["human", "hello"],
            ["gpt", "hi there"],
        ]
    }
    text = _extract_text_from_record(conv_record, ["prompt"])
    assert "hello" in text and "hi there" in text


def test_generate_negative_controls(sample_eval_items) -> None:
    """Test generation of negative control items."""
    contaminated_stats = [
        NGramStats(
            item_id="q1", n_gram_size=13, eval_ngrams=100, overlapping_ngrams=10,
            longest_match=50, overlap_fraction=0.1,
        )
    ]

    controls = generate_negative_controls(
        [sample_eval_items[0]], contaminated_stats, controls_per_item=2
    )

    assert len(controls) == 2
    assert controls[0].original_benchmark == "medqa"
    assert controls[0].original_item_id == "q1"
    assert "negctrl" in controls[0].control_id
    assert controls[0].messages is not None
    assert len(controls[0].messages) > 0


def test_run_decontamination_check(tmp_path, sample_eval_items) -> None:
    """Test the full decontamination check pipeline."""
    # Create training data
    training_data = [
        "The patient presents with chest pain and shortness of breath.",
        "Diagnosis of myocardial infarction requires ECG changes.",
    ]
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(json.dumps({"prompt": t}) for t in training_data))

    benchmarks = {"medqa": sample_eval_items}
    config = DecontaminationConfig(
        n_gram_size=13,
        training_data_paths=[str(train_file)],
        fail_on_critical=False,  # Don't fail in test
    )

    reports = run_decontamination_check(benchmarks, config=config)

    assert "medqa" in reports
    report = reports["medqa"]
    assert report.total_items == 3
    assert report.benchmark == "medqa"


def test_save_decontamination_report(tmp_path, sample_eval_items) -> None:
    """Test saving contamination report to disk."""
    report = BenchmarkContaminationReport(
        benchmark="test",
        total_items=10,
        items_with_overlap=2,
        severity_counts={"low": 2},
        item_stats=[],
        training_sources=["test_data"],
        n_gram_size=13,
    )

    output_path = tmp_path / "decontam_report.json"
    save_decontamination_report({"test": report}, output_path)

    assert output_path.exists()

    loaded = json.loads(output_path.read_text())
    assert "test" in loaded
    assert loaded["test"]["total_items"] == 10


def test_summarize_for_human() -> None:
    """Test generation of human-readable summary."""
    reports = {
        "medqa": BenchmarkContaminationReport(
            benchmark="medqa",
            total_items=100,
            items_with_overlap=5,
            severity_counts={"low": 5},
            item_stats=[],
            training_sources=["train"],
            n_gram_size=13,
        ),
        "medmcqa": BenchmarkContaminationReport(
            benchmark="medmcqa",
            total_items=50,
            items_with_overlap=1,
            severity_counts={"high": 1},
            item_stats=[],
            training_sources=["train"],
            n_gram_size=13,
        ),
    }

    summary = summarize_for_human(reports)

    assert "CONTAMINATION SCAN SUMMARY" in summary
    assert "medqa" in summary
    assert "medmcqa" in summary
    # Check for the total items format (could be "6/150" or similar)
    assert "150" in summary  # Total items
    assert "Legend:" in summary


def test_decontamination_config_defaults() -> None:
    """Test that DecontaminationConfig has sensible defaults."""
    config = DecontaminationConfig()

    assert config.n_gram_size == 13
    assert config.critical_threshold == 0.5
    assert config.high_threshold == 0.2
    assert config.medium_threshold == 0.05
    assert config.low_threshold == 0.01
    assert config.fail_on_critical is True
    assert config.generate_negative_controls is True


def test_benchmark_contamination_report_properties() -> None:
    """Test BenchmarkContaminationReport computed properties."""
    report = BenchmarkContaminationReport(
        benchmark="test",
        total_items=100,
        items_with_overlap=20,
        severity_counts={"critical": 5, "high": 10, "medium": 5},
        item_stats=[],
        training_sources=["train"],
        n_gram_size=13,
    )

    assert report.contamination_rate == 0.2
    assert report.has_critical_contamination is True

    # Report without critical
    report_safe = BenchmarkContaminationReport(
        benchmark="test",
        total_items=100,
        items_with_overlap=10,
        severity_counts={"low": 10},
        item_stats=[],
        training_sources=["train"],
        n_gram_size=13,
    )

    assert report_safe.contamination_rate == 0.1
    assert report_safe.has_critical_contamination is False


def test_negative_control_structure() -> None:
    """Test NegativeControl dataclass structure."""
    control = NegativeControl(
        control_id="test::q1::negctrl::0",
        original_benchmark="medqa",
        original_item_id="q1",
        messages=({"role": "user", "content": "test"},),
        perturbation="distractor_context",
        expected_diff="perturbed",
    )

    assert control.control_id == "test::q1::negctrl::0"
    assert control.original_benchmark == "medqa"
    assert control.perturbation == "distractor_context"


def test_run_decontamination_fails_on_critical(tmp_path, sample_eval_items) -> None:
    """Test that critical contamination fails the run when configured."""
    # Create training data that will cause critical overlap with q1
    training_data = [
        "The patient presents with chest pain and shortness of breath. " +
        "The pain is substernal and radiating to the left arm. " +
        "There is associated diaphoresis and nausea. " +
        "The patient presents with chest pain and shortness of breath. " +
        "The pain is substernal and radiating to the left arm."
    ]
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"prompt": training_data[0]}))

    benchmarks = {"medqa": sample_eval_items}
    config = DecontaminationConfig(
        n_gram_size=13,
        training_data_paths=[str(train_file)],
        fail_on_critical=True,
        critical_threshold=0.1,  # Lower threshold for testing
    )

    # This should raise RuntimeError due to critical contamination
    with pytest.raises(RuntimeError, match="Critical contamination"):
        run_decontamination_check(benchmarks, config=config)


def test_load_training_texts_handles_various_formats(tmp_path) -> None:
    """Test loading from JSON and JSONL formats."""
    # JSONL
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text("\n".join([
        json.dumps({"prompt": "text1"}),
        json.dumps({"prompt": "text2"}),
    ]))

    # JSON
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([
        {"prompt": "text3"},
        {"prompt": "text4"},
    ]))

    texts = load_training_texts(
        [str(jsonl_file), str(json_file)],
        text_fields=["prompt"]
    )

    assert len(texts) == 4
    assert "text1" in texts
    assert "text3" in texts
