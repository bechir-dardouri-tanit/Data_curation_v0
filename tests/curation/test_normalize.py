"""S1 normalize tests: honest drop accounting, LID robustness, GlotLID observability.

No network, no fasttext model, no /scratch: iter_source_files, detect_lang's
model handle and the store paths are all injected/monkeypatched per test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from medrl.curation.schema import CorpusItem, SourceRecord
from medrl.curation.stages import normalize


def _record() -> SourceRecord:
    return SourceRecord(
        source_id="generalthought_biology",
        url_or_hf_id="org/ds",
        content_sha256="0" * 64,
        n_rows=4,
        downloaded_at=datetime.now(UTC),
    )


def _jsonl(tmp_path: Any, rows: list[dict[str, Any]]) -> Any:
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


@pytest.fixture
def no_store_scratch(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    """Point the stage's store paths at tmp_path (run_normalize writes snapshots)."""
    monkeypatch.setattr(normalize, "stage_dir", lambda run_id, name: tmp_path / name)
    yield tmp_path


def test_drop_accounting_counts_every_row(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, no_store_scratch: Any
) -> None:
    """Regression: rows_seen/items_kept were both incremented over YIELDED items
    only, so dropped_by_mapper was structurally {source: 0} and the manifest's
    drop table was always zero -- the drops it named (mapper-None filters) were
    exactly the ones not counted."""
    rows = [
        {"field": "biology", "question": "kept row"},  # kept
        {"field": "chemistry", "question": "filtered row"},  # mapper returns None
        {"question": "boom"},  # mapper raises
        {"field": "biology", "question": "lid-poisoned"},  # detect_lang raises
    ]
    path = _jsonl(tmp_path, rows)
    monkeypatch.setattr(normalize, "iter_source_files", lambda record: iter([path]))
    monkeypatch.setattr("medrl.curation.store.read_registry", lambda run_id: [_record()])

    # force a genuine mapper exception (the real mapper None-filters instead)
    real_mapper = normalize.MAPPERS["generalthought_biology"]

    def flaky_mapper(sid: str, row: dict[str, Any]) -> CorpusItem | None:
        if row.get("question") == "boom":
            raise ValueError("malformed row")
        return real_mapper(sid, row)

    monkeypatch.setitem(normalize.MAPPERS, "generalthought_biology", flaky_mapper)

    real_detect = normalize.detect_lang

    def fake_detect(text: str) -> tuple[str, float]:
        if "lid-poisoned" in text:
            raise TypeError("predict(): incompatible function arguments")
        return real_detect(text)

    monkeypatch.setattr(normalize, "detect_lang", fake_detect)

    counts: dict[str, int] = {"rows_raw": 0, "mapper_none": 0, "errored": 0, "kept": 0}
    items = list(
        normalize.materialize_source(
            "generalthought_biology", {"generalthought_biology": _record()}, counts
        )
    )

    assert counts == {"rows_raw": 4, "mapper_none": 1, "errored": 2, "kept": 1}
    assert [it.id for it in items] == ["generalthought_biology:0:0"]


def test_stage_entry_publishes_real_drop_table(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, no_store_scratch: Any
) -> None:
    rows = [
        {"field": "biology", "question": "kept row"},
        {"field": "chemistry", "question": "filtered row"},
    ]
    path = _jsonl(tmp_path, rows)
    monkeypatch.setattr(normalize, "iter_source_files", lambda record: iter([path]))
    monkeypatch.setattr("medrl.curation.store.read_registry", lambda run_id: [_record()])
    monkeypatch.setattr(normalize, "detect_lang", lambda text: ("en", 0.99))

    manifest = normalize.stage_entry("run-x", sources=["generalthought_biology"])

    assert manifest.rows_in == 2  # raw rows read, not just the kept ones
    assert manifest.rows_out == 1
    assert manifest.notes["dropped_by_mapper"] == {"generalthought_biology": 1}
    assert manifest.notes["errored_rows"] == {"generalthought_biology": 0}
    assert manifest.notes["rows_out_matches_kept"] is True
    # second-opinion availability is observable, not silently absent
    assert manifest.notes["glotlid_second_opinion"] is False


def test_detect_lang_scrubs_lone_surrogates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: fasttext's pybind cast raises TypeError on a lone surrogate
    (valid inside a .jsonl row), aborting the whole pass from a single row."""
    seen: dict[str, str] = {}

    class _FakeLid:
        class f:  # noqa: N801 - mirrors fasttext's .f accessor
            @staticmethod
            def predict(text: str, k: int, threshold: float, on_unicode_error: str):
                seen["text"] = text
                return [(0.99, "__label__en")]

    monkeypatch.setattr(normalize, "_lid", lambda: _FakeLid())
    monkeypatch.setattr(normalize, "_glotlid_probed", True)
    monkeypatch.setattr(normalize, "_glotlid_model", None)

    lang, score = normalize.detect_lang("treatment plan x\ud800y")

    assert (lang, score) == ("en", 0.99)
    assert "\ud800" not in seen["text"], "the surrogate must be scrubbed before predict"


def test_glotlid_absence_is_observable_not_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression: 'from gltPID import GlotLID' always raises, silently swallowed
    per row -- the second opinion could never run and nothing said so."""

    def _boom():
        raise ModuleNotFoundError("No module named 'gltPID'")

    monkeypatch.setattr(normalize, "_glotlid", _boom)
    monkeypatch.setattr(normalize, "_glotlid_probed", False)
    monkeypatch.setattr(normalize, "_glotlid_model", None)

    assert normalize.glotlid_available() is False
    assert any("gltPID" in r.message or "GlotLID" in r.message for r in caplog.records)


def test_detect_lang_survives_low_confidence_without_second_opinion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLid:
        class f:  # noqa: N801
            @staticmethod
            def predict(text: str, k: int, threshold: float, on_unicode_error: str):
                return [(0.50, "__label__en")]

    monkeypatch.setattr(normalize, "_lid", lambda: _FakeLid())
    monkeypatch.setattr(normalize, "_glotlid_probed", True)
    monkeypatch.setattr(normalize, "_glotlid_model", None)  # unavailable

    # below the floor normally routes to GlotLID; absent, fastText verdict stands
    assert normalize.detect_lang("kvacks?") == ("en", 0.50)
