"""S0 registry tests: the provenance ledger's pinning + hashing contracts.

acquire()/iter_source_files() used to float on latest main (so the recorded
hf_revision could cite bytes nobody ever read, and a re-run months later saw
different data), and the content hash skipped every file >= 1 GiB (hashed as
the literal b"<large>"), so a corrupted re-download of a multi-GB shard
verified clean against the registry.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from medrl.curation import registry
from medrl.curation.schema import SourceRecord

PINNED_SHA = "abc123"
ALLOW_PATTERNS = ["*.jsonl", "*.json", "*.parquet"]


def _hub_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry,
        "fetch_hub_metadata",
        lambda hf_id: {"resolved_id": hf_id, "sha": PINNED_SHA, "licence_tag": "mit", "gated": False},
    )


def _install_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, Any]) -> list[dict]:
    """Replace huggingface_hub.snapshot_download: record call kwargs, materialize
    the named files (an int value makes a sparse file of that many bytes) plus a
    non-data README the ledger must ignore."""
    calls: list[dict] = []

    def fake_download(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        d = tmp_path / f"snap{len(calls)}"
        d.mkdir(parents=True)
        for name, blob in files.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                if isinstance(blob, int):
                    f.truncate(blob)  # sparse: st_size without writing the bytes
                else:
                    f.write(blob)
        (d / "README.md").write_bytes(b"not data")
        return str(d)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    return calls


def test_acquire_pins_revision_and_hashes_exact_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blob = b'{"q": "what"}\n' * 7
    calls = _install_snapshot(monkeypatch, tmp_path, {"data/train.jsonl": blob})
    _hub_meta(monkeypatch)

    plan = registry.SourcePlan(source_id="s", hf_id="org/ds", expected_rows=1)
    rec = registry.acquire(plan, tmp_path / "out")

    # S0 must pin the revision it records -- floating on main would let a
    # force-push between S0 and S1 make hf_revision/content_sha256 describe
    # bytes nobody read.
    assert calls[0]["revision"] == PINNED_SHA
    assert calls[0]["allow_patterns"] == ALLOW_PATTERNS
    assert rec.hf_revision == PINNED_SHA
    expected = hashlib.sha256()
    expected.update(b"train.jsonl")
    expected.update(blob)
    assert rec.content_sha256 == expected.hexdigest()


def test_acquire_hashes_content_at_the_old_gib_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hash once short-circuited files >= 1 GiB to b"<large>": the largest
    shards contributed only their filename, so truncated/re-encoded re-downloads
    verified clean. A file AT the old cutoff must contribute its actual bytes."""
    calls = _install_snapshot(monkeypatch, tmp_path, {"shard.jsonl": 1 << 30})
    _hub_meta(monkeypatch)

    plan = registry.SourcePlan(source_id="s", hf_id="org/ds", expected_rows=1)
    rec = registry.acquire(plan, tmp_path / "out")

    assert calls[0]["revision"] == PINNED_SHA
    expected = hashlib.sha256()
    expected.update(b"shard.jsonl")
    zero_mb = bytes(1 << 20)
    for _ in range(1 << 10):  # 1 GiB of NUL bytes, the sparse file's content
        expected.update(zero_mb)
    assert rec.content_sha256 == expected.hexdigest()


def test_iter_source_files_pins_revision_and_file_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1 must read the same bytes (revision) and the same file set
    (allow_patterns) S0 hashed: a repo shipping both .json and .jsonl exports
    would otherwise be read twice, duplicating every row."""
    calls = _install_snapshot(
        monkeypatch, tmp_path, {"a.jsonl": b"x\n", "nested/b.parquet": b"PARI"}
    )
    rec = SourceRecord(
        source_id="s",
        url_or_hf_id="resolved/org-ds",
        hf_revision=PINNED_SHA,
        content_sha256="0" * 64,
        n_rows=1,
        downloaded_at=datetime.now(UTC),
    )

    files = list(registry.iter_source_files(rec))

    assert calls[0]["repo_id"] == "resolved/org-ds"
    assert calls[0]["revision"] == PINNED_SHA
    assert calls[0]["allow_patterns"] == ALLOW_PATTERNS
    assert [p.name for p in files] == ["a.jsonl", "b.parquet"]
