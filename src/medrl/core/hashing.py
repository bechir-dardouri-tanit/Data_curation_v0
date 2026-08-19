"""Content hashing.

Every stage output is addressed by the hash of its inputs, so re-running an identical
configuration is a cache hit and an ablation sweep never silently re-computes an arm it
already has. Hashes must be stable across processes and machines, which rules out
``hash()`` and any dict-ordering dependence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20
DIGEST_LEN = 16


def _canonical(obj: Any) -> Any:
    """Convert to a form with a deterministic JSON encoding.

    Mappings are key-sorted; sets become sorted lists; paths and everything exotic fall
    back to ``str``. Floats are tagged and keep full repr precision, so 0.1 never
    collides with 0.10000001 -- and a float never collides with the *string* spelling
    of itself, which plain ``repr`` would let happen (``1.5`` and ``"1.5"`` hashed
    identically before the tag). Lists and tuples intentionally share one form: for
    content addressing, same content is same address.
    """
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonical(v) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return {"$float": repr(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):  # pydantic
        return _canonical(obj.model_dump(mode="json"))
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON, suitable for hashing."""
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_obj(obj: Any, *, length: int = DIGEST_LEN) -> str:
    """Stable short digest of any JSON-able / pydantic object."""
    return hashlib.blake2b(canonical_json(obj).encode("utf-8"), digest_size=32).hexdigest()[:length]


def hash_file(path: str | Path, *, length: int = DIGEST_LEN) -> str:
    """Digest of a file's bytes, streamed so multi-GB safetensors don't blow up memory."""
    h = hashlib.blake2b(digest_size=32)
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()[:length]


def hash_files(paths: Iterable[str | Path], *, length: int = DIGEST_LEN) -> str:
    """Digest of a set of files, independent of iteration order.

    Hashes ``(path-as-given, content-digest)`` pairs rather than concatenated bytes, so
    any rename or move between directories is detected and file ordering is irrelevant.
    Keying by the *full* given path matters: two ``weights.json`` in different
    directories are different inputs, and keying by basename made their contents
    interchangeable without a hash change. Callers wanting a root-relative view
    (relocation-invariant) should use :func:`hash_dir`.
    """
    entries = sorted((str(p), hash_file(p)) for p in paths)
    return hash_obj(entries, length=length)


def hash_dir(
    root: str | Path,
    *,
    patterns: Sequence[str] = ("**/*",),
    length: int = DIGEST_LEN,
) -> str:
    """Digest of a directory tree, keyed by path-relative names."""
    root_path = Path(root)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(p for p in root_path.glob(pattern) if p.is_file())
    entries = sorted(
        (str(p.relative_to(root_path)), hash_file(p)) for p in dict.fromkeys(files)
    )
    return hash_obj(entries, length=length)


def hash_text(text: str, *, length: int = DIGEST_LEN) -> str:
    """Digest of a string. Used for prompt-template hashes."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()[:length]
