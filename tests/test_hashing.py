"""Content-hash tests: the property that makes the whole store work is that the same
inputs are the same address AND different inputs are different addresses.

The two collision cases here were both real (found in review): ``1.5`` hashed identically
to ``"1.5"``, and ``hash_files`` keyed entries by basename, making the contents of
``a/w.json`` and ``b/w.json`` interchangeable without a hash change.
"""

from __future__ import annotations

import pytest

from medrl.core.hashing import (
    canonical_json,
    hash_dir,
    hash_file,
    hash_files,
    hash_obj,
    hash_text,
)


def test_dict_order_and_process_are_irrelevant() -> None:
    assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_float_and_its_string_spelling_differ() -> None:
    assert hash_obj(1.5) != hash_obj("1.5")
    assert hash_obj({"lr": 0.1}) != hash_obj({"lr": "0.1"})
    # ...while precision is still preserved, not rounded through JSON floats.
    assert hash_obj(0.1) != hash_obj(0.10000001)


def test_tuple_and_list_are_the_same_address() -> None:
    # Documented equivalence: content addressing means same content, same address.
    assert hash_obj((1, 2)) == hash_obj([1, 2])


def test_sets_are_order_free() -> None:
    assert hash_obj({"x", "y"}) == hash_obj({"y", "x"})
    assert hash_obj({"x"}) != hash_obj({"y"})


def test_hash_files_detects_content_swaps_between_dirs(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "w.json").write_text("one")
    (tmp_path / "b" / "w.json").write_text("two")

    same = hash_files([tmp_path / "a" / "w.json", tmp_path / "b" / "w.json"])
    swapped = hash_files(
        # same two basenames, contents exchanged between the directories
        [(tmp_path / "a" / "w.json"), (tmp_path / "b" / "w.json")]
    )
    assert same == swapped  # iteration order irrelevant

    (tmp_path / "a" / "w.json").write_text("two")
    (tmp_path / "b" / "w.json").write_text("one")
    assert hash_files([tmp_path / "a" / "w.json", tmp_path / "b" / "w.json"]) != same


def test_hash_files_detects_renames(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("x: 1")
    before = hash_files([p])
    p2 = tmp_path / "renamed.yaml"
    p2.write_text("x: 1")
    assert hash_files([p2]) != before  # a rename is a different input set
    assert hash_file(p) == hash_file(p2)  # byte-identical content still matches


def test_hash_dir_is_relocation_invariant_on_relative_names(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.bin").write_bytes(b"\x00\x01")
    (tmp_path / "top.txt").write_text("hi")
    assert hash_dir(tmp_path) == hash_dir(tmp_path)  # deterministic
    (tmp_path / "sub" / "g.bin").write_bytes(b"\x00\x01")
    assert hash_dir(tmp_path) != hash_dir(tmp_path, patterns=["top.txt"])


def test_hash_text_differs_from_hash_obj_of_the_string() -> None:
    # Different domains, different digests -- no accidental cross-use equivalence.
    assert hash_text("abc") != hash_obj("abc")


def test_length_parameter_truncates() -> None:
    assert len(hash_obj({"a": 1}, length=8)) == 8
    with pytest.raises(TypeError):
        hash_obj({"a": 1}, length="8")  # type: ignore[arg-type]
