"""S2 structural-flag tests: per-check synthetic items, the flags-not-deletes
stream through the real parquet store, flag_rates math, and a seeded property
batch of corrupted strings."""

from __future__ import annotations

import random
from typing import Any

import pytest

from medrl.curation import store
from medrl.curation.schema import CorpusItem, Flags, StageError
from medrl.curation.stages import structural
from medrl.curation.thresholds import THRESHOLDS, snapshot

_CLEAN_USER = "What is the recommended first-line treatment for anaphylaxis in adults?"
_CLEAN_ASSISTANT = (
    "Intramuscular adrenaline into the anterolateral thigh is the recommended "
    "first-line treatment, repeated after five minutes if there is no improvement."
)


def _item(item_id: str = "t:1", **overrides: Any) -> CorpusItem:
    """A default-clean row; overrides replace fields wholesale."""
    overrides.setdefault("source", "unit_src")
    overrides.setdefault(
        "messages",
        [
            {"role": "user", "content": _CLEAN_USER},
            {"role": "assistant", "content": _CLEAN_ASSISTANT},
        ],
    )
    return CorpusItem(id=item_id, **overrides)


def _flagged_by(flag: str, item: CorpusItem) -> bool:
    """Run one registry check directly -- the registry wiring under test."""
    return structural.CHECKS[flag](item)


def _run_stage(tmp_path: Any, items: list[CorpusItem]) -> Any:
    inp = tmp_path / "01_normalize"
    out = tmp_path / "02_structural"
    inp.mkdir(parents=True)  # write_items assumes the snapshot dir exists
    store.write_items(items, inp)
    return structural.stage_entry("test-run", input_dir=inp, output_dir=out), inp, out


# --------------------------------------------------------------------------
# Registry shape.
# --------------------------------------------------------------------------


def test_registry_keys_are_flags_fields():
    s2_flags = {
        f"f_{n}"
        for n in ("empty", "length", "truncated", "repetition", "lang", "encoding", "refusal")
    }
    assert set(structural.CHECKS) == s2_flags
    assert set(structural.CHECKS) <= set(Flags.model_fields)


def test_unknown_check_key_fails_loudly(tmp_path, monkeypatch):
    def bogus(item: CorpusItem) -> bool:
        return False

    monkeypatch.setattr(structural, "CHECKS", {"f_not_in_schema": bogus})
    inp = tmp_path / "01_normalize"
    inp.mkdir()
    store.write_items([_item()], inp)
    with pytest.raises(StageError, match="f_not_in_schema"):
        structural.stage_entry("x", input_dir=inp, output_dir=tmp_path / "02_structural")


def test_missing_input_snapshot_fails_loudly(tmp_path):
    with pytest.raises(StageError, match="01_normalize"):
        structural.stage_entry(
            "x", input_dir=tmp_path / "01_normalize", output_dir=tmp_path / "02_structural"
        )


# --------------------------------------------------------------------------
# One synthetic item per flag, plus the clean row.
# --------------------------------------------------------------------------


def test_clean_item_sets_no_flags():
    item = _item()
    for name in structural.CHECKS:
        assert _flagged_by(name, item) is False, name
    assert item.flags.any() is False


def test_f_empty_whitespace_only_or_missing_user_content():
    blank = _item(
        messages=[
            {"role": "user", "content": " \t\n "},
            {"role": "assistant", "content": _CLEAN_ASSISTANT},
        ]
    )
    assert _flagged_by("f_empty", blank) is True
    assert _flagged_by("f_empty", _item(messages=[])) is True  # no user turn: no question
    assert _flagged_by("f_empty", _item()) is False


def test_f_length_below_min_and_above_max(monkeypatch):
    tiny = _item(
        "t:2",
        messages=[{"role": "user", "content": "Short?"}, {"role": "assistant", "content": "Yes."}],
    )
    # Real thresholds: 3-4 tokens is under min_tokens=8 in either counting mode.
    assert _flagged_by("f_length", tiny) is True
    assert _flagged_by("f_length", _item()) is False

    # Bounds come from THRESHOLDS, not literals: widen the window past the tiny
    # row and the clean row flips to flagged while the tiny row still passes.
    # The window [1, 8] contains the tiny row's count in BOTH counting modes
    # (6 Qwen tokens vs 2 whitespace words) and excludes the clean row's
    # (45 vs 27), so the pair of assertions holds whichever mode is live.
    monkeypatch.setattr(
        structural,
        "THRESHOLDS",
        THRESHOLDS.model_copy(update={"min_tokens": 1, "max_tokens": 8}),
    )
    assert _flagged_by("f_length", _item()) is True
    assert _flagged_by("f_length", tiny) is False


def test_f_truncated_open_think_without_close():
    unclosed = _item(
        "t:2",
        messages=[
            {"role": "user", "content": _CLEAN_USER},
            {
                "role": "assistant",
                "content": "<think>\nAssess the airway first, then adrenaline dosing...",
            },
        ],
    )
    assert _flagged_by("f_truncated", unclosed) is True
    unclosed_thinking = _item("t:3", thinking="<think>\nlong deliberation with no closure")
    assert _flagged_by("f_truncated", unclosed_thinking) is True


def test_f_truncated_closed_pair_or_close_only_not_flagged():
    closed = _item(
        "t:2",
        messages=[
            {"role": "user", "content": _CLEAN_USER},
            {
                "role": "assistant",
                "content": "<think>\nAirway first.</think> Give intramuscular adrenaline.",
            },
        ],
    )
    assert _flagged_by("f_truncated", closed) is False
    # Close without open is complete by the eval convention: serving prefills <think>.
    close_only = _item(
        "t:3",
        messages=[
            {"role": "user", "content": _CLEAN_USER},
            {"role": "assistant", "content": "</think> Give intramuscular adrenaline."},
        ],
    )
    assert _flagged_by("f_truncated", close_only) is False
    assert _flagged_by("f_truncated", _item(thinking="<think>\ndone.</think>")) is False


def test_f_repetition_strictly_above_max_occurrences():
    sentence = (
        "The recommended dose of intramuscular adrenaline for an adult is five hundred micrograms."
    )
    assert len(sentence) >= THRESHOLDS.repetition_span_chars

    def row(copies: int) -> CorpusItem:
        return _item(
            "t:r",
            messages=[
                {"role": "user", "content": _CLEAN_USER},
                {"role": "assistant", "content": " ".join([sentence] * copies)},
            ],
        )

    assert _flagged_by("f_repetition", row(THRESHOLDS.repetition_max_occurrences)) is False
    assert _flagged_by("f_repetition", row(THRESHOLDS.repetition_max_occurrences + 1)) is True


def test_f_lang_outside_en_fr():
    assert _flagged_by("f_lang", _item("t:2", lang="de")) is True
    assert _flagged_by("f_lang", _item("t:3", lang="fr")) is False
    assert _flagged_by("f_lang", _item("t:4", lang="en")) is False


def test_f_encoding_mojibake_and_replacement_chars():
    damaged = _item(
        "t:2",
        messages=[
            {"role": "user", "content": _CLEAN_USER},
            {
                "role": "assistant",
                "content": "The summary follows. " + structural.MOJIBAKE_SEQUENCES[0] * 8,
            },
        ],
    )
    assert _flagged_by("f_encoding", damaged) is True

    borderline = _item(
        "t:3",
        messages=[
            {"role": "user", "content": "x" * 998 + "��"},
            {"role": "assistant", "content": _CLEAN_ASSISTANT},
        ],
    )
    # exactly the sort of near-miss the ratio exists to place: 2/1000 > 0.001.
    assert _flagged_by("f_encoding", borderline) is True


def test_f_encoding_boundary_is_strictly_above_threshold():
    at_threshold = _item("t:2", messages=[{"role": "user", "content": "x" * 999 + "�"}])
    # 1 hit / 1000 chars == encoding_damage_max_ratio: strictly-above, so clean.
    assert _flagged_by("f_encoding", at_threshold) is False
    just_under = _item("t:3", messages=[{"role": "user", "content": "x" * 1200 + "�"}])
    assert _flagged_by("f_encoding", just_under) is False


def test_f_encoding_legitimate_french_accents_not_flagged():
    french = _item(
        "t:2",
        lang="fr",
        messages=[
            {"role": "user", "content": "Quels sont les effets secondaires de ce traitement ?"},
            {
                "role": "assistant",
                "content": "L'infirmière de l'hôpital a donné le café au patient âgé — il va mieux, "
                "cœur calme, à l'înserment près.",
            },
        ],
    )
    assert _flagged_by("f_encoding", french) is False


def test_f_refusal_matches_templates_case_insensitively():
    positives = [
        "I'm sorry, but I cannot disclose that information.",
        "AS AN AI LANGUAGE MODEL, I cannot provide medical advice.",
        "I am not able to assist with that request.",
        "I must decline to answer this question.",
        "That would be against my guidelines.",
    ]
    for i, text in enumerate(positives):
        item = _item(
            f"t:{i}",
            messages=[
                {"role": "user", "content": _CLEAN_USER},
                {"role": "assistant", "content": text},
            ],
        )
        assert _flagged_by("f_refusal", item) is True, text

    negatives = [
        "I recommend seeing a doctor promptly.",
        "The patient apologised and continued.",
        _CLEAN_ASSISTANT,
    ]
    for i, text in enumerate(negatives):
        item = _item(
            f"t:n{i}",
            messages=[
                {"role": "user", "content": _CLEAN_USER},
                {"role": "assistant", "content": text},
            ],
        )
        assert _flagged_by("f_refusal", item) is False, text


def test_f_refusal_ignores_thinking_surface():
    # Assistant content is the refusal surface; a hedge inside the trace is not.
    item = _item("t:2", thinking="Should I decline? No -- as an AI I should just answer carefully.")
    assert _flagged_by("f_refusal", item) is False


# --------------------------------------------------------------------------
# Property-style: 20 seeded corrupted strings, each caught by the right check.
# --------------------------------------------------------------------------


def test_property_twenty_random_corruptions_flagged():
    rng = random.Random(20260926)
    words = [
        "patient",
        "renal",
        "dose",
        "fever",
        "trial",
        "vein",
        "cardiac",
        "heparin",
        "scan",
        "ward",
    ]
    for i in range(20):
        if i % 2 == 0:
            sentence = " ".join(rng.choice(words) for _ in range(12))
            assert len(sentence) >= THRESHOLDS.repetition_span_chars
            text = " ".join([sentence] * (THRESHOLDS.repetition_max_occurrences + 2))
            item = _item(
                f"p:{i}",
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": text},
                ],
            )
            assert _flagged_by("f_repetition", item) is True, (i, text[:80])
            assert _flagged_by("f_encoding", item) is False
        else:
            base = " ".join(rng.choice(words) for _ in range(30))
            clean = _item(
                f"p:{i}",
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": base},
                ],
            )
            assert _flagged_by("f_encoding", clean) is False
            corrupted = base
            for _ in range(rng.randint(2, 6)):
                pos = rng.randrange(len(corrupted))
                corrupted = (
                    corrupted[:pos] + rng.choice(structural.MOJIBAKE_SEQUENCES) + corrupted[pos:]
                )
            bad = _item(
                f"p:{i}",
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": corrupted},
                ],
            )
            assert _flagged_by("f_encoding", bad) is True, (i, corrupted[:80])
            assert _flagged_by("f_repetition", bad) is False


# --------------------------------------------------------------------------
# Tokenizer contract.
# --------------------------------------------------------------------------


def test_count_tokens_and_declared_mode():
    assert structural.tokenizer_mode() in {structural.TOKENIZER_ID, "whitespace_approx"}
    assert structural.count_tokens("asthma exacerbation treatment guidelines") >= 4
    assert structural.count_tokens("") == 0


# --------------------------------------------------------------------------
# Stage entry over the real store: invariant, roundtrip, flag_rates math.
# --------------------------------------------------------------------------


def test_stage_never_drops_and_hashes_match(tmp_path):
    items = [
        _item("a:1", source="src_a"),
        _item("a:2", source="src_a", lang="de"),
        _item("b:1", source="src_b"),
    ]
    manifest, inp, out = _run_stage(tmp_path, items)

    assert manifest.rows_in == manifest.rows_out == 3
    assert store.count_items(out) == 3
    assert manifest.stage == "02_structural"
    assert manifest.input_sha256 == store.content_sha256(inp)
    assert manifest.output_sha256 == store.content_sha256(out)
    assert manifest.thresholds == snapshot()
    assert set(manifest.flag_rates) == set(structural.CHECKS)
    assert manifest.config["checks"] == sorted(structural.CHECKS)
    assert manifest.config["tokenizer_mode"] in {structural.TOKENIZER_ID, "whitespace_approx"}


def test_flags_survive_store_roundtrip_and_preserve_upstream(tmp_path):
    items = [
        _item("r:1", lang="de"),
        _item("r:2"),
        _item(
            "r:3",
            messages=[
                {"role": "user", "content": "  "},
                {"role": "assistant", "content": _CLEAN_ASSISTANT},
            ],
        ),
        _item("r:4", lang="de", flags=Flags(f_dup_exact=True)),  # upstream flag arrives set
    ]
    _run_stage(tmp_path, items)
    out = tmp_path / "02_structural"
    back = {it.id: it for it in store.iter_items(out)}

    assert back["r:1"].flags.f_lang is True
    assert back["r:1"].flags.f_empty is False
    assert back["r:2"].flags.any() is False
    assert back["r:3"].flags.f_empty is True
    assert back["r:4"].flags.f_dup_exact is True  # never cleared
    assert back["r:4"].flags.f_lang is True  # and S2 adds its own on top


def test_flag_rates_math_on_mixed_batch(tmp_path):
    items = [
        _item("a:1", source="src_a"),  # clean
        _item(
            "a:2",
            source="src_a",
            messages=[
                {"role": "user", "content": "   "},
                {"role": "assistant", "content": _CLEAN_ASSISTANT},
            ],
        ),  # f_empty
        _item("a:3", source="src_a", lang="de"),  # f_lang
        _item(
            "b:1",
            source="src_b",
            messages=[
                {"role": "user", "content": "How should I store insulin at home?"},
                {
                    "role": "assistant",
                    "content": "I'm sorry, but I cannot provide storage guidance for "
                    "prescription devices without more context.",
                },
            ],
        ),  # f_refusal
    ]
    manifest, _inp, _out = _run_stage(tmp_path, items)

    assert manifest.flag_rates["f_empty"] == {
        "src_a": pytest.approx(1 / 3),
        "src_b": 0.0,
        "_all": pytest.approx(0.25),
    }
    assert manifest.flag_rates["f_lang"] == {
        "src_a": pytest.approx(1 / 3),
        "src_b": 0.0,
        "_all": pytest.approx(0.25),
    }
    assert manifest.flag_rates["f_refusal"] == {
        "src_a": 0.0,
        "src_b": 1.0,
        "_all": pytest.approx(0.25),
    }
    for unset in ("f_length", "f_truncated", "f_repetition", "f_encoding"):
        assert manifest.flag_rates[unset] == {"src_a": 0.0, "src_b": 0.0, "_all": 0.0}
    assert manifest.notes["rows_flagged_any"] == 3


def test_rerun_replaces_own_snapshot_instead_of_appending(tmp_path: Any) -> None:
    """Regression: stage_entry used to mkdir without clearing stale parts, and
    write_items APPENDS with continuing part numbers -- a re-run (the documented
    correction flow) doubled every row on disk, then crashed on rows_in/out."""
    items = [_item(f"t:{i}") for i in range(5)]
    inp = tmp_path / "01_normalize"
    inp.mkdir(parents=True)
    store.write_items(items, inp)
    out = tmp_path / "02_structural"

    m1 = structural.stage_entry("test-run", input_dir=inp, output_dir=out)
    m2 = structural.stage_entry("test-run", input_dir=inp, output_dir=out)

    assert m1.rows_out == m2.rows_out == 5, "stale parts must go before the re-write"
    assert len(list(out.glob("part-*.parquet"))) == 1
    assert m2.output_sha256 == m1.output_sha256
