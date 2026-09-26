"""S2 -- structural quality flags: the cheap, deterministic defect pass.

Every defect detectable by counting and regex -- emptiness, token-budget
violations, unterminated think traces, degenerate repetition, wrong language,
encoding damage, refusals -- is flagged here, before the expensive stages
(embeddings, LLM scoring, pass@k) spend GPU time on rows a regex could have
condemned. Nothing is dropped: flags are the delete, and mixtures stay
executable SQL over ``flags_*`` columns.

Checks are a registry (:data:`CHECKS`) of pure ``CorpusItem -> bool``
predicates so each is individually testable and the stage stream is a single
application loop. The registry keys are validated against
:class:`~medrl.curation.schema.Flags` at stage start -- a schema drift fails
loudly instead of silently writing flags the parquet projection cannot carry.

Token counting uses the cached Qwen3.5 tokenizer (the same family the cluster
serves and trains, whose 40960 ``max_model_len`` motivates ``max_tokens``
headroom), loaded lazily with ``local_files_only`` so the stage stays
deterministic and offline. Boxes without the cache fall back to a whitespace
approximation -- lenient at both bounds, because whitespace words undercount
subword tokens for English medical text (~1.3 tokens/word).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medrl.core.logging import get_logger
from medrl.curation import store
from medrl.curation.schema import CorpusItem, Flags, StageError, StageManifest, utcnow
from medrl.curation.thresholds import THRESHOLDS, snapshot

log = get_logger(__name__)

INPUT_STAGE = "01_normalize"
OUTPUT_STAGE = "02_structural"

ALLOWED_LANGS: frozenset[str] = frozenset({"en", "fr"})
"""S1 LID writes lang verbatim; anything outside the program's two working
languages is flagged here and stays in the table for the per-source reports."""

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
"""The eval convention (medrl/eval/generate.py ``_THINK_PREFILL``): the serving
stack prefills ``<think>`` and grades a response that never closes it as
truncated. S2 mirrors exactly that predicate on stored content."""

REFUSAL_PATTERNS: tuple[str, ...] = (
    r"i cannot provide",
    r"i (?:cannot|can'?t|am unable to) (?:help|assist|answer)",
    r"i'?m sorry,? but",
    r"as an ai(?: language model| assistant|,)?\b",
    r"i am not able to",
    r"i'?m not able to",
    r"i must (?:decline|refuse)",
    r"against my (?:programming|guidelines|ethical guidelines)",
)
"""Refusal templates as one constant, matched case-insensitively over assistant
content. Deliberately phrase-level: a refusal flag is evidence for review and
mixture exclusion, not a delete, so precision is allowed to beat recall here."""

_REFUSAL_RE: re.Pattern[str] = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

REPLACEMENT_CHAR = "�"

MOJIBAKE_SEQUENCES: tuple[str, ...] = (
    # UTF-8 bytes re-decoded as Latin-1/cp1252 -- the signatures French corpora
    # acquire through mis-decoded scrapes. All multi-byte, so accented French
    # words like "caf\u00e9" or "\u00e2ge" never match.
    "Ã©",  # e-acute
    "Ã¨",  # e-grave
    "Ãª",  # e-circumflex
    "Ã«",  # e-diaeresis
    "Ã§",  # c-cedilla
    "Ã\u00a0",  # a-grave (NBSP second byte kept as an escape: invisible in source)
    "Ã¢",  # a-circumflex
    "Ã®",  # i-circumflex
    "Ã\u00b4",  # o-circumflex (U+00B4 second byte kept as an escape: ruff RUF001-ambiguous)
    "Ã»",  # u-circumflex
    "â€™",  # right single quote
    "â€œ",  # left double quote
    "â€\u009d",  # right double quote
    "Â\u00a0",  # non-breaking space (NBSP kept as an escape: invisible in source)
)

TOKENIZER_ID = "Qwen/Qwen3.5-4B"
"""Tokenizer matching the served/trained family; any size in the family shares
it, so token counts here agree with the serving preflight in eval/runner.py."""

_whitespace_fallback_note = (
    "whitespace approximation: len(text.split()), which undercounts subword "
    "tokens (~1.3x for English medical text) and is therefore lenient at both bounds"
)

_loaded_tokenizer: Any = None
_tokenizer_unavailable = False


def _tokenizer_or_none() -> Any:
    """Lazily load TOKENIZER_ID from the local HF cache; None when unavailable.

    local_files_only keeps the stage offline-deterministic: the cluster always
    has the cache (serving and training resolve the same id); a dev box without
    it gets the documented whitespace fallback instead of a Hub fetch.
    """
    global _loaded_tokenizer, _tokenizer_unavailable
    if _loaded_tokenizer is None and not _tokenizer_unavailable:
        try:
            from transformers import AutoTokenizer

            _loaded_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
        except Exception as exc:
            _tokenizer_unavailable = True
            log.warning(
                "S2 tokenizer %s unavailable (%s); %s", TOKENIZER_ID, exc, _whitespace_fallback_note
            )
    return _loaded_tokenizer


def count_tokens(text: str) -> int:
    """Token count over ``text`` with the cached family tokenizer.

    Special tokens are excluded: the bounds bound *content*, and chat-template
    overhead is absorbed by the max_tokens headroom under the cluster's
    max_model_len. Falls back to whitespace words when the tokenizer is not
    cached -- see ``_whitespace_fallback_note`` for the bias direction.
    """
    tok = _tokenizer_or_none()
    if tok is None:
        return len(text.split())
    return len(tok(text, add_special_tokens=False)["input_ids"])


def tokenizer_mode() -> str:
    """Which counting path is live: TOKENIZER_ID, or 'whitespace_approx'."""
    return TOKENIZER_ID if _tokenizer_or_none() is not None else "whitespace_approx"


# --------------------------------------------------------------------------
# Text surfaces -- one definition of "which text does each check read".
# --------------------------------------------------------------------------


def _user_text(item: CorpusItem) -> str:
    return "\n".join(m.get("content", "") for m in item.messages if m.get("role") == "user")


def _assistant_text(item: CorpusItem) -> str:
    return "\n".join(m.get("content", "") for m in item.messages if m.get("role") == "assistant")


def _row_text(item: CorpusItem) -> str:
    """The whole trained row: every message plus the thinking trace.

    Length and encoding are properties of what a training step would consume,
    so system messages count too; the gold ``answer`` column does not (it is a
    label, not trained-on content at S2 time).
    """
    return "\n".join(m.get("content", "") for m in item.messages) + "\n" + (item.thinking or "")


def _has_repeated_span(text: str, span_chars: int, max_occurrences: int) -> bool:
    """True when any ``span_chars``-char window occurs more than max times.

    Short-circuits at the first window that crosses the bound, so the common
    clean row pays one Counter pass and degenerate rows pay almost nothing.
    """
    counts: Counter[str] = Counter()
    for i in range(len(text) - span_chars + 1):
        window = text[i : i + span_chars]
        counts[window] += 1
        if counts[window] > max_occurrences:
            return True
    return False


def _encoding_damage(text: str) -> float:
    """(U+FFFD count + mojibake-sequence occurrences) / total chars."""
    if not text:
        return 0.0
    hits = text.count(REPLACEMENT_CHAR) + sum(text.count(seq) for seq in MOJIBAKE_SEQUENCES)
    return hits / len(text)


# --------------------------------------------------------------------------
# The check registry. Keys MUST be Flags fields (validated at stage start).
# Each predicate returns True = set the flag. Individually testable by design;
# thresholds are read at call time so tests can patch a modified THRESHOLDS.
# --------------------------------------------------------------------------

CheckFn = Callable[[CorpusItem], bool]


def _check_empty(item: CorpusItem) -> bool:
    """No question: user content strips to nothing (or no user turn exists)."""
    return not _user_text(item).strip()


def _check_length(item: CorpusItem) -> bool:
    """Row token count outside [min_tokens, max_tokens].

    max_tokens mirrors the cluster's max_model_len headroom -- a row that
    cannot be trained is waste the GPUs should never see; min_tokens kills
    fragments too short to carry learnable signal.
    """
    n = count_tokens(_row_text(item))
    return n < THRESHOLDS.min_tokens or n > THRESHOLDS.max_tokens


def _check_truncated(item: CorpusItem) -> bool:
    """An opened think trace that never closes, in assistant content or thinking.

    Exact eval-side convention: a closed pair (or a close without an open --
    the serving prefill owns the opening tag) is complete by definition.
    """
    for text in (_assistant_text(item), item.thinking or ""):
        if THINK_OPEN in text and THINK_CLOSE not in text:
            return True
    return False


def _check_repetition(item: CorpusItem) -> bool:
    """A repeated span in the assistant answer or thinking trace.

    Scanned separately per surface: a join could only manufacture windows that
    exist in neither text alone.
    """
    for text in (_assistant_text(item), item.thinking or ""):
        if _has_repeated_span(
            text, THRESHOLDS.repetition_span_chars, THRESHOLDS.repetition_max_occurrences
        ):
            return True
    return False


def _check_lang(item: CorpusItem) -> bool:
    """S1's LID verdict is outside the program's working languages."""
    return item.lang not in ALLOWED_LANGS


def _check_encoding(item: CorpusItem) -> bool:
    """Mojibake or replacement characters above the damage ratio."""
    return _encoding_damage(_row_text(item)) > THRESHOLDS.encoding_damage_max_ratio


def _check_refusal(item: CorpusItem) -> bool:
    """Assistant content matches a refusal template (case-insensitive)."""
    return _REFUSAL_RE.search(_assistant_text(item)) is not None


CHECKS: dict[str, CheckFn] = {
    "f_empty": _check_empty,
    "f_length": _check_length,
    "f_truncated": _check_truncated,
    "f_repetition": _check_repetition,
    "f_lang": _check_lang,
    "f_encoding": _check_encoding,
    "f_refusal": _check_refusal,
}


def validate_check_registry() -> None:
    """Fail loudly when a CHECKS key is not a Flags field.

    The parquet projection carries flags as flat ``flags_*`` columns built from
    ``Flags.model_fields`` (store.py); a check writing an unknown flag would
    silently vanish at the write boundary, so the stage refuses to start
    instead of producing rows whose reported flags are not on disk.
    """
    unknown = sorted(set(CHECKS) - set(Flags.model_fields))
    if unknown:
        raise StageError(f"S2: CHECKS keys are not Flags fields: {unknown}")


# --------------------------------------------------------------------------
# Tally + stage entry.
# --------------------------------------------------------------------------


@dataclass
class FlagTally:
    """Per-source set-counts accumulated while the stage streams.

    Kept as counts, not running rates, so the manifest's rates are exactly
    sets/rows per source (plus the '_all' pooled rate) and unit-testable
    without touching parquet.
    """

    flag_names: Sequence[str]
    rows_per_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sets_per_flag: dict[str, dict[str, int]] = field(default_factory=dict)
    rows_with_any_flag: int = 0

    def __post_init__(self) -> None:
        if not self.sets_per_flag:
            self.sets_per_flag = {name: defaultdict(int) for name in self.flag_names}

    def record(self, source: str, hits: Mapping[str, bool]) -> None:
        self.rows_per_source[source] += 1
        any_hit = False
        for name, hit in hits.items():
            if hit:
                self.sets_per_flag[name][source] += 1
                any_hit = True
        if any_hit:
            self.rows_with_any_flag += 1

    def rates(self) -> dict[str, dict[str, float]]:
        """flag -> {source: set-rate} + '_all'; every seen source appears, 0.0 included."""
        total = sum(self.rows_per_source.values())
        out: dict[str, dict[str, float]] = {}
        for name in self.flag_names:
            sets = self.sets_per_flag[name]
            per_source: dict[str, float] = {
                src: sets[src] / n for src, n in sorted(self.rows_per_source.items())
            }
            per_source["_all"] = (sum(sets.values()) / total) if total else 0.0
            out[name] = per_source
        return out


def _flagged(items: Iterator[CorpusItem], tally: FlagTally) -> Iterator[CorpusItem]:
    """Stream: run every check, set hit flags (False -> True only), tally.

    Upstream flags are preserved: S2 sets its own names and never clears --
    a correction means re-running the stage, not editing rows.
    """
    for item in items:
        hits = {name: check(item) for name, check in CHECKS.items()}
        tally.record(item.source, hits)
        updates = {name: True for name, hit in hits.items() if hit}
        if updates:
            yield item.model_copy(update={"flags": item.flags.model_copy(update=updates)})
        else:
            yield item


def stage_entry(
    run_id: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> StageManifest:
    """S2: flag structural defects on every row of 01_normalize -> 02_structural.

    Streams the input snapshot once; the same pass writes the output (sorted by
    id inside write_items) and accumulates the per-source flag tallies that
    become ``flag_rates``. Optional input/output dirs let tests run against
    tmp_path instead of the run's scratch layout.

    A missing input snapshot is a StageError, not an empty pass: running S2
    over a directory S1 never wrote would silently produce a zero-row
    02_structural and overwrite a good snapshot with it.
    """
    started = utcnow()
    inp = input_dir if input_dir is not None else store.stage_dir(run_id, INPUT_STAGE)
    out = output_dir if output_dir is not None else store.stage_dir(run_id, OUTPUT_STAGE)
    if not inp.is_dir():
        raise StageError(f"S2: input snapshot {inp} does not exist -- run {INPUT_STAGE} first")
    out.mkdir(parents=True, exist_ok=True)  # explicit dirs skip stage_dir's mkdir
    validate_check_registry()

    manifest = StageManifest(
        run_id=run_id,
        stage=OUTPUT_STAGE,
        started_at=started,
        config={
            "input_dir": str(inp),
            "checks": sorted(CHECKS),
            "allowed_langs": sorted(ALLOWED_LANGS),
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_mode": tokenizer_mode(),
        },
        thresholds=snapshot(THRESHOLDS),
    )

    tally = FlagTally(tuple(CHECKS))
    store.write_items(_flagged(store.iter_items(inp), tally), out)

    manifest.rows_in = store.count_items(inp)
    manifest.rows_out = store.count_items(out)
    manifest.input_sha256 = store.content_sha256(inp)
    manifest.output_sha256 = store.content_sha256(out)
    manifest.flag_rates = tally.rates()
    manifest.notes = {"rows_flagged_any": tally.rows_with_any_flag}
    assert manifest.rows_in == manifest.rows_out, "flags-not-deletes: S2 never drops"
    return manifest


__all__ = [
    "ALLOWED_LANGS",
    "CHECKS",
    "INPUT_STAGE",
    "MOJIBAKE_SEQUENCES",
    "OUTPUT_STAGE",
    "REFUSAL_PATTERNS",
    "THINK_CLOSE",
    "THINK_OPEN",
    "TOKENIZER_ID",
    "FlagTally",
    "count_tokens",
    "stage_entry",
    "tokenizer_mode",
    "validate_check_registry",
]
