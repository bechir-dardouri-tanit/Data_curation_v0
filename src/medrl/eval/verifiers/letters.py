"""Letter-identity verification for MCQA answers.

Gold columns disagree with themselves about format ("B", "b", "(B)", "B.", "B)"), and a
tolerance implemented twice -- once in eval, once in rewards -- eventually disagrees with
itself too, which is precisely the skew this package exists to prevent. So the alphabet
normalization lives in exactly one place (:func:`medrl.eval.extraction.normalize_letter`)
and both paths call it.

Everything here is total: a malformed gold (say ``"Beta"``) returns ``False`` instead of
raising, because gold strings flow from dataset columns and a crash would turn one dirty
row into a failed run.
"""

from __future__ import annotations

from medrl.eval.extraction import normalize_letter


def verify_letter(predicted: str | None, gold: str, letters: str = "ABCDE") -> bool:
    """Case-insensitive equality within ``letters``, decoration-tolerant on *either* side.

    ``predicted`` may be ``None`` (extraction failed) -- that is a plain ``False``, the
    same verdict the scorer would reach, so reward code can feed extraction output in
    directly. A gold that does not normalize to a single letter also yields ``False``:
    it is a dataset bug that should surface as a surprising 0% on that item and be fixed
    upstream, not an exception mid-rollout. ``letters`` is the task's alphabet (A-J for
    MMLU-Pro-style tasks) and must match what extraction was called with.
    """
    pred = normalize_letter(predicted, letters)
    gold_letter = normalize_letter(gold, letters)
    return pred is not None and pred == gold_letter
