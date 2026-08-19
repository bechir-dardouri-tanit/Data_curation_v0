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


def verify_letter(predicted: str | None, gold: str) -> bool:
    """Case-insensitive A-E equality with decoration tolerance on *either* side.

    ``predicted`` may be ``None`` (extraction failed) -- that is a plain ``False``, the
    same verdict the scorer would reach, so reward code can feed extraction output in
    directly. A gold that does not normalize to a single letter also yields ``False``:
    it is a dataset bug that should surface as a surprising 0% on that item and be fixed
    upstream, not an exception mid-rollout.
    """
    pred = normalize_letter(predicted)
    gold_letter = normalize_letter(gold)
    return pred is not None and pred == gold_letter
