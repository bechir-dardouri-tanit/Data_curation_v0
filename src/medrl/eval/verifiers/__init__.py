"""Ground-truth verifiers shared by evaluation scoring and RL rewards.

Same principle as :mod:`medrl.eval.extraction`: one implementation, used on both sides,
so the reward can never reward what the eval would not score. Import from here, not from
the submodules -- the package boundary is the stable API.
"""

from __future__ import annotations

from medrl.eval.verifiers.format_rules import FORMAT_RULES, FormatRule
from medrl.eval.verifiers.letters import verify_letter
from medrl.eval.verifiers.numbers import parse_quantity, verify_number

__all__ = [
    "FORMAT_RULES",
    "FormatRule",
    "parse_quantity",
    "verify_letter",
    "verify_number",
]
