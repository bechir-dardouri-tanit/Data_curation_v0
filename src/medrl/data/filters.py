"""Quality filters and validators for the medrl pipeline.

Implements data quality checks: language detection, length filtering, pattern
matching, and medical domain validation. Filters are composable and can be
applied in sequence to build a quality pipeline.

Typical usage:
    >>> from medrl.data.filters import LengthFilter, LanguageFilter
    >>> items = [{"text": "Hi"}, {"text": "A" * 10000}]
    >>> length_filter = LengthFilter(min_length=5, max_length=5000)
    >>> filtered = [item for item in items if length_filter(item)]
    >>> len(filtered)
    1
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from medrl.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Filter Base Classes
# =============================================================================


class Filter(ABC):
    """Abstract base class for data filters.

    All filters implement __call__ with a consistent signature: accept an item
    (dict) and return True if the item passes, False otherwise.

    Examples:
        >>> class MyFilter(Filter):
        ...     def __call__(self, item):
        ...         return len(item.get("text", "")) > 10
    """

    @abstractmethod
    def __call__(self, item: dict[str, Any]) -> bool:
        """Return True if item passes the filter, False otherwise.

        Args:
            item: Data item as a dictionary.

        Returns:
            True if item passes, False to reject.
        """
        ...

    def __and__(self, other: Filter) -> Filter:
        """Combine two filters with AND logic.

        Examples:
            >>> f1 = LengthFilter(min_length=10)
            >>> f2 = LanguageFilter(allowed=["en"])
            >>> combined = f1 & f2
        """
        return AndFilter(self, other)

    def __or__(self, other: Filter) -> Filter:
        """Combine two filters with OR logic."""
        return OrFilter(self, other)

    def __invert__(self) -> Filter:
        """Negate a filter.

        Examples:
            >>> f = LengthFilter(min_length=10)
            >>> negated = ~f  # Passes items shorter than 10
        """
        return NotFilter(self)


@dataclass(frozen=True)
class AndFilter(Filter):
    """Logical AND of two filters. Passes only if both pass."""

    left: Filter
    right: Filter

    def __call__(self, item: dict[str, Any]) -> bool:
        return self.left(item) and self.right(item)


@dataclass(frozen=True)
class OrFilter(Filter):
    """Logical OR of two filters. Passes if either passes."""

    left: Filter
    right: Filter

    def __call__(self, item: dict[str, Any]) -> bool:
        return self.left(item) or self.right(item)


@dataclass(frozen=True)
class NotFilter(Filter):
    """Logical negation of a filter."""

    filter: Filter

    def __call__(self, item: dict[str, Any]) -> bool:
        return not self.filter(item)


# =============================================================================
# Text Quality Filters
# =============================================================================


@dataclass(frozen=True)
class LengthFilter(Filter):
    """Filter by text length.

    Removes items that are too short (low information) or too long (likely
    corrupted or multi-document concatenations).

    Attributes:
        text_key: Dictionary key containing the text.
        min_length: Minimum character length (inclusive).
        max_length: Maximum character length (inclusive).

    Examples:
        >>> f = LengthFilter(min_length=10, max_length=1000)
        >>> f({"text": "Hello"})
        False
        >>> f({"text": "Hello world!"})
        True
    """

    text_key: str = "text"
    min_length: int = 10
    max_length: int = 10000

    def __call__(self, item: dict[str, Any]) -> bool:
        text = str(item.get(self.text_key, ""))
        length = len(text)
        if length < self.min_length:
            logger.debug(f"Item too short: {length} < {self.min_length}")
            return False
        if length > self.max_length:
            logger.debug(f"Item too long: {length} > {self.max_length}")
            return False
        return True


@dataclass(frozen=True)
class WordCountFilter(Filter):
    """Filter by word count.

    Similar to LengthFilter but counts words instead of characters. More
    robust for different languages with different character densities.

    Attributes:
        text_key: Dictionary key containing the text.
        min_words: Minimum word count (inclusive).
        max_words: Maximum word count (inclusive).

    Examples:
        >>> f = WordCountFilter(min_words=3, max_words=500)
        >>> f({"text": "Hello world"})
        False
        >>> f({"text": "Hello world test"})
        True
    """

    text_key: str = "text"
    min_words: int = 5
    max_words: int = 2000

    def __call__(self, item: dict[str, Any]) -> bool:
        text = str(item.get(self.text_key, ""))
        words = text.split()
        count = len(words)
        if count < self.min_words:
            logger.debug(f"Item has too few words: {count} < {self.min_words}")
            return False
        if count > self.max_words:
            logger.debug(f"Item has too many words: {count} > {self.max_words}")
            return False
        return True


@dataclass(frozen=True)
class RatioFilter(Filter):
    """Filter by special character ratio.

    Rejects items with too many special characters (URLs, gibberish) or too many
    numbers (data-heavy, not natural language).

    Attributes:
        text_key: Dictionary key containing the text.
        max_special_ratio: Maximum ratio of special chars to total chars (0-1).
        max_number_ratio: Maximum ratio of digits to total chars (0-1).

    Examples:
        >>> f = RatioFilter(max_special_ratio=0.3)
        >>> f({"text": "!!!***@@@"})  # High special char ratio
        False
    """

    text_key: str = "text"
    max_special_ratio: float = 0.5
    max_number_ratio: float = 0.3

    def __call__(self, item: dict[str, Any]) -> bool:
        text = str(item.get(self.text_key, ""))
        if not text:
            return False

        total = len(text)

        # Count special characters (non-alphanumeric, non-space)
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special / total if total > 0 else 0

        # Count digits
        digits = sum(1 for c in text if c.isdigit())
        number_ratio = digits / total if total > 0 else 0

        if special_ratio > self.max_special_ratio:
            logger.debug(f"Item has too many special chars: {special_ratio:.2f}")
            return False
        if number_ratio > self.max_number_ratio:
            logger.debug(f"Item has too many digits: {number_ratio:.2f}")
            return False

        return True


# =============================================================================
# Language Filters
# =============================================================================


@dataclass(frozen=True)
class LanguageFilter(Filter):
    """Filter by detected language.

    Uses fast language detection to keep only items in target languages.

    Attributes:
        text_key: Dictionary key containing the text.
        allowed: Set of allowed ISO 639-1 language codes.
        min_confidence: Minimum detection confidence (0-1).

    Raises:
        ImportError: If langdetect is not installed.

    Examples:
        >>> f = LanguageFilter(allowed=["en", "es"])
        >>> f({"text": "Hello world"})  # English
        True
        >>> f({"text": "Bonjour monde"})  # French
        False
    """

    text_key: str = "text"
    allowed: tuple[str, ...] = ("en",)
    min_confidence: float = 0.8

    def __call__(self, item: dict[str, Any]) -> bool:
        try:
            from langdetect import LangDetectException, detect  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "langdetect is required for language filtering. "
                "Install with: pip install langdetect"
            ) from e

        text = str(item.get(self.text_key, ""))

        try:
            lang = detect(text)
        except LangDetectException:
            logger.debug("Language detection failed, rejecting item")
            return False

        if lang not in self.allowed:
            logger.debug(f"Item language '{lang}' not in allowed {self.allowed}")
            return False

        return True


# =============================================================================
# Pattern Filters
# =============================================================================


@dataclass(frozen=True)
class RegexFilter(Filter):
    r"""Filter by regex pattern matching.

    Passes items that match (or don't match) a regex pattern. Useful for
    detecting and removing boilerplate, disclaimers, or specific content types.

    Attributes:
        text_key: Dictionary key containing the text.
        pattern: Regex pattern to match.
        invert: If True, reject matching items (pass non-matching).

    Examples:
        >>> # Remove items containing URLs
        >>> url_filter = RegexFilter(pattern=r'https?://\S+', invert=True)
        >>> url_filter({"text": "Visit https://example.com"})
        False
    """

    text_key: str = "text"
    pattern: str = r".*"
    invert: bool = False
    flags: int = re.IGNORECASE
    _compiled: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        """Compile the regex pattern."""
        object.__setattr__(self, "_compiled", re.compile(self.pattern, self.flags))

    def __call__(self, item: dict[str, Any]) -> bool:
        text = str(item.get(self.text_key, ""))
        matches = bool(self._compiled.search(text))
        return not matches if self.invert else matches


@dataclass(frozen=True)
class BoilerplateFilter(Filter):
    """Remove common boilerplate and disclaimers.

    Detects and filters out common legal disclaimers, copyright notices, and
        text_key: Dictionary key containing the text.
        patterns: List of regex patterns for boilerplate content.

    Examples:
        >>> f = BoilerplateFilter()
        >>> f({"text": "Copyright 2024 All rights reserved"})
        False
    """

    text_key: str = "text"
    patterns: tuple[str, ...] = (
        r"all rights reserved",
        r"copyright ©?\s*\d{4}",
        r"terms of service",
        r"privacy policy",
        r"by using this",
        r"not a substitute for professional medical advice",
        r"please consult a doctor",
    )

    def __call__(self, item: dict[str, Any]) -> bool:
        text = str(item.get(self.text_key, "")).lower()

        for pattern in self.patterns:
            if re.search(pattern, text):
                logger.debug(f"Item matches boilerplate pattern: {pattern[:30]}...")
                return False

        return True


# =============================================================================
# Medical Domain Filters
# =============================================================================


@dataclass(frozen=True)
class MedicalTerminologyFilter(Filter):
    """Keep items containing medical terminology.

    Ensures the dataset is focused on medical content by requiring at least
    some medical vocabulary. Useful for filtering general-purpose data.

    Attributes:
        text_key: Dictionary key containing the text.
        min_terms: Minimum number of medical terms required.
        terminology: Tuple of medical terms/phrases to look for.

    Examples:
        >>> f = MedicalTerminologyFilter(min_terms=1)
        >>> f({"text": "The patient has hypertension and diabetes."})
        True
        >>> f({"text": "I like pizza and video games."})
        False
    """

    text_key: str = "text"
    min_terms: int = 1
    terminology: tuple[str, ...] = (
        # Common medical terms
        "patient",
        "diagnosis",
        "symptom",
        "treatment",
        "therapy",
        "medication",
        "disease",
        "syndrome",
        "clinical",
        "medical",
        "prescription",
        "dosage",
        "adverse",
        "effect",
        "contraindication",
        "hypertension",
        "diabetes",
        "pain",
        "fever",
        "inflammation",
        "infection",
        "antibiotic",
        "analgesic",
        "nsaid",
    )

    def __call__(self, item: dict[str, Any]) -> bool:
        text = str(item.get(self.text_key, "")).lower()

        # Count matching terms
        matches = sum(1 for term in self.terminology if term.lower() in text)

        if matches < self.min_terms:
            logger.debug(f"Item contains only {matches} medical terms (min {self.min_terms})")
            return False

        return True


@dataclass(frozen=True)
class PIIFilter(Filter):
    """Filter items containing potential PII (personally identifiable information).

    Detects common PII patterns like email addresses, phone numbers, and SSNs.
    Privacy-critical for medical data.

    Attributes:
        text_key: Dictionary key containing the text.
        patterns: Regex patterns for PII detection.

    Examples:
        >>> f = PIIFilter()
        >>> f({"text": "Email me at john@example.com"})
        False
    """

    text_key: str = "text"
    patterns: tuple[str, ...] = (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{3}-\d{3}-\d{4}\b",  # Phone
        r"\b\d{10,}\b",  # Long numbers (potential IDs)
    )

    def __call__(self, item: dict[str, Any]) -> bool:
        text = str(item.get(self.text_key, ""))

        for pattern in self.patterns:
            if re.search(pattern, text):
                logger.debug(f"Item contains potential PII matching: {pattern[:30]}...")
                return False

        return True


# =============================================================================
# Composite Filter Pipeline
# =============================================================================


class FilterPipeline:
    """A sequence of filters applied in order.

    Tracks statistics and provides logging for each filter stage.

    Attributes:
        filters: List of (name, filter) tuples.

    Examples:
        >>> pipeline = FilterPipeline()
        >>> pipeline.add("length", LengthFilter(min_length=10))
        >>> pipeline.add("language", LanguageFilter(allowed=["en"]))
        >>> filtered = pipeline.filter(items)
    """

    def __init__(self) -> None:
        """Initialize an empty filter pipeline."""
        self.filters: list[tuple[str, Filter]] = []

    def add(self, name: str, filter: Filter) -> FilterPipeline:
        """Add a filter to the pipeline.

        Args:
            name: Identifier for this filter stage.
            filter: The filter instance.

        Returns:
            Self for chaining.
        """
        self.filters.append((name, filter))
        return self

    def filter(self, items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply all filters to the items.

        Args:
            items: Input items to filter.

        Returns:
            List of items that passed all filters.
        """
        current = list(items)
        stats: dict[str, int] = {}

        for name, filter in self.filters:
            before = len(current)
            current = [item for item in current if filter(item)]
            after = len(current)
            removed = before - after
            stats[name] = removed

            logger.info(f"Filter '{name}': {before} -> {after} (removed {removed})")

        logger.info(f"Filter pipeline complete: {len(current)} items retained")
        return current

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Return statistics about the filters."""
        # Note: This would need state tracking to be fully implemented
        return {name: {"filter": str(filter)} for name, filter in self.filters}


# =============================================================================
# Validator Functions
# =============================================================================


def validate_item(item: dict[str, Any], required_keys: tuple[str, ...]) -> bool:
    """Validate that an item has all required keys.

    Args:
        item: Item dictionary.
        required_keys: Tuple of required key names.

    Returns:
        True if all required keys are present.

    Examples:
        >>> validate_item({"text": "Hello"}, ("text", "source"))
        False
    """
    return all(key in item for key in required_keys)


def validate_response_text(text: str, min_length: int = 1) -> bool:
    """Validate response text for training.

    Checks that response is non-empty and not just whitespace.

    Args:
        text: Response text to validate.
        min_length: Minimum character length.

    Returns:
        True if response is valid.
    """
    return len(text.strip()) >= min_length


def validate_message_format(messages: list[dict[str, str]]) -> bool:
    """Validate OpenAI-style message format.

    Ensures messages have required 'role' and 'content' keys and valid roles.

    Args:
        messages: List of message dictionaries.

    Returns:
        True if messages are correctly formatted.
    """
    if not messages:
        return False

    valid_roles = {"system", "user", "assistant"}

    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in valid_roles:
            return False
        if not isinstance(msg["content"], str):
            return False

    return True
