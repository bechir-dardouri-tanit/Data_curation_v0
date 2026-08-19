"""Typed registries.

Benchmarks, scorers, verifiers and reward components are all looked up by string key so a
YAML config can name them. A plain dict would do, but a registry that fails loudly on
duplicate keys and reports near-misses on lookup failure turns a whole class of config
typos into an immediate, readable error.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> object mapping with duplicate detection and did-you-mean lookup."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T | None = None) -> Callable[[T], T] | T:
        """Register ``obj`` under ``name``. Usable as a decorator or a direct call."""
        if obj is not None:
            self._insert(name, obj)
            return obj

        def decorator(inner: T) -> T:
            self._insert(name, inner)
            return inner

        return decorator

    def _insert(self, name: str, obj: T) -> None:
        if name in self._items:
            raise ValueError(
                f"{self._kind} {name!r} is already registered by "
                f"{getattr(self._items[name], '__qualname__', self._items[name])}"
            )
        self._items[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            close = difflib.get_close_matches(name, self._items, n=3, cutoff=0.5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise KeyError(
                f"unknown {self._kind} {name!r}. Known: {sorted(self._items)}.{hint}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> list[str]:
        return sorted(self._items)
