"""Core interfaces and metadata objects for all AMQ implementations.

This module is intentionally lightweight and dependency-free so every filter
backend (Bloom, Cuckoo, XOR, learned) can share the same contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self, Sequence, runtime_checkable


@dataclass(slots=True)
class FilterBuildMetadata:
    """Build-time metadata captured for a filter instance."""

    filter_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    inserted_keys: int = 0
    target_false_positive_rate: float | None = None
    actual_memory_usage_bytes: int | None = None
    build_time_seconds: float | None = None

    def memory_per_key_bytes(self) -> float | None:
        """Return bytes per inserted key when available."""
        if self.inserted_keys <= 0 or self.actual_memory_usage_bytes is None:
            return None
        return self.actual_memory_usage_bytes / self.inserted_keys

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@runtime_checkable
class AMQFilter(Protocol):
    """Shared interface for approximate membership query filters."""

    def build(self, keys: Sequence[str]) -> None:
        """Build or rebuild the filter from a full key sequence."""

    def contains(self, key: str) -> bool:
        """Return membership result for one key."""

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        """Return membership results for a sequence of keys."""

    def memory_usage_bytes(self) -> int:
        """Return in-memory usage estimate in bytes."""

    def save(self, path: str | Path) -> None:
        """Serialize filter state to disk."""

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load serialized filter state and return a new instance."""

    def stats(self) -> dict[str, Any]:
        """Return implementation-specific statistics and metadata."""


class AMQFilterBase(ABC):
    """Abstract base class with a shared default for batch queries."""

    metadata: FilterBuildMetadata | None

    @abstractmethod
    def build(self, keys: Sequence[str]) -> None:
        """Build or rebuild the filter from a full key sequence."""

    @abstractmethod
    def contains(self, key: str) -> bool:
        """Return membership result for one key."""

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        """Default batch query implementation via repeated single queries."""
        return [self.contains(key) for key in keys]

    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """Return in-memory usage estimate in bytes."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialize filter state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Load serialized filter state and return a new instance."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return implementation-specific statistics and metadata."""


@runtime_checkable
class SupportsDeletion(Protocol):
    """Optional capability protocol for filters that support deletion."""

    def delete(self, key: str) -> bool:
        """Delete a key if present and return whether deletion succeeded."""


__all__ = [
    "AMQFilter",
    "AMQFilterBase",
    "FilterBuildMetadata",
    "SupportsDeletion",
]
