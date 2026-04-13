"""Common interfaces and metadata objects for AMQ filters.

This module defines the core contracts used across Bloom, Cuckoo, XOR,
and learned filter implementations. Keeping these contracts centralized
makes benchmarking consistent and enables backend swapping later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self, Sequence


@dataclass(slots=True)
class FilterBuildMetadata:
    """Metadata captured after building an AMQ filter instance."""

    filter_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    inserted_keys: int = 0
    target_false_positive_rate: float | None = None
    actual_memory_usage_bytes: int | None = None
    build_time_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


class AMQFilter(Protocol):
    """Protocol implemented by all approximate membership query filters.

    Implementations should provide deterministic behavior given a fixed seed
    or deterministic hash strategy, and they should never produce false
    negatives for inserted keys unless the structure is explicitly documented
    as probabilistic in that dimension (not expected for current filters).
    """

    def build(self, keys: Sequence[str]) -> None:
        """Build or rebuild the filter from a sequence of keys."""

    def contains(self, key: str) -> bool:
        """Return membership query result for a single key."""

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        """Return membership query results for a batch of keys."""

    def memory_usage_bytes(self) -> int:
        """Return in-memory size estimate in bytes for this filter."""

    def save(self, path: str | Path) -> None:
        """Serialize filter state to disk."""

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load serialized filter state from disk and return an instance."""

    def stats(self) -> dict[str, Any]:
        """Return implementation-specific runtime statistics and metadata."""


class SupportsDeletion(Protocol):
    """Optional capability protocol for AMQ filters supporting deletions."""

    def delete(self, key: str) -> bool:
        """Delete a key if present and return whether deletion succeeded."""
