"""Backup AMQ filter used by learned-filter pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from bloom_filters.bloom_filter import BloomFilter


@dataclass(slots=True)
class BackupBloomFilter:
    """Backup Bloom filter that protects against model false negatives."""

    bloom: BloomFilter

    @classmethod
    def build(
        cls,
        positive_kmers: Sequence[str],
        *,
        false_positive_rate: float = 1e-3,
        hash_seed: int = 0,
    ) -> "BackupBloomFilter":
        bloom = BloomFilter(
            expected_items=max(1, len(positive_kmers)),
            false_positive_rate=false_positive_rate,
            hash_seed=hash_seed,
        )
        bloom.build(list(positive_kmers))
        return cls(bloom=bloom)

    def contains(self, key: str) -> bool:
        return self.bloom.contains(key)

    def save(self, path: str | Path) -> None:
        self.bloom.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "BackupBloomFilter":
        return cls(bloom=BloomFilter.load(path))
