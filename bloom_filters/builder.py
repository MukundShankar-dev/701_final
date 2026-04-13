"""Helpers for building Bloom filters from in-memory keys or files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from bloom_filters.bloom_filter import BloomFilter
from data.io_utils import load_kmers


def build_bloom_filter(
    keys: Sequence[str],
    *,
    expected_items: int | None = None,
    false_positive_rate: float = 1e-3,
    hash_seed: int = 0,
) -> BloomFilter:
    """Build a Bloom filter from key sequence."""
    n = expected_items if expected_items is not None else len(keys)
    if n <= 0:
        raise ValueError(f"expected_items must be > 0, got {n}")

    filt = BloomFilter(
        expected_items=n,
        false_positive_rate=false_positive_rate,
        hash_seed=hash_seed,
    )
    filt.build(keys)
    return filt


def build_bloom_filter_from_kmer_file(
    kmer_file: str | Path,
    *,
    expected_items: int | None = None,
    false_positive_rate: float = 1e-3,
    hash_seed: int = 0,
    deduplicate: bool = False,
) -> BloomFilter:
    """Build a Bloom filter from one-k-mer-per-line file."""
    keys = list(load_kmers(kmer_file, deduplicate=deduplicate))
    n = expected_items if expected_items is not None else len(keys)
    return build_bloom_filter(
        keys=keys,
        expected_items=n,
        false_positive_rate=false_positive_rate,
        hash_seed=hash_seed,
    )
