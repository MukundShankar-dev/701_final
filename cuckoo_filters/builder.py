"""Helpers for building Cuckoo filters from keys or k-mer files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from cuckoo_filters.cuckoo_filter import CuckooFilter
from data.io_utils import load_kmers


def build_cuckoo_filter(
    keys: Sequence[str],
    *,
    capacity: int | None = None,
    bucket_size: int = 4,
    fingerprint_bits: int = 12,
    max_relocations: int = 500,
    random_seed: int = 0,
) -> CuckooFilter:
    """Build a Cuckoo filter from key sequence."""
    cap = capacity if capacity is not None else len(keys)
    if cap <= 0:
        raise ValueError(f"capacity must be > 0, got {cap}")

    filt = CuckooFilter(
        capacity=cap,
        bucket_size=bucket_size,
        fingerprint_bits=fingerprint_bits,
        max_relocations=max_relocations,
        random_seed=random_seed,
    )
    filt.build(keys)
    return filt


def build_cuckoo_filter_from_kmer_file(
    kmer_file: str | Path,
    *,
    capacity: int | None = None,
    bucket_size: int = 4,
    fingerprint_bits: int = 12,
    max_relocations: int = 500,
    random_seed: int = 0,
    deduplicate: bool = False,
) -> CuckooFilter:
    """Build a Cuckoo filter from one-k-mer-per-line file."""
    keys = list(load_kmers(kmer_file, deduplicate=deduplicate))
    cap = capacity if capacity is not None else len(keys)
    return build_cuckoo_filter(
        keys=keys,
        capacity=cap,
        bucket_size=bucket_size,
        fingerprint_bits=fingerprint_bits,
        max_relocations=max_relocations,
        random_seed=random_seed,
    )
