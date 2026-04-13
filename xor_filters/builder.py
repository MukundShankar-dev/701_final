"""Helpers for building XOR filter facade from keys or k-mer files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from data.io_utils import load_kmers
from xor_filters.xor_filter import XorFilter


def build_xor_filter(
    keys: Sequence[str],
    *,
    fingerprint_bits: int = 8,
    backend: str = "auto",
) -> XorFilter:
    """Build XOR filter facade from key sequence."""
    if len(keys) == 0:
        raise ValueError("keys must be non-empty")

    filt = XorFilter(fingerprint_bits=fingerprint_bits, backend=backend)
    filt.build(keys)
    return filt


def build_xor_filter_from_kmer_file(
    kmer_file: str | Path,
    *,
    fingerprint_bits: int = 8,
    backend: str = "auto",
    deduplicate: bool = False,
) -> XorFilter:
    """Build XOR filter facade from one-k-mer-per-line file."""
    keys = list(load_kmers(kmer_file, deduplicate=deduplicate))
    return build_xor_filter(
        keys=keys,
        fingerprint_bits=fingerprint_bits,
        backend=backend,
    )
