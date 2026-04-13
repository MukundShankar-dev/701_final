"""k-mer extraction and transformation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.fasta_loader import iter_fasta_sequences

_DNA_ALPHABET = frozenset({"A", "C", "G", "T"})
_RC_MAP = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "a": "t", "c": "g", "g": "c", "t": "a"})


def reverse_complement(seq: str) -> str:
    """Return reverse complement for a DNA sequence.

    Raises:
        ValueError: If the sequence contains non-ACGT characters.
    """
    upper = seq.upper()
    if any(base not in _DNA_ALPHABET for base in upper):
        raise ValueError(f"Sequence contains non-ACGT character(s): {seq}")
    return upper.translate(_RC_MAP)[::-1].upper()


def canonical_kmer(kmer: str) -> str:
    """Return lexicographically canonical representation of a k-mer."""
    upper = kmer.upper()
    rc = reverse_complement(upper)
    return min(upper, rc)


def generate_kmers(sequence: str, k: int, canonical: bool = False) -> Iterator[str]:
    """Yield valid k-mers from a sequence.

    Invalid windows containing characters outside ``A,C,G,T`` are skipped.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    seq = sequence.upper()
    if len(seq) < k:
        return

    for idx in range(0, len(seq) - k + 1):
        kmer = seq[idx : idx + k]
        if any(base not in _DNA_ALPHABET for base in kmer):
            continue
        yield canonical_kmer(kmer) if canonical else kmer


def load_kmers_from_fasta(
    path: str | Path,
    k: int,
    canonical: bool = False,
    deduplicate: bool = False,
) -> Iterator[str]:
    """Stream k-mers extracted from a FASTA file."""
    seen: set[str] | None = set() if deduplicate else None

    for sequence in iter_fasta_sequences(path=path, uppercase=True):
        for kmer in generate_kmers(sequence=sequence, k=k, canonical=canonical):
            if seen is not None:
                if kmer in seen:
                    continue
                seen.add(kmer)
            yield kmer
