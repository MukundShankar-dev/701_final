"""Synthetic genomic-like data generation utilities."""

from __future__ import annotations

import random
from typing import Iterator

from data.kmers import generate_kmers


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def generate_synthetic_sequence(length: int, gc_bias: float = 0.5, seed: int | None = None) -> str:
    """Generate one synthetic DNA sequence.

    Args:
        length: Sequence length in bases.
        gc_bias: Target probability mass assigned to ``G`` and ``C`` combined.
        seed: Optional seed for deterministic generation.
    """
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    if not 0.0 <= gc_bias <= 1.0:
        raise ValueError(f"gc_bias must be in [0, 1], got {gc_bias}")

    rng = _rng(seed)
    p_gc_each = gc_bias / 2.0
    p_at_each = (1.0 - gc_bias) / 2.0

    bases = ["A", "C", "G", "T"]
    probs = [p_at_each, p_gc_each, p_gc_each, p_at_each]

    return "".join(rng.choices(bases, weights=probs, k=length))


def generate_synthetic_contigs(
    num_contigs: int,
    contig_length: int,
    gc_bias: float = 0.5,
    seed: int | None = None,
    header_prefix: str = "contig",
) -> Iterator[tuple[str, str]]:
    """Yield synthetic FASTA-like ``(header, sequence)`` contigs."""
    if num_contigs <= 0:
        raise ValueError(f"num_contigs must be > 0, got {num_contigs}")

    rng = _rng(seed)
    for idx in range(num_contigs):
        contig_seed = rng.randint(0, 2**31 - 1)
        sequence = generate_synthetic_sequence(
            length=contig_length,
            gc_bias=gc_bias,
            seed=contig_seed,
        )
        yield f"{header_prefix}_{idx}", sequence


def generate_synthetic_kmer_set(
    num_kmers: int,
    k: int,
    gc_bias: float = 0.5,
    seed: int | None = None,
) -> set[str]:
    """Generate a synthetic set of unique k-mers."""
    if num_kmers <= 0:
        raise ValueError(f"num_kmers must be > 0, got {num_kmers}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    rng = _rng(seed)
    out: set[str] = set()
    max_attempts = max(10 * num_kmers, 1000)
    attempts = 0

    while len(out) < num_kmers and attempts < max_attempts:
        attempts += 1
        seq_seed = rng.randint(0, 2**31 - 1)
        sequence = generate_synthetic_sequence(length=k, gc_bias=gc_bias, seed=seq_seed)
        out.add(sequence)

    if len(out) < num_kmers:
        raise RuntimeError(
            f"Unable to generate requested unique k-mers: requested={num_kmers}, got={len(out)}"
        )

    return out


def mutate_kmer(kmer: str, mutation_rate: float, rng: random.Random) -> str:
    """Apply point mutations to a k-mer according to mutation rate."""
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")

    bases = ["A", "C", "G", "T"]
    chars = list(kmer.upper())

    for i, base in enumerate(chars):
        if rng.random() >= mutation_rate:
            continue
        choices = [b for b in bases if b != base]
        chars[i] = rng.choice(choices)

    mutated = "".join(chars)
    if mutated == kmer and mutation_rate > 0:
        i = rng.randrange(len(chars))
        choices = [b for b in bases if b != chars[i]]
        chars[i] = rng.choice(choices)
        mutated = "".join(chars)
    return mutated


def generate_negative_kmers(
    positive_kmers: set[str],
    count: int,
    mutation_rate: float = 0.2,
    seed: int | None = None,
) -> list[str]:
    """Generate negative k-mers not present in the provided positive set.

    Strategy: mutate sampled positives first (hard negatives), then fall back to
    random synthetic generation if needed.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if not positive_kmers and count > 0:
        raise ValueError("positive_kmers must be non-empty when count > 0")

    rng = _rng(seed)
    positives = list(positive_kmers)
    k = len(positives[0]) if positives else 0

    negatives: set[str] = set()
    max_attempts = max(20 * count, 1000)
    attempts = 0

    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        candidate = mutate_kmer(rng.choice(positives), mutation_rate=mutation_rate, rng=rng)
        if candidate not in positive_kmers:
            negatives.add(candidate)

    while len(negatives) < count:
        candidate = generate_synthetic_sequence(length=k, gc_bias=0.5, seed=rng.randint(0, 2**31 - 1))
        if candidate not in positive_kmers:
            negatives.add(candidate)

    return list(negatives)


def kmers_from_synthetic_contigs(
    num_contigs: int,
    contig_length: int,
    k: int,
    gc_bias: float = 0.5,
    canonical: bool = False,
    deduplicate: bool = True,
    seed: int | None = None,
) -> Iterator[str]:
    """Generate k-mers directly from synthetic contigs."""
    seen: set[str] | None = set() if deduplicate else None

    for _, sequence in generate_synthetic_contigs(
        num_contigs=num_contigs,
        contig_length=contig_length,
        gc_bias=gc_bias,
        seed=seed,
    ):
        for kmer in generate_kmers(sequence=sequence, k=k, canonical=canonical):
            if seen is not None:
                if kmer in seen:
                    continue
                seen.add(kmer)
            yield kmer
