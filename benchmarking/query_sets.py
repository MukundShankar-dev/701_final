"""Query set generation utilities for AMQ benchmarking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from data.synthetic import generate_negative_kmers


@dataclass(slots=True)
class QuerySets:
    positives: list[str]
    negatives: list[str]


@dataclass(slots=True)
class LearnedSplits:
    train_kmers: list[str]
    train_labels: list[int]
    val_kmers: list[str]
    val_labels: list[int]
    test_kmers: list[str]
    test_labels: list[int]


def sample_query_sets(
    inserted_keys: Sequence[str],
    *,
    positive_count: int,
    negative_count: int,
    random_seed: int = 0,
) -> QuerySets:
    """Sample positive and negative queries for benchmark evaluation."""
    if not inserted_keys:
        raise ValueError("inserted_keys must be non-empty")
    if positive_count < 0 or negative_count < 0:
        raise ValueError("query counts must be >= 0")

    rng = random.Random(random_seed)
    keys = list(inserted_keys)

    if positive_count <= len(keys):
        positives = rng.sample(keys, positive_count)
    else:
        positives = [rng.choice(keys) for _ in range(positive_count)]

    negatives = generate_negative_kmers(
        positive_kmers=set(keys),
        count=negative_count,
        mutation_rate=0.2,
        seed=random_seed + 17,
    )
    return QuerySets(positives=positives, negatives=negatives)


def make_learned_splits(
    kmers: Sequence[str],
    labels: Sequence[int],
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> LearnedSplits:
    """Build train/validation/test splits from labeled arrays."""
    if len(kmers) != len(labels):
        raise ValueError("kmers and labels must have same length")

    n = len(kmers)
    train_end = int(train_fraction * n)
    val_end = int((train_fraction + val_fraction) * n)

    return LearnedSplits(
        train_kmers=list(kmers[:train_end]),
        train_labels=list(labels[:train_end]),
        val_kmers=list(kmers[train_end:val_end]),
        val_labels=list(labels[train_end:val_end]),
        test_kmers=list(kmers[val_end:]),
        test_labels=list(labels[val_end:]),
    )
