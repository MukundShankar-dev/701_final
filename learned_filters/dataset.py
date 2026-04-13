"""Dataset utilities for learned-filter training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from data.synthetic import generate_negative_kmers


@dataclass(slots=True)
class LabeledKmerDataset:
    kmers: list[str]
    labels: list[int]


def build_training_dataset(
    positive_kmers: Sequence[str],
    *,
    negative_count: int | None = None,
    negative_mutation_rate: float = 0.2,
    random_seed: int = 0,
) -> LabeledKmerDataset:
    """Create labeled dataset from positive set and generated negatives."""
    positives = list(dict.fromkeys(k.upper() for k in positive_kmers))
    if not positives:
        raise ValueError("positive_kmers must be non-empty")

    neg_count = negative_count if negative_count is not None else len(positives)
    negatives = generate_negative_kmers(
        positive_kmers=set(positives),
        count=neg_count,
        mutation_rate=negative_mutation_rate,
        seed=random_seed,
    )

    kmers = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    rng = random.Random(random_seed)
    idx = list(range(len(kmers)))
    rng.shuffle(idx)

    return LabeledKmerDataset(
        kmers=[kmers[i] for i in idx],
        labels=[labels[i] for i in idx],
    )


def split_dataset(
    dataset: LabeledKmerDataset,
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> tuple[LabeledKmerDataset, LabeledKmerDataset, LabeledKmerDataset]:
    """Split dataset into train/validation/test partitions."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    n = len(dataset.kmers)
    train_end = int(train_fraction * n)
    val_end = int((train_fraction + val_fraction) * n)

    def _slice(start: int, end: int) -> LabeledKmerDataset:
        return LabeledKmerDataset(
            kmers=dataset.kmers[start:end],
            labels=dataset.labels[start:end],
        )

    return _slice(0, train_end), _slice(train_end, val_end), _slice(val_end, n)
