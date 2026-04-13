"""Timing utilities for benchmark runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class TimingResult:
    total_seconds: float
    mean_seconds: float
    per_item_seconds: list[float]


def time_callable(fn: Callable[[], T], *, repetitions: int = 1, warmup: int = 0) -> tuple[T, TimingResult]:
    """Measure callable runtime with optional warmup and repetitions."""
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        fn()

    outputs: list[T] = []
    per_rep: list[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        outputs.append(fn())
        per_rep.append(time.perf_counter() - t0)

    total = sum(per_rep)
    return outputs[-1], TimingResult(total_seconds=total, mean_seconds=total / repetitions, per_item_seconds=per_rep)


def measure_batch_query_time(
    batch_fn: Callable[[Sequence[str]], list[bool]],
    queries: Sequence[str],
    *,
    repetitions: int = 3,
    warmup: int = 1,
) -> TimingResult:
    """Time batch query execution across repeated runs."""
    _, out = time_callable(lambda: batch_fn(queries), repetitions=repetitions, warmup=warmup)
    return out
