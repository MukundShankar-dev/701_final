"""Best-effort cache behavior proxies for Python/macOS benchmarking.

True hardware cache profiling is not directly exposed in pure Python in a
portable way. These helpers provide timing-based proxies only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from benchmarking.timing import measure_batch_query_time


@dataclass(slots=True)
class CacheProxyMetrics:
    sequential_qps: float
    random_qps: float
    repeated_qps: float
    batch_size: int


def run_cache_proxy_bench(
    batch_contains: Callable[[Sequence[str]], list[bool]],
    queries: Sequence[str],
    *,
    random_seed: int = 0,
    repetitions: int = 3,
) -> CacheProxyMetrics:
    """Compute timing proxies for different query access patterns."""
    if not queries:
        raise ValueError("queries must be non-empty")

    seq = list(queries)
    rng = random.Random(random_seed)
    rnd = list(queries)
    rng.shuffle(rnd)
    repeated = [queries[0]] * len(queries)

    seq_t = measure_batch_query_time(batch_contains, seq, repetitions=repetitions)
    rnd_t = measure_batch_query_time(batch_contains, rnd, repetitions=repetitions)
    rep_t = measure_batch_query_time(batch_contains, repeated, repetitions=repetitions)

    n = len(queries)
    return CacheProxyMetrics(
        sequential_qps=n / max(1e-12, seq_t.mean_seconds),
        random_qps=n / max(1e-12, rnd_t.mean_seconds),
        repeated_qps=n / max(1e-12, rep_t.mean_seconds),
        batch_size=n,
    )
