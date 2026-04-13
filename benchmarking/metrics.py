"""Metric computations for AMQ benchmark outputs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from benchmarking.timing import TimingResult


@dataclass(slots=True)
class QueryMetrics:
    true_positive_rate: float
    false_positive_rate: float
    throughput_qps: float
    average_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float


@dataclass(slots=True)
class BuildMetrics:
    build_time_seconds: float
    memory_usage_bytes: int
    memory_per_inserted_key_bytes: float
    load_factor: float | None = None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = int((len(values) - 1) * q)
    return sorted(values)[idx]


def compute_query_metrics(
    positive_truth_count: int,
    positive_pred_count: int,
    negative_pred_count: int,
    negative_total_count: int,
    timing: TimingResult,
    query_count: int,
) -> QueryMetrics:
    tpr = positive_pred_count / max(1, positive_truth_count)
    fpr = negative_pred_count / max(1, negative_total_count)

    mean_run_s = timing.mean_seconds
    throughput = query_count / max(1e-12, mean_run_s)

    per_run_latency_us = [
        (run_s / max(1, query_count)) * 1e6 for run_s in timing.per_item_seconds
    ]

    return QueryMetrics(
        true_positive_rate=float(tpr),
        false_positive_rate=float(fpr),
        throughput_qps=float(throughput),
        average_latency_us=float(statistics.mean(per_run_latency_us)),
        p50_latency_us=float(_quantile(per_run_latency_us, 0.50)),
        p95_latency_us=float(_quantile(per_run_latency_us, 0.95)),
        p99_latency_us=float(_quantile(per_run_latency_us, 0.99)),
    )


def compute_build_metrics(
    build_time_seconds: float,
    memory_usage_bytes: int,
    inserted_keys: int,
    load_factor: float | None = None,
) -> BuildMetrics:
    return BuildMetrics(
        build_time_seconds=float(build_time_seconds),
        memory_usage_bytes=int(memory_usage_bytes),
        memory_per_inserted_key_bytes=float(memory_usage_bytes / max(1, inserted_keys)),
        load_factor=load_factor,
    )
