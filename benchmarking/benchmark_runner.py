"""Unified benchmark orchestration across AMQ filter families."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmarking.cache_proxy import run_cache_proxy_bench
from benchmarking.experiment_config import ExperimentConfig
from benchmarking.metrics import compute_build_metrics, compute_query_metrics
from benchmarking.query_sets import sample_query_sets
from benchmarking.results import BenchmarkRunResult, append_results_csv, save_run_result_json
from benchmarking.timing import TimingResult, measure_batch_query_time
from bloom_filters.bloom_filter import BloomFilter
from cuckoo_filters.cuckoo_filter import CuckooFilter
from data.io_utils import load_kmers
from learned_filters.learned_filter import LearnedFilter
from xor_filters.xor_filter import XorFilter


def _param_signature(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


def _make_filter(config: ExperimentConfig) -> Any:
    params = dict(config.filter_params)
    filter_type = config.filter_type.lower()

    if filter_type == "bloom":
        return BloomFilter(
            expected_items=int(params.get("expected_items", 1)),
            false_positive_rate=float(params.get("false_positive_rate", 1e-3)),
            hash_seed=int(params.get("hash_seed", config.random_seed)),
        )

    if filter_type == "cuckoo":
        return CuckooFilter(
            capacity=int(params.get("capacity", 1)),
            bucket_size=int(params.get("bucket_size", 4)),
            fingerprint_bits=int(params.get("fingerprint_bits", 12)),
            max_relocations=int(params.get("max_relocations", 500)),
            random_seed=int(params.get("random_seed", config.random_seed)),
        )

    if filter_type == "xor":
        return XorFilter(
            fingerprint_bits=int(params.get("fingerprint_bits", 8)),
            backend=str(params.get("backend", "auto")),
            hash_seed=int(params.get("hash_seed", config.random_seed)),
            size_factor=float(params.get("size_factor", 1.23)),
            max_retries=int(params.get("max_retries", 64)),
        )

    if filter_type == "learned":
        return LearnedFilter(
            k=config.k,
            model_threshold=float(params.get("model_threshold", 0.5)),
            backup_false_positive_rate=float(params.get("backup_false_positive_rate", 1e-3)),
            random_seed=int(params.get("random_seed", config.random_seed)),
            model_backend=str(params.get("model_backend", "ngram_sgd")),
            ngram_features=int(params.get("ngram_features", 4096)),
            ngram_range=tuple(params.get("ngram_range", (3, 5))),
        )

    raise ValueError(f"Unsupported filter type: {config.filter_type}")


def _prepare_keys(config: ExperimentConfig) -> list[str]:
    keys = list(load_kmers(config.dataset_path, deduplicate=True))
    if not keys:
        raise ValueError(f"No keys loaded from dataset: {config.dataset_path}")
    return [k.upper() for k in keys if len(k) == config.k]


def run_single_benchmark(config: ExperimentConfig, *, run_index: int = 0) -> BenchmarkRunResult:
    keys = _prepare_keys(config)

    # Fill default size parameters from dataset cardinality for fair comparisons.
    if config.filter_type.lower() == "bloom":
        config.filter_params.setdefault("expected_items", len(keys))
    if config.filter_type.lower() == "cuckoo":
        config.filter_params.setdefault("capacity", len(keys))

    filt = _make_filter(config)

    t0 = time.perf_counter()
    filt.build(keys)
    build_time = time.perf_counter() - t0

    queries = sample_query_sets(
        inserted_keys=keys,
        positive_count=config.positive_query_count,
        negative_count=config.negative_query_count,
        random_seed=config.random_seed + run_index,
    )

    pos_timing = measure_batch_query_time(
        filt.batch_contains,
        queries.positives,
        repetitions=max(1, config.repetitions),
    )
    neg_timing = measure_batch_query_time(
        filt.batch_contains,
        queries.negatives,
        repetitions=max(1, config.repetitions),
    )

    pos_preds = filt.batch_contains(queries.positives)
    neg_preds = filt.batch_contains(queries.negatives)

    positive_pred_count = sum(1 for v in pos_preds if v)
    negative_pred_count = sum(1 for v in neg_preds if v)

    total_query_count = len(queries.positives) + len(queries.negatives)
    merged_timing = TimingResult(
        total_seconds=pos_timing.total_seconds + neg_timing.total_seconds,
        mean_seconds=pos_timing.mean_seconds + neg_timing.mean_seconds,
        per_item_seconds=[
            p + n for p, n in zip(pos_timing.per_item_seconds, neg_timing.per_item_seconds, strict=True)
        ],
    )

    query_metrics = compute_query_metrics(
        positive_truth_count=len(queries.positives),
        positive_pred_count=positive_pred_count,
        negative_pred_count=negative_pred_count,
        negative_total_count=len(queries.negatives),
        timing=merged_timing,
        query_count=total_query_count,
    )

    stats = filt.stats()
    build_metrics = compute_build_metrics(
        build_time_seconds=build_time,
        memory_usage_bytes=filt.memory_usage_bytes(),
        inserted_keys=len(keys),
        load_factor=float(stats.get("load_factor")) if "load_factor" in stats else None,
    )

    cache_metrics = run_cache_proxy_bench(
        filt.batch_contains,
        queries.negatives[: min(len(queries.negatives), 5000)],
        random_seed=config.random_seed + run_index,
    )

    param_sig = _param_signature(dict(config.filter_params))
    result = BenchmarkRunResult(
        run_id=f"{config.filter_type}_{config.dataset_name}_k{config.k}_{param_sig}_run{run_index}",
        dataset_name=config.dataset_name,
        filter_type=config.filter_type,
        filter_params=dict(config.filter_params),
        k=config.k,
        inserted_keys=len(keys),
        build_time_seconds=build_metrics.build_time_seconds,
        memory_usage_bytes=build_metrics.memory_usage_bytes,
        memory_per_kmer_bytes=build_metrics.memory_per_inserted_key_bytes,
        true_positive_rate=query_metrics.true_positive_rate,
        false_positive_rate=query_metrics.false_positive_rate,
        throughput_qps=query_metrics.throughput_qps,
        avg_latency_us=query_metrics.average_latency_us,
        p50_latency_us=query_metrics.p50_latency_us,
        p95_latency_us=query_metrics.p95_latency_us,
        p99_latency_us=query_metrics.p99_latency_us,
        cache_proxy={
            "sequential_qps": cache_metrics.sequential_qps,
            "random_qps": cache_metrics.random_qps,
            "repeated_qps": cache_metrics.repeated_qps,
            "batch_size": cache_metrics.batch_size,
        },
        extra_stats=stats,
    )
    return result


def run_and_save(config: ExperimentConfig) -> list[BenchmarkRunResult]:
    """Run benchmark repetitions and persist JSON + aggregate CSV outputs."""
    out_dir = Path(config.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkRunResult] = []
    reps = max(1, config.repetitions)
    for run_index in range(reps):
        result = run_single_benchmark(config, run_index=run_index)
        json_path = out_dir / f"{result.run_id}.json"
        save_run_result_json(result, json_path)
        results.append(result)

    csv_path = out_dir / "aggregate_results.csv"
    append_results_csv(results, csv_path)

    return results
