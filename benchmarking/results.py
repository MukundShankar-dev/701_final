"""Result dataclasses and serializers for benchmark runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.io_utils import append_csv_rows, save_json


@dataclass(slots=True)
class BenchmarkRunResult:
    run_id: str
    dataset_name: str
    filter_type: str
    filter_params: dict[str, Any]
    k: int
    inserted_keys: int
    build_time_seconds: float
    memory_usage_bytes: int
    memory_per_kmer_bytes: float
    true_positive_rate: float
    false_positive_rate: float
    throughput_qps: float
    avg_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    cache_proxy: dict[str, Any] = field(default_factory=dict)
    extra_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["filter_params"] = dict(self.filter_params)
        out["cache_proxy"] = dict(self.cache_proxy)
        out["extra_stats"] = dict(self.extra_stats)
        return out


def save_run_result_json(result: BenchmarkRunResult, output_path: str | Path) -> None:
    save_json(output_path, result.to_dict())


def append_results_csv(results: list[BenchmarkRunResult], csv_path: str | Path) -> int:
    if not results:
        return 0

    rows = [r.to_dict() for r in results]
    out_path = Path(csv_path)
    if out_path.exists():
        existing_run_ids: set[str] = set()
        with out_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                run_id = row.get("run_id")
                if run_id:
                    existing_run_ids.add(run_id)
        rows = [row for row in rows if row.get("run_id") not in existing_run_ids]
        if not rows:
            return 0

    fieldnames = list(rows[0].keys())
    return append_csv_rows(out_path, rows, fieldnames=fieldnames)
