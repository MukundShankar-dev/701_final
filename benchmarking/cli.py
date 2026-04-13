"""CLI for running benchmark configurations."""

from __future__ import annotations

import argparse
import json

from benchmarking.benchmark_runner import run_and_save
from benchmarking.experiment_config import ExperimentConfig


def _run_command(args: argparse.Namespace) -> None:
    config = ExperimentConfig.load(args.config)
    results = run_and_save(config)

    print(f"Completed {len(results)} benchmark run(s)")
    for result in results:
        summary = {
            "run_id": result.run_id,
            "filter": result.filter_type,
            "dataset": result.dataset_name,
            "tpr": result.true_positive_rate,
            "fpr": result.false_positive_rate,
            "throughput_qps": result.throughput_qps,
            "memory_per_kmer_bytes": result.memory_per_kmer_bytes,
        }
        print(json.dumps(summary, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMQ benchmarking CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run benchmark from config JSON")
    run.add_argument("--config", required=True, help="Experiment config path")
    run.set_defaults(func=_run_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
