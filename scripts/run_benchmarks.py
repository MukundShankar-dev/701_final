"""Run benchmark sweeps across filter types, k, and target FPR."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking.benchmark_runner import run_and_save
from benchmarking.experiment_config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AMQ benchmark sweeps")
    parser.add_argument("--dataset", required=True, help="Input one-k-mer-per-line file")
    parser.add_argument("--dataset-name", default="input", help="Dataset label")
    parser.add_argument("--output-dir", default="benchmarking/results", help="Result directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per run")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ks = [15, 21, 31]
    fprs = [1e-2, 1e-3, 1e-4]
    filters = ["bloom", "cuckoo", "xor", "learned"]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_runs = 0
    for k in ks:
        for filt in filters:
            for fpr in fprs:
                params: dict[str, object]
                if filt == "bloom":
                    params = {"false_positive_rate": fpr}
                elif filt == "cuckoo":
                    fp_bits = 8 if fpr >= 1e-2 else 12 if fpr >= 1e-3 else 14
                    params = {"fingerprint_bits": fp_bits}
                elif filt == "xor":
                    fp_bits = 8 if fpr >= 1e-2 else 10 if fpr >= 1e-3 else 12
                    params = {"fingerprint_bits": fp_bits, "backend": "auto"}
                else:
                    params = {"backup_false_positive_rate": fpr, "model_threshold": 0.5}

                cfg = ExperimentConfig(
                    dataset_name=args.dataset_name,
                    dataset_path=args.dataset,
                    k=k,
                    canonicalize=False,
                    filter_type=filt,
                    filter_params=params,
                    positive_query_count=10_000,
                    negative_query_count=10_000,
                    random_seed=args.seed,
                    output_directory=str(out_root / f"k{k}" / filt),
                    repetitions=args.repetitions,
                )
                run_and_save(cfg)
                total_runs += 1

    print(f"Completed sweep configs: {total_runs}")


if __name__ == "__main__":
    main()
