"""Run benchmark sweeps across filter types, k, and target FPR."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Sequence

from benchmarking.benchmark_runner import run_and_save
from benchmarking.experiment_config import ExperimentConfig


def _parse_csv_list(raw: str, cast: type[int] | type[float] | type[str]) -> list[int] | list[float] | list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return [cast(item) for item in items]


def _infer_k_from_dataset(dataset_path: str) -> int:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            key = raw.strip()
            if key:
                return len(key)
    raise ValueError(f"Dataset file is empty: {path}")


def _cuckoo_fingerprint_bits_for_target_fpr(target_fpr: float, *, bucket_size: int = 4) -> int:
    """Return fingerprint bits using ``FPR ~= (2*b)/2^f`` for Cuckoo filters."""
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    if bucket_size <= 0:
        raise ValueError("bucket_size must be > 0")
    return max(2, math.ceil(math.log2((2.0 * bucket_size) / target_fpr)))


def _xor_fingerprint_bits_for_target_fpr(target_fpr: float) -> int:
    """Pick fingerprint bits so ``2^-f`` is at or below the target FPR."""
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return max(2, math.ceil(-math.log2(target_fpr)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AMQ benchmark sweeps")
    parser.add_argument("--dataset", required=True, help="Input one-k-mer-per-line file")
    parser.add_argument("--dataset-name", default="input", help="Dataset label")
    parser.add_argument("--output-dir", default="benchmarking/results", help="Result directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing new benchmark results",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per run")
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated k values (e.g., 15,21,31). Defaults to inferred dataset k.",
    )
    parser.add_argument(
        "--fprs",
        default="1e-2,1e-3,1e-4",
        help="Comma-separated target FPR values.",
    )
    parser.add_argument(
        "--filters",
        default="bloom,cuckoo,xor,learned",
        help="Comma-separated filter families to run.",
    )
    return parser


def _prepare_output_root(raw_output_dir: str, *, overwrite: bool) -> Path:
    if not raw_output_dir.strip():
        raise ValueError("--output-dir cannot be empty; set OUT or pass an explicit path")

    out_root = Path(raw_output_dir)
    resolved = out_root.resolve()
    cwd = Path.cwd().resolve()

    if overwrite:
        forbidden = {Path("/").resolve(), cwd, cwd.parent, Path.home().resolve()}
        if resolved in forbidden:
            raise ValueError(f"Refusing to overwrite unsafe output directory: {resolved}")
        if out_root.exists():
            shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def main() -> None:
    args = build_parser().parse_args()

    if args.k_values is None:
        ks: Sequence[int] = [_infer_k_from_dataset(args.dataset)]
    else:
        ks = _parse_csv_list(args.k_values, int)

    fprs = _parse_csv_list(args.fprs, float)
    filters = _parse_csv_list(args.filters, str)

    if not fprs:
        raise ValueError("--fprs must contain at least one value")
    for fpr in fprs:
        if not 0.0 < float(fpr) < 1.0:
            raise ValueError(f"FPR values must be in (0, 1), got {fpr}")

    out_root = _prepare_output_root(args.output_dir, overwrite=args.overwrite)

    total_runs = 0
    for k in ks:
        for filt in filters:
            for fpr in fprs:
                params: dict[str, object]
                if filt == "bloom":
                    params = {"false_positive_rate": fpr}
                elif filt == "cuckoo":
                    fp_bits = _cuckoo_fingerprint_bits_for_target_fpr(float(fpr), bucket_size=4)
                    params = {
                        "false_positive_rate": fpr,
                        "fingerprint_bits": fp_bits,
                        "bucket_size": 4,
                    }
                elif filt == "xor":
                    fp_bits = _xor_fingerprint_bits_for_target_fpr(float(fpr))
                    params = {
                        "false_positive_rate": fpr,
                        "fingerprint_bits": fp_bits,
                        "backend": "auto",
                        "hash_seed": args.seed,
                    }
                else:
                    params = {
                        "backup_false_positive_rate": fpr,
                        "model_threshold": 0.5,
                        "model_backend": "ngram_sgd",
                        "ngram_features": 4096,
                        "ngram_range": (3, 5),
                    }

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
