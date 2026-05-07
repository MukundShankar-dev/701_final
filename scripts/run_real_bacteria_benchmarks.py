"""Run resumable benchmark sweeps for the real bacterial k-mer datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_GLOB = "data/datasets/real/diverse_bacteria_4/k*/manifest.tsv"
DEFAULT_OUTPUT_ROOT = "benchmarking/final_results/real/diverse_bacteria_4"
DEFAULT_FILTERS = "bloom,cuckoo,xor,learned"
DEFAULT_FPRS = "1e-2,1e-3,1e-4"


@dataclass(frozen=True, slots=True)
class RealDataset:
    dataset_id: str
    accession: str
    organism: str
    k: int
    unique_kmers: int
    fasta_path: Path
    kmers_path: Path
    counts_path: Path
    jellyfish_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    dataset: RealDataset
    filter_type: str
    target_fpr: float


@dataclass(frozen=True, slots=True)
class CompletionStatus:
    json_runs: int
    aggregate_runs: int
    complete: bool


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in _parse_csv_list(raw)]


def _parse_int_set(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    return {int(item) for item in _parse_csv_list(raw)}


def _parse_str_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    return set(_parse_csv_list(raw))


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str, log_handle: TextIO | None = None) -> None:
    line = f"[{_timestamp()}] {message}"
    print(line, flush=True)
    if log_handle is not None:
        log_handle.write(f"{line}\n")
        log_handle.flush()


def _target_fpr_from_result(payload: dict[str, object], filter_type: str) -> float | None:
    params = payload.get("filter_params")
    if not isinstance(params, dict):
        return None
    raw = params.get("backup_false_positive_rate") if filter_type == "learned" else params.get("false_positive_rate")
    if raw is None:
        return None
    return float(raw)


def _same_fpr(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-12, abs(right) * 1e-9)


def _task_output_dir(task: BenchmarkTask, output_root: Path) -> Path:
    return output_root / task.dataset.dataset_id / f"k{task.dataset.k}" / task.filter_type


def _matching_json_run_ids(task: BenchmarkTask, output_root: Path) -> set[str]:
    out_dir = _task_output_dir(task, output_root)
    run_ids: set[str] = set()
    for path in out_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("dataset_name") != task.dataset.dataset_id:
            continue
        if str(payload.get("filter_type", "")).lower() != task.filter_type:
            continue
        if int(payload.get("k", -1)) != task.dataset.k:
            continue
        target_fpr = _target_fpr_from_result(payload, task.filter_type)
        if target_fpr is None or not _same_fpr(target_fpr, task.target_fpr):
            continue

        run_id = payload.get("run_id")
        if isinstance(run_id, str):
            run_ids.add(run_id)
    return run_ids


def _aggregate_run_ids(out_dir: Path) -> set[str]:
    csv_path = out_dir / "aggregate_results.csv"
    if not csv_path.exists():
        return set()

    run_ids: set[str] = set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                run_id = row.get("run_id")
                if run_id:
                    run_ids.add(run_id)
    except OSError:
        return set()
    return run_ids


def _completion_status(task: BenchmarkTask, output_root: Path, repetitions: int) -> CompletionStatus:
    out_dir = _task_output_dir(task, output_root)
    json_ids = _matching_json_run_ids(task, output_root)
    aggregate_ids = _aggregate_run_ids(out_dir)
    aggregate_matches = len(json_ids & aggregate_ids)
    return CompletionStatus(
        json_runs=len(json_ids),
        aggregate_runs=aggregate_matches,
        complete=len(json_ids) >= repetitions and aggregate_matches >= repetitions,
    )


def _load_datasets(manifest_glob: str, *, k_values: set[int] | None, dataset_ids: set[str] | None) -> list[RealDataset]:
    datasets: list[RealDataset] = []
    for manifest in sorted(REPO_ROOT.glob(manifest_glob)):
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                dataset = RealDataset(
                    dataset_id=row["dataset_id"],
                    accession=row["accession"],
                    organism=row["organism"],
                    k=int(row["k"]),
                    unique_kmers=int(row["unique_kmers"]),
                    fasta_path=Path(row["fasta_path"]),
                    kmers_path=Path(row["kmers_path"]),
                    counts_path=Path(row["counts_path"]),
                    jellyfish_path=Path(row["jellyfish_path"]),
                )
                if k_values is not None and dataset.k not in k_values:
                    continue
                if dataset_ids is not None and dataset.dataset_id not in dataset_ids:
                    continue
                if not (REPO_ROOT / dataset.kmers_path).exists():
                    raise FileNotFoundError(f"k-mer file from manifest does not exist: {dataset.kmers_path}")
                datasets.append(dataset)

    if not datasets:
        raise ValueError(f"No datasets found for manifest glob: {manifest_glob}")
    return datasets


def _build_command(task: BenchmarkTask, args: argparse.Namespace, output_root: Path) -> list[str]:
    return [
        args.python,
        str(REPO_ROOT / args.benchmark_script),
        "--dataset",
        str(task.dataset.kmers_path),
        "--dataset-name",
        task.dataset.dataset_id,
        "--output-dir",
        str(output_root / task.dataset.dataset_id),
        "--seed",
        str(args.seed),
        "--repetitions",
        str(args.repetitions),
        "--k-values",
        str(task.dataset.k),
        "--filters",
        task.filter_type,
        "--fprs",
        f"{task.target_fpr:g}",
    ]


def _run_task(task: BenchmarkTask, args: argparse.Namespace, output_root: Path, log_handle: TextIO | None) -> bool:
    command = _build_command(task, args, output_root)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not current_pythonpath else f"{REPO_ROOT}{os.pathsep}{current_pythonpath}"
    env.setdefault("PYTHONUNBUFFERED", "1")

    _log("COMMAND " + shlex.join(command), log_handle)
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if line:
            _log(f"  {line}", log_handle)

    return_code = process.wait()
    elapsed = time.perf_counter() - start
    if return_code != 0:
        _log(f"FAILED after {elapsed:.1f}s with exit code {return_code}", log_handle)
        return False

    status = _completion_status(task, output_root, args.repetitions)
    out_dir = _task_output_dir(task, output_root)
    _log(
        "DONE "
        f"after {elapsed:.1f}s; json_runs={status.json_runs}; "
        f"aggregate_runs={status.aggregate_runs}; output={out_dir}",
        log_handle,
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable real bacterial genome benchmark sweeps")
    parser.add_argument("--manifest-glob", default=DEFAULT_MANIFEST_GLOB, help="Glob for TSV manifests")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root for benchmark outputs")
    parser.add_argument("--seed", type=int, default=7, help="Random seed passed to benchmark runner")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per filter/FPR task")
    parser.add_argument("--fprs", default=DEFAULT_FPRS, help="Comma-separated target FPRs")
    parser.add_argument("--filters", default=DEFAULT_FILTERS, help="Comma-separated filter families")
    parser.add_argument("--k-values", default=None, help="Optional comma-separated k values to include")
    parser.add_argument("--datasets", default=None, help="Optional comma-separated manifest dataset_id values to include")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of tasks to run")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running benchmarks")
    parser.add_argument("--rerun-completed", action="store_true", help="Run tasks even when matching outputs already exist")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed benchmark command")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch run_benchmarks.py")
    parser.add_argument("--benchmark-script", default="scripts/run_benchmarks.py", help="Benchmark runner script path")
    parser.add_argument("--log-file", default=None, help="Progress log path; defaults under --output-root")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else output_root / "run_real_bacteria_benchmarks.log"
    if not log_path.is_absolute():
        log_path = REPO_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    filters = [item.lower() for item in _parse_csv_list(args.filters)]
    fprs = _parse_float_list(args.fprs)
    datasets = _load_datasets(
        args.manifest_glob,
        k_values=_parse_int_set(args.k_values),
        dataset_ids=_parse_str_set(args.datasets),
    )

    tasks = [BenchmarkTask(dataset=dataset, filter_type=filt, target_fpr=fpr) for dataset in datasets for filt in filters for fpr in fprs]
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]

    with log_path.open("a", encoding="utf-8") as log_handle:
        _log(
            f"Planned tasks={len(tasks)} datasets={len(datasets)} filters={filters} "
            f"fprs={[f'{fpr:g}' for fpr in fprs]} repetitions={args.repetitions}",
            log_handle,
        )
        _log(f"Output root: {output_root}", log_handle)
        _log(f"Progress log: {log_path}", log_handle)

        skipped = 0
        succeeded = 0
        failed = 0
        for index, task in enumerate(tasks, start=1):
            status = _completion_status(task, output_root, args.repetitions)
            label = (
                f"[{index}/{len(tasks)}] dataset={task.dataset.dataset_id} "
                f"organism={task.dataset.organism!r} k={task.dataset.k} "
                f"filter={task.filter_type} fpr={task.target_fpr:g}"
            )
            if status.complete and not args.rerun_completed:
                skipped += 1
                _log(f"SKIP {label}; json_runs={status.json_runs}; aggregate_runs={status.aggregate_runs}", log_handle)
                continue

            _log(f"START {label}; kmers={task.dataset.unique_kmers:,}", log_handle)
            if args.dry_run:
                _log("DRY RUN " + shlex.join(_build_command(task, args, output_root)), log_handle)
                continue

            if _run_task(task, args, output_root, log_handle):
                succeeded += 1
            else:
                failed += 1
                if args.stop_on_error:
                    break

        _log(f"SUMMARY succeeded={succeeded} skipped={skipped} failed={failed} dry_run={args.dry_run}", log_handle)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
