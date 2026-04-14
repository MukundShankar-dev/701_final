"""Plotting helpers for benchmark result visualization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results_dataframe(results_dir: str | Path) -> pd.DataFrame:
    """Load benchmark run JSON files recursively into a DataFrame."""
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    rows: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        if path.name.startswith("."):
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        if "run_id" not in payload or "filter_type" not in payload:
            continue

        row: dict[str, Any] = dict(payload)
        fp = row.get("filter_params", {})
        if isinstance(fp, dict):
            target_fpr = (
                fp.get("false_positive_rate")
                or fp.get("backup_false_positive_rate")
                or None
            )
        else:
            target_fpr = None

        extra = row.get("extra_stats", {})
        if target_fpr is None and isinstance(extra, dict):
            target_fpr = extra.get("target_false_positive_rate")

        row["target_fpr"] = target_fpr

        backend_name = ""
        if isinstance(extra, dict):
            params = extra.get("parameters", {})
            if isinstance(params, dict):
                backend_name = str(params.get("backend", ""))
        row["backend"] = backend_name

        rows.append(row)

    if not rows:
        raise ValueError(f"No benchmark run JSON files found in {root}")

    df = pd.DataFrame(rows)
    df["target_fpr"] = pd.to_numeric(df["target_fpr"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    def _label_row(row: pd.Series) -> str:
        base = str(row.get("filter_type", "unknown"))
        backend = str(row.get("backend", ""))
        if base == "xor" and backend.startswith("fallback"):
            return "xor (fallback)"
        return base

    df["plot_filter_label"] = df.apply(_label_row, axis=1)
    return df


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _select_reference_fpr(df: pd.DataFrame, preferred: float = 1e-3) -> float | None:
    vals = [float(v) for v in df["target_fpr"].dropna().unique()]
    if not vals:
        return None
    return min(vals, key=lambda v: abs(math.log10(v) - math.log10(preferred)))


def _slice_by_reference_fpr(df: pd.DataFrame, reference_fpr: float | None) -> pd.DataFrame:
    """Select rows per (filter, k) nearest to a reference target FPR."""
    if reference_fpr is None:
        return df.copy()

    chunks: list[pd.DataFrame] = []
    for (_, _), group in df.groupby(["plot_filter_label", "k"], dropna=False):
        valid = [float(v) for v in group["target_fpr"].dropna().unique()]
        if not valid:
            chunks.append(group)
            continue

        chosen = min(valid, key=lambda v: abs(math.log10(v) - math.log10(reference_fpr)))
        chunks.append(group[np.isclose(group["target_fpr"], chosen, rtol=1e-12, atol=0.0)])

    if not chunks:
        return df.copy()
    return pd.concat(chunks, ignore_index=True)


def plot_false_positive_rate(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    title_suffix: str = "",
) -> Path:
    out = Path(output_dir) / "false_positive_rate_by_filter.png"

    grouped = (
        df.groupby(["plot_filter_label", "k"], dropna=False)["false_positive_rate"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for filt, sub in grouped.groupby("plot_filter_label"):
        ax.plot(sub["k"], sub["false_positive_rate"], marker="o", label=filt)

    ax.set_title(f"Empirical False Positive Rate by Filter and k{title_suffix}")
    ax.set_xlabel("k")
    ax.set_ylabel("False Positive Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()

    _save_fig(out)
    return out


def plot_throughput(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    title_suffix: str = "",
) -> Path:
    out = Path(output_dir) / "throughput_by_filter.png"

    grouped = (
        df.groupby(["plot_filter_label", "k"], dropna=False)["throughput_qps"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for filt, sub in grouped.groupby("plot_filter_label"):
        ax.plot(sub["k"], sub["throughput_qps"], marker="o", label=filt)

    ax.set_title(f"Mean Throughput by Filter and k{title_suffix}")
    ax.set_xlabel("k")
    ax.set_ylabel("Queries / second")
    ax.grid(True, alpha=0.25)
    ax.legend()

    _save_fig(out)
    return out


def plot_memory_per_kmer(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    title_suffix: str = "",
) -> Path:
    out = Path(output_dir) / "memory_per_kmer_by_filter.png"

    grouped = (
        df.groupby(["plot_filter_label", "k"], dropna=False)["memory_per_kmer_bytes"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for filt, sub in grouped.groupby("plot_filter_label"):
        ax.plot(sub["k"], sub["memory_per_kmer_bytes"], marker="o", label=filt)

    ax.set_title(f"Memory per k-mer by Filter and k{title_suffix}")
    ax.set_xlabel("k")
    ax.set_ylabel("Bytes per inserted k-mer")
    ax.grid(True, alpha=0.25)
    ax.legend()

    _save_fig(out)
    return out


def plot_build_time(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    title_suffix: str = "",
) -> Path:
    out = Path(output_dir) / "build_time_by_filter.png"

    grouped = (
        df.groupby(["plot_filter_label", "k"], dropna=False)["build_time_seconds"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for filt, sub in grouped.groupby("plot_filter_label"):
        ax.plot(sub["k"], sub["build_time_seconds"], marker="o", label=filt)

    ax.set_title(f"Build Time by Filter and k{title_suffix}")
    ax.set_xlabel("k")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.25)
    ax.legend()

    _save_fig(out)
    return out


def plot_fpr_sweep(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot achieved FPR versus target FPR for each filter label."""
    out = Path(output_dir) / "fpr_vs_target.png"

    usable = df[df["target_fpr"].notna()].copy()
    if usable.empty:
        # Create an empty figure with a clear message if no target FPR info exists.
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title("Achieved vs Target FPR")
        ax.text(0.5, 0.5, "No target FPR metadata available", ha="center", va="center")
        ax.axis("off")
        _save_fig(out)
        return out

    grouped = (
        usable.groupby(["plot_filter_label", "target_fpr"], dropna=False)["false_positive_rate"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for filt, sub in grouped.groupby("plot_filter_label"):
        ax.plot(sub["target_fpr"], sub["false_positive_rate"], marker="o", label=filt)

    ax.set_title("Achieved FPR vs Target FPR")
    ax.set_xlabel("Target FPR")
    ax.set_ylabel("Empirical FPR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()

    _save_fig(out)
    return out


def generate_default_plots(results_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate a default plot set from benchmark JSON results."""
    df = load_results_dataframe(results_dir)
    reference_fpr = _select_reference_fpr(df, preferred=1e-3)
    cross_k_df = _slice_by_reference_fpr(df, reference_fpr)

    suffix = ""
    if reference_fpr is not None:
        suffix = f" (nearest target FPR per filter/k, ref={reference_fpr:.1e})"

    outputs = [
        plot_false_positive_rate(cross_k_df, output_dir, title_suffix=suffix),
        plot_throughput(cross_k_df, output_dir, title_suffix=suffix),
        plot_memory_per_kmer(cross_k_df, output_dir, title_suffix=suffix),
        plot_build_time(cross_k_df, output_dir, title_suffix=suffix),
        plot_fpr_sweep(df, output_dir),
    ]
    return outputs
