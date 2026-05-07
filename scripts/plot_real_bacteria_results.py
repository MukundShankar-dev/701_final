"""Generate aggregate SVG plots for real bacterial-genome benchmark results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FILTER_ORDER = ["bloom", "cuckoo", "xor", "learned"]
FILTER_COLORS = {
    "bloom": "#1f77b4",
    "cuckoo": "#ff7f0e",
    "xor": "#d62728",
    "learned": "#2ca02c",
}


@dataclass(frozen=True, slots=True)
class RunRow:
    dataset_name: str
    genome_id: str
    filter_label: str
    k: int
    target_fpr: float
    inserted_keys: int
    build_time_seconds: float
    build_time_per_1m_kmers: float
    throughput_qps: float
    memory_per_kmer_bytes: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class CrossKSummaryRow:
    filter_label: str
    k: int
    build_time_per_1m_kmers_mean: float
    build_time_per_1m_kmers_std: float
    throughput_qps_mean: float
    throughput_qps_std: float
    memory_per_kmer_bytes_mean: float
    memory_per_kmer_bytes_std: float
    n_genomes: int


@dataclass(frozen=True, slots=True)
class FprSummaryRow:
    filter_label: str
    target_fpr: float
    false_positive_rate_mean: float
    false_positive_rate_std: float
    n_observations: int


@dataclass(frozen=True, slots=True)
class PlotArea:
    x: float
    y: float
    width: float
    height: float


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return statistics.fmean(vals)


def _std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def _target_fpr(payload: dict[str, Any]) -> float | None:
    params = payload.get("filter_params")
    if isinstance(params, dict):
        raw = params.get("false_positive_rate")
        if raw is None:
            raw = params.get("backup_false_positive_rate")
        if raw is not None:
            return float(raw)

    extra = payload.get("extra_stats")
    if isinstance(extra, dict):
        raw = extra.get("target_false_positive_rate")
        if raw is not None:
            return float(raw)
    return None


def _filter_label(payload: dict[str, Any]) -> str:
    filt = str(payload.get("filter_type", "unknown"))
    extra = payload.get("extra_stats")
    if filt == "xor" and isinstance(extra, dict):
        params = extra.get("parameters")
        if isinstance(params, dict) and str(params.get("backend", "")).startswith("fallback"):
            return "xor (fallback)"
    return filt


def load_run_rows(results_dir: str | Path) -> list[RunRow]:
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    rows: list[RunRow] = []
    for path in root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict) or "run_id" not in payload:
            continue

        target_fpr = _target_fpr(payload)
        if target_fpr is None:
            continue

        dataset_name = str(payload["dataset_name"])
        inserted_keys = int(payload["inserted_keys"])
        if inserted_keys <= 0:
            continue

        build_time_seconds = float(payload["build_time_seconds"])
        rows.append(
            RunRow(
                dataset_name=dataset_name,
                genome_id=re.sub(r"_k\d+$", "", dataset_name),
                filter_label=_filter_label(payload),
                k=int(payload["k"]),
                target_fpr=float(target_fpr),
                inserted_keys=inserted_keys,
                build_time_seconds=build_time_seconds,
                build_time_per_1m_kmers=build_time_seconds / inserted_keys * 1_000_000,
                throughput_qps=float(payload["throughput_qps"]),
                memory_per_kmer_bytes=float(payload["memory_per_kmer_bytes"]),
                false_positive_rate=float(payload["false_positive_rate"]),
            )
        )

    if not rows:
        raise ValueError(f"No benchmark run JSON files found in {root}")
    return rows


def _closest_target(rows: list[RunRow], reference_fpr: float) -> float:
    targets = sorted({row.target_fpr for row in rows})
    return min(targets, key=lambda v: abs(math.log10(v) - math.log10(reference_fpr)))


def summarize_cross_k(rows: list[RunRow], reference_fpr: float) -> list[CrossKSummaryRow]:
    by_genome_filter_k: dict[tuple[str, str, int], list[RunRow]] = defaultdict(list)
    for row in rows:
        by_genome_filter_k[(row.genome_id, row.filter_label, row.k)].append(row)

    per_genome: list[dict[str, float | str | int]] = []
    for (genome_id, filter_label, k), group in by_genome_filter_k.items():
        chosen = _closest_target(group, reference_fpr)
        selected = [row for row in group if math.isclose(row.target_fpr, chosen, rel_tol=1e-12, abs_tol=0.0)]
        per_genome.append({
            "genome_id": genome_id,
            "filter_label": filter_label,
            "k": k,
            "build_time_per_1m_kmers": _mean(row.build_time_per_1m_kmers for row in selected),
            "throughput_qps": _mean(row.throughput_qps for row in selected),
            "memory_per_kmer_bytes": _mean(row.memory_per_kmer_bytes for row in selected),
        })

    by_filter_k: dict[tuple[str, int], list[dict[str, float | str | int]]] = defaultdict(list)
    for row in per_genome:
        by_filter_k[(str(row["filter_label"]), int(row["k"]))].append(row)

    summary: list[CrossKSummaryRow] = []
    for (filter_label, k), group in sorted(by_filter_k.items(), key=lambda item: (_filter_sort_key(item[0][0]), item[0][1])):
        build_vals = [float(row["build_time_per_1m_kmers"]) for row in group]
        throughput_vals = [float(row["throughput_qps"]) for row in group]
        memory_vals = [float(row["memory_per_kmer_bytes"]) for row in group]
        summary.append(
            CrossKSummaryRow(
                filter_label=filter_label,
                k=k,
                build_time_per_1m_kmers_mean=_mean(build_vals),
                build_time_per_1m_kmers_std=_std(build_vals),
                throughput_qps_mean=_mean(throughput_vals),
                throughput_qps_std=_std(throughput_vals),
                memory_per_kmer_bytes_mean=_mean(memory_vals),
                memory_per_kmer_bytes_std=_std(memory_vals),
                n_genomes=len({str(row["genome_id"]) for row in group}),
            )
        )
    return summary


def summarize_fpr_sweep(rows: list[RunRow]) -> list[FprSummaryRow]:
    by_genome_k_filter_target: dict[tuple[str, int, str, float], list[RunRow]] = defaultdict(list)
    for row in rows:
        by_genome_k_filter_target[(row.genome_id, row.k, row.filter_label, row.target_fpr)].append(row)

    per_config: list[dict[str, float | str | int]] = []
    for (_, k, filter_label, target_fpr), group in by_genome_k_filter_target.items():
        per_config.append({
            "k": k,
            "filter_label": filter_label,
            "target_fpr": target_fpr,
            "false_positive_rate": _mean(row.false_positive_rate for row in group),
        })

    by_filter_target: dict[tuple[str, float], list[dict[str, float | str | int]]] = defaultdict(list)
    for row in per_config:
        by_filter_target[(str(row["filter_label"]), float(row["target_fpr"]))].append(row)

    summary: list[FprSummaryRow] = []
    for (filter_label, target_fpr), group in sorted(by_filter_target.items(), key=lambda item: (_filter_sort_key(item[0][0]), item[0][1])):
        vals = [float(row["false_positive_rate"]) for row in group]
        summary.append(
            FprSummaryRow(
                filter_label=filter_label,
                target_fpr=target_fpr,
                false_positive_rate_mean=_mean(vals),
                false_positive_rate_std=_std(vals),
                n_observations=len(vals),
            )
        )
    return summary


def _filter_sort_key(label: str) -> int:
    return FILTER_ORDER.index(label) if label in FILTER_ORDER else len(FILTER_ORDER)


def _write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def _fmt_num(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_sci(value: float) -> str:
    return f"1e{int(round(math.log10(value)))}" if value > 0 else "0"


def _linear_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if vmax <= vmin:
        return [vmin]
    raw_step = (vmax - vmin) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = min((1, 2, 5, 10), key=lambda m: abs(raw_step - m * magnitude)) * magnitude
    start = math.floor(vmin / step) * step
    ticks: list[float] = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.5:
            ticks.append(value)
        value += step
    return ticks[: count + 2]


def _log_ticks(vmin: float, vmax: float) -> list[float]:
    lo = math.floor(math.log10(max(vmin, 1e-12)))
    hi = math.ceil(math.log10(max(vmax, 1e-12)))
    return [10 ** exp for exp in range(lo, hi + 1)]


def _scale(value: float, vmin: float, vmax: float, size: float, *, log: bool = False) -> float:
    if log:
        value = math.log10(max(value, 1e-12))
        vmin = math.log10(max(vmin, 1e-12))
        vmax = math.log10(max(vmax, 1e-12))
    if vmax == vmin:
        return size / 2
    return (value - vmin) / (vmax - vmin) * size


def _text(x: float, y: float, content: str, *, size: int = 13, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(content)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, *, color: str = "#444", width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _circle(x: float, y: float, *, color: str, radius: float = 4.0) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="white" stroke-width="1"/>'


def _polyline(points: list[tuple[float, float]], *, color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'


def _render_axes(
    area: PlotArea,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    x_ticks: list[float],
    y_ticks: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_log: bool = False,
    y_log: bool = False,
) -> list[str]:
    parts: list[str] = []
    parts.append(_text(area.x + area.width / 2, area.y - 18, title, size=15))
    parts.append(_line(area.x, area.y, area.x, area.y + area.height, color="#555", width=1.2))
    parts.append(_line(area.x, area.y + area.height, area.x + area.width, area.y + area.height, color="#555", width=1.2))

    for tick in x_ticks:
        px = area.x + _scale(tick, x_min, x_max, area.width, log=x_log)
        parts.append(_line(px, area.y, px, area.y + area.height, color="#ddd", width=0.7))
        label = _fmt_sci(tick) if x_log else _fmt_num(tick)
        parts.append(_text(px, area.y + area.height + 22, label, size=11))

    for tick in y_ticks:
        py = area.y + area.height - _scale(tick, y_min, y_max, area.height, log=y_log)
        parts.append(_line(area.x, py, area.x + area.width, py, color="#ddd", width=0.7))
        label = _fmt_sci(tick) if y_log else _fmt_num(tick)
        parts.append(_text(area.x - 8, py + 4, label, size=11, anchor="end"))

    parts.append(_text(area.x + area.width / 2, area.y + area.height + 48, xlabel, size=13))
    parts.append(
        f'<text x="{area.x - 55:.1f}" y="{area.y + area.height / 2:.1f}" '
        f'font-family="Arial, sans-serif" font-size="13" text-anchor="middle" '
        f'transform="rotate(-90 {area.x - 55:.1f} {area.y + area.height / 2:.1f})">{html.escape(ylabel)}</text>'
    )
    return parts


def _render_legend(area: PlotArea, labels: list[str]) -> list[str]:
    parts: list[str] = []
    x = area.x + 15
    y = area.y + 18
    for i, label in enumerate(labels):
        yy = y + i * 20
        color = FILTER_COLORS.get(label, "#555")
        parts.append(_line(x, yy - 4, x + 18, yy - 4, color=color, width=2.0))
        parts.append(_circle(x + 9, yy - 4, color=color, radius=3.5))
        parts.append(_text(x + 26, yy, label, size=12, anchor="start"))
    return parts


def _series_labels(rows: Iterable[Any]) -> list[str]:
    present = {row.filter_label for row in rows}
    return [label for label in FILTER_ORDER if label in present] + sorted(present - set(FILTER_ORDER))


def _metric_range(values: list[tuple[float, float]], *, baseline_zero: bool, log: bool = False) -> tuple[float, float]:
    if log:
        positives = [max(mean - std, 1e-12) for mean, std in values if mean > 0]
        high = [mean + std for mean, std in values if mean + std > 0]
        return min(positives), max(high)

    lows = [mean - std for mean, std in values]
    highs = [mean + std for mean, std in values]
    vmin = 0.0 if baseline_zero else min(lows)
    vmax = max(highs)
    pad = (vmax - vmin) * 0.08 if vmax > vmin else max(1.0, vmax * 0.1)
    return max(0.0, vmin - pad) if baseline_zero else vmin - pad, vmax + pad


def _render_metric_by_k(
    area: PlotArea,
    rows: list[CrossKSummaryRow],
    *,
    mean_attr: str,
    std_attr: str,
    title: str,
    ylabel: str,
    baseline_zero: bool,
) -> list[str]:
    values = [(float(getattr(row, mean_attr)), float(getattr(row, std_attr))) for row in rows]
    y_min, y_max = _metric_range(values, baseline_zero=baseline_zero)
    x_min, x_max = 14.0, 32.0
    parts = _render_axes(
        area,
        title=title,
        xlabel="k",
        ylabel=ylabel,
        x_ticks=[15, 21, 31],
        y_ticks=_linear_ticks(y_min, y_max),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    labels = _series_labels(rows)
    for label in labels:
        sub = sorted([row for row in rows if row.filter_label == label], key=lambda row: row.k)
        color = FILTER_COLORS.get(label, "#555")
        points: list[tuple[float, float]] = []
        for row in sub:
            mean = float(getattr(row, mean_attr))
            std = float(getattr(row, std_attr))
            px = area.x + _scale(row.k, x_min, x_max, area.width)
            py = area.y + area.height - _scale(mean, y_min, y_max, area.height)
            lo = area.y + area.height - _scale(max(y_min, mean - std), y_min, y_max, area.height)
            hi = area.y + area.height - _scale(mean + std, y_min, y_max, area.height)
            parts.append(_line(px, lo, px, hi, color=color, width=1.0))
            parts.append(_line(px - 4, lo, px + 4, lo, color=color, width=1.0))
            parts.append(_line(px - 4, hi, px + 4, hi, color=color, width=1.0))
            points.append((px, py))
        if points:
            parts.append(_polyline(points, color=color))
            parts.extend(_circle(px, py, color=color) for px, py in points)

    parts.extend(_render_legend(area, labels))
    return parts


def _render_fpr_sweep(area: PlotArea, rows: list[FprSummaryRow]) -> list[str]:
    detection_floor = 5e-5
    x_min, x_max = 1e-4, 1e-2
    values = [
        (max(row.false_positive_rate_mean, detection_floor), row.false_positive_rate_std)
        for row in rows
    ]
    y_min, y_max = _metric_range(values, baseline_zero=False, log=True)
    y_min = min(y_min, detection_floor)
    y_max = max(y_max, x_max * 1.5)

    parts = _render_axes(
        area,
        title="Achieved FPR vs Target FPR (mean +/- std across genomes and k)",
        xlabel="Target FPR",
        ylabel="Empirical FPR",
        x_ticks=[1e-4, 1e-3, 1e-2],
        y_ticks=_log_ticks(y_min, y_max),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_log=True,
        y_log=True,
    )

    ideal = [
        (
            area.x + _scale(value, x_min, x_max, area.width, log=True),
            area.y + area.height - _scale(value, y_min, y_max, area.height, log=True),
        )
        for value in [x_min, x_max]
    ]
    parts.append(_line(ideal[0][0], ideal[0][1], ideal[1][0], ideal[1][1], color="#777", width=1.2, dash="4 4"))

    labels = _series_labels(rows)
    for label in labels:
        sub = sorted([row for row in rows if row.filter_label == label], key=lambda row: row.target_fpr)
        color = FILTER_COLORS.get(label, "#555")
        points: list[tuple[float, float]] = []
        for row in sub:
            mean = max(row.false_positive_rate_mean, detection_floor)
            std = row.false_positive_rate_std
            px = area.x + _scale(row.target_fpr, x_min, x_max, area.width, log=True)
            py = area.y + area.height - _scale(mean, y_min, y_max, area.height, log=True)
            low_val = max(detection_floor, mean - std)
            high_val = max(detection_floor, mean + std)
            lo = area.y + area.height - _scale(low_val, y_min, y_max, area.height, log=True)
            hi = area.y + area.height - _scale(high_val, y_min, y_max, area.height, log=True)
            parts.append(_line(px, lo, px, hi, color=color, width=1.0))
            parts.append(_line(px - 4, lo, px + 4, lo, color=color, width=1.0))
            parts.append(_line(px - 4, hi, px + 4, hi, color=color, width=1.0))
            points.append((px, py))
        if points:
            parts.append(_polyline(points, color=color))
            parts.extend(_circle(px, py, color=color) for px, py in points)

    parts.extend(_render_legend(area, labels + ["ideal"]))
    return parts


def _svg_document(width: int, height: int, parts: list[str]) -> str:
    body = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <rect width="100%" height="100%" fill="white"/>\n'
        f'  {body}\n'
        f'</svg>\n'
    )


def write_performance_svg(rows: list[CrossKSummaryRow], output_dir: Path, reference_fpr: float) -> Path:
    path = output_dir / "real_build_time_and_throughput.svg"
    subtitle = f"target FPR={reference_fpr:.1e}; unweighted mean +/- std across genomes"
    parts: list[str] = []
    parts.extend(
        _render_metric_by_k(
            PlotArea(80, 55, 610, 340),
            rows,
            mean_attr="build_time_per_1m_kmers_mean",
            std_attr="build_time_per_1m_kmers_std",
            title=f"Build Time per 1M k-mers by Filter and k ({subtitle})",
            ylabel="Seconds per 1M inserted k-mers",
            baseline_zero=True,
        )
    )
    parts.extend(
        _render_metric_by_k(
            PlotArea(825, 55, 610, 340),
            rows,
            mean_attr="throughput_qps_mean",
            std_attr="throughput_qps_std",
            title=f"Mean Throughput by Filter and k ({subtitle})",
            ylabel="Queries / second",
            baseline_zero=True,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_document(1500, 470, parts), encoding="utf-8")
    return path


def write_fpr_memory_svg(
    cross_k_rows: list[CrossKSummaryRow],
    fpr_rows: list[FprSummaryRow],
    output_dir: Path,
    reference_fpr: float,
) -> Path:
    path = output_dir / "real_fpr_and_memory.svg"
    subtitle = f"target FPR={reference_fpr:.1e}; unweighted mean +/- std across genomes"
    parts: list[str] = []
    parts.extend(_render_fpr_sweep(PlotArea(80, 55, 610, 340), fpr_rows))
    parts.extend(
        _render_metric_by_k(
            PlotArea(825, 55, 610, 340),
            cross_k_rows,
            mean_attr="memory_per_kmer_bytes_mean",
            std_attr="memory_per_kmer_bytes_std",
            title=f"Memory per k-mer by Filter and k ({subtitle})",
            ylabel="Bytes per inserted k-mer",
            baseline_zero=False,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_document(1500, 470, parts), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot real bacterial benchmark aggregates")
    parser.add_argument(
        "--results-dir",
        default="benchmarking/final_results/real/diverse_bacteria_4",
        help="Directory containing real benchmark JSON outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarking/final_results/real/diverse_bacteria_4/plots",
        help="Directory for generated plots and summary CSVs",
    )
    parser.add_argument("--reference-fpr", type=float, default=1e-3, help="Target FPR used for cross-k plots")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_run_rows(args.results_dir)
    cross_k_summary = summarize_cross_k(rows, args.reference_fpr)
    fpr_summary = summarize_fpr_sweep(rows)

    cross_k_csv = output_dir / "real_cross_k_summary.csv"
    fpr_csv = output_dir / "real_fpr_sweep_summary.csv"
    _write_csv(cross_k_csv, cross_k_summary)
    _write_csv(fpr_csv, fpr_summary)

    outputs = [
        write_performance_svg(cross_k_summary, output_dir, args.reference_fpr),
        write_fpr_memory_svg(cross_k_summary, fpr_summary, output_dir, args.reference_fpr),
        cross_k_csv,
        fpr_csv,
    ]

    print(f"Loaded benchmark JSON files: {len(rows)}")
    print("Generated real-data plot outputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
