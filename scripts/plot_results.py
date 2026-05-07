"""Generate benchmark plots from JSON run outputs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def _prepare_plot_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "amq_matplotlib_cache"
    mplconfig = cache_root / "mplconfig"
    xdg_cache = cache_root / "xdg"
    mplconfig.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AMQ benchmark results")
    parser.add_argument(
        "--results-dir",
        default="benchmarking/results",
        help="Directory containing benchmark JSON outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarking/results/plots",
        help="Output directory for PNG plots",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _prepare_plot_cache()

    from benchmarking.plotting import generate_default_plots

    outputs = generate_default_plots(args.results_dir, args.output_dir)
    print("Generated plots:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
