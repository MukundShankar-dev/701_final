"""Generate benchmark plots from JSON run outputs."""

from __future__ import annotations

import argparse

from benchmarking.plotting import generate_default_plots


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
    outputs = generate_default_plots(args.results_dir, args.output_dir)
    print("Generated plots:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
