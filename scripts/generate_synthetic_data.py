"""Generate synthetic k-mer datasets for benchmarking."""

from __future__ import annotations

import argparse

from data.io_utils import save_kmers
from data.synthetic import kmers_from_synthetic_contigs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic k-mer dataset")
    parser.add_argument("--output", required=True, help="Output one-k-mer-per-line file")
    parser.add_argument("--k", type=int, required=True, help="k-mer size")
    parser.add_argument("--num-contigs", type=int, default=10, help="Number of synthetic contigs")
    parser.add_argument("--contig-length", type=int, default=100_000, help="Length per contig")
    parser.add_argument("--gc-bias", type=float, default=0.5, help="GC bias in [0, 1]")
    parser.add_argument("--canonical", action="store_true", help="Canonicalize k-mers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kmers = kmers_from_synthetic_contigs(
        num_contigs=args.num_contigs,
        contig_length=args.contig_length,
        k=args.k,
        gc_bias=args.gc_bias,
        canonical=args.canonical,
        deduplicate=True,
        seed=args.seed,
    )
    count = save_kmers(args.output, kmers)
    print(f"Generated {count} k-mers into {args.output}")


if __name__ == "__main__":
    main()
