"""CLI for building and querying XOR filter artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.io_utils import load_kmers
from xor_filters.builder import build_xor_filter_from_kmer_file
from xor_filters.xor_filter import XorFilter


def _build_command(args: argparse.Namespace) -> None:
    filt = build_xor_filter_from_kmer_file(
        kmer_file=args.kmer_file,
        fingerprint_bits=args.fingerprint_bits,
        backend=args.backend,
        hash_seed=args.seed,
        size_factor=args.size_factor,
        max_retries=args.max_retries,
        deduplicate=args.deduplicate,
    )
    filt.save(args.output)

    print(f"Saved XOR filter artifact: {args.output}")
    print(json.dumps(filt.stats(), indent=2, sort_keys=True))


def _query_command(args: argparse.Namespace) -> None:
    filt = XorFilter.load(args.filter)

    if args.key is not None:
        print("present" if filt.contains(args.key) else "absent")
        return

    if args.query_file is None:
        raise ValueError("Provide --key or --query-file")

    queries = list(load_kmers(args.query_file, deduplicate=False))
    decisions = filt.batch_contains(queries)

    if args.output is None:
        for key, present in zip(queries, decisions, strict=True):
            print(f"{key}\t{int(present)}")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for key, present in zip(queries, decisions, strict=True):
                handle.write(f"{key}\t{int(present)}\n")
        print(f"Wrote query decisions: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XOR filter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build XOR filter from k-mer file")
    build.add_argument("--kmer-file", required=True, help="Path to one-k-mer-per-line file")
    build.add_argument("--output", required=True, help="Output XOR artifact JSON path")
    build.add_argument("--fingerprint-bits", type=int, default=8, help="Fingerprint size in bits")
    build.add_argument("--seed", type=int, default=0, help="Hash seed")
    build.add_argument("--size-factor", type=float, default=1.23, help="XOR array size factor")
    build.add_argument("--max-retries", type=int, default=64, help="Maximum peel retries")
    build.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "native", "python", "fallback"],
        help="Backend selection strategy; fallback is a backward-compatible alias for python",
    )
    build.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate input k-mers before building",
    )
    build.set_defaults(func=_build_command)

    query = subparsers.add_parser("query", help="Query a saved XOR filter")
    query.add_argument("--filter", required=True, help="Path to XOR artifact JSON")
    query_group = query.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--key", help="Single key to query")
    query_group.add_argument("--query-file", help="File with one query key per line")
    query.add_argument("--output", default=None, help="Optional batch-output file")
    query.set_defaults(func=_query_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
