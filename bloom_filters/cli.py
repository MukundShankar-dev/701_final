"""CLI for building and querying Bloom filter artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bloom_filters.bloom_filter import BloomFilter
from bloom_filters.builder import build_bloom_filter_from_kmer_file
from data.io_utils import load_kmers


def _build_command(args: argparse.Namespace) -> None:
    bloom = build_bloom_filter_from_kmer_file(
        kmer_file=args.kmer_file,
        expected_items=args.expected_items,
        false_positive_rate=args.fpr,
        hash_seed=args.seed,
        deduplicate=args.deduplicate,
    )
    bloom.save(args.output)

    print(f"Saved Bloom filter artifact: {args.output}")
    print(json.dumps(bloom.stats(), indent=2, sort_keys=True))


def _query_command(args: argparse.Namespace) -> None:
    bloom = BloomFilter.load(args.filter)

    if args.key is not None:
        print("present" if bloom.contains(args.key) else "absent")
        return

    if args.query_file is None:
        raise ValueError("Provide --key or --query-file")

    queries = list(load_kmers(args.query_file, deduplicate=False))
    decisions = bloom.batch_contains(queries)

    if args.output is None:
        for key, present in zip(queries, decisions, strict=True):
            print(f"{key}\t{int(present)}")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for key, present in zip(queries, decisions, strict=True):
                handle.write(f"{key}\t{int(present)}\n")
        print(f"Wrote query decisions: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bloom filter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build Bloom filter from k-mer file")
    build.add_argument("--kmer-file", required=True, help="Path to one-k-mer-per-line file")
    build.add_argument("--output", required=True, help="Output Bloom artifact JSON path")
    build.add_argument(
        "--expected-items",
        type=int,
        default=None,
        help="Expected item count (defaults to observed line count)",
    )
    build.add_argument("--fpr", type=float, default=1e-3, help="Target false positive rate")
    build.add_argument("--seed", type=int, default=0, help="Deterministic hash seed")
    build.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate input k-mers before building",
    )
    build.set_defaults(func=_build_command)

    query = subparsers.add_parser("query", help="Query a saved Bloom filter")
    query.add_argument("--filter", required=True, help="Path to Bloom artifact JSON")
    query_group = query.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--key", help="Single key to query")
    query_group.add_argument("--query-file", help="File with one query key per line")
    query.add_argument(
        "--output",
        default=None,
        help="Optional output file for batch query results",
    )
    query.set_defaults(func=_query_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
