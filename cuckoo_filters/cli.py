"""CLI for building and querying Cuckoo filter artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuckoo_filters.builder import build_cuckoo_filter_from_kmer_file
from cuckoo_filters.cuckoo_filter import CuckooFilter
from data.io_utils import load_kmers


def _build_command(args: argparse.Namespace) -> None:
    filt = build_cuckoo_filter_from_kmer_file(
        kmer_file=args.kmer_file,
        capacity=args.capacity,
        bucket_size=args.bucket_size,
        fingerprint_bits=args.fingerprint_bits,
        max_relocations=args.max_relocations,
        random_seed=args.seed,
        deduplicate=args.deduplicate,
    )
    filt.save(args.output)

    print(f"Saved Cuckoo filter artifact: {args.output}")
    print(json.dumps(filt.stats(), indent=2, sort_keys=True))


def _query_command(args: argparse.Namespace) -> None:
    filt = CuckooFilter.load(args.filter)

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


def _delete_command(args: argparse.Namespace) -> None:
    filt = CuckooFilter.load(args.filter)
    deleted = filt.delete(args.key)

    print("deleted" if deleted else "not-found")
    if args.output is not None:
        filt.save(args.output)
        print(f"Saved updated Cuckoo filter artifact: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cuckoo filter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build Cuckoo filter from k-mer file")
    build.add_argument("--kmer-file", required=True, help="Path to one-k-mer-per-line file")
    build.add_argument("--output", required=True, help="Output Cuckoo artifact JSON path")
    build.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Filter capacity, defaults to observed input count",
    )
    build.add_argument("--bucket-size", type=int, default=4, help="Items per bucket")
    build.add_argument("--fingerprint-bits", type=int, default=12, help="Fingerprint size in bits")
    build.add_argument("--max-relocations", type=int, default=500, help="Maximum relocation attempts")
    build.add_argument("--seed", type=int, default=0, help="Deterministic random seed")
    build.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate input k-mers before building",
    )
    build.set_defaults(func=_build_command)

    query = subparsers.add_parser("query", help="Query a saved Cuckoo filter")
    query.add_argument("--filter", required=True, help="Path to Cuckoo artifact JSON")
    query_group = query.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--key", help="Single key to query")
    query_group.add_argument("--query-file", help="File with one query key per line")
    query.add_argument("--output", default=None, help="Optional batch-output file")
    query.set_defaults(func=_query_command)

    delete = subparsers.add_parser("delete", help="Delete one key from saved Cuckoo filter")
    delete.add_argument("--filter", required=True, help="Path to Cuckoo artifact JSON")
    delete.add_argument("--key", required=True, help="Key to delete")
    delete.add_argument(
        "--output",
        default=None,
        help="Optional path to save updated artifact",
    )
    delete.set_defaults(func=_delete_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
