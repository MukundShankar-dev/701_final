"""Build all filter types from a shared k-mer input file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bloom_filters.builder import build_bloom_filter_from_kmer_file
from cuckoo_filters.builder import build_cuckoo_filter_from_kmer_file
from learned_filters.learned_filter import LearnedFilter
from data.io_utils import load_kmers
from xor_filters.builder import build_xor_filter_from_kmer_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build all AMQ filters")
    parser.add_argument("--kmer-file", required=True, help="Input k-mer file")
    parser.add_argument("--output-dir", required=True, help="Output artifact directory")
    parser.add_argument("--k", type=int, required=True, help="k-mer length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bloom = build_bloom_filter_from_kmer_file(
        args.kmer_file,
        false_positive_rate=1e-3,
        deduplicate=True,
        hash_seed=args.seed,
    )
    bloom_path = out_dir / "bloom.json"
    bloom.save(bloom_path)

    cuckoo = build_cuckoo_filter_from_kmer_file(
        args.kmer_file,
        deduplicate=True,
        random_seed=args.seed,
    )
    cuckoo_path = out_dir / "cuckoo.json"
    cuckoo.save(cuckoo_path)

    xorf = build_xor_filter_from_kmer_file(
        args.kmer_file,
        deduplicate=True,
        backend="auto",
        hash_seed=args.seed,
    )
    xor_path = out_dir / "xor.json"
    xorf.save(xor_path)

    positives = list(load_kmers(args.kmer_file, deduplicate=True))
    learned = LearnedFilter(k=args.k, random_seed=args.seed)
    learned.train(positives)
    learned.save(out_dir / "learned")

    summary = {
        "bloom": bloom.stats(),
        "cuckoo": cuckoo.stats(),
        "xor": xorf.stats(),
        "learned": learned.stats(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
