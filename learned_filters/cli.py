"""CLI for training and querying learned filter artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.io_utils import load_kmers
from learned_filters.dataset import build_training_dataset, split_dataset
from learned_filters.learned_filter import LearnedFilter


def _train_command(args: argparse.Namespace) -> None:
    positives = list(load_kmers(args.kmer_file, deduplicate=args.deduplicate))
    if not positives:
        raise ValueError("Input k-mer file is empty")

    k = args.k if args.k is not None else len(positives[0])
    filt = LearnedFilter(
        k=k,
        model_threshold=args.threshold,
        backup_false_positive_rate=args.backup_fpr,
        random_seed=args.seed,
        model_backend=args.model_backend,
        ngram_features=args.ngram_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        total_false_positive_rate=args.total_fpr,
        model_false_positive_rate=args.model_fpr,
        prefilter_false_positive_rate=args.prefilter_fpr,
        refit_model_on_full_dataset=args.refit_full,
    )

    metrics = filt.train(
        positive_kmers=positives,
        negative_count=args.negative_count,
        negative_mutation_rate=args.negative_mutation_rate,
    )
    filt.save(args.output)

    print(f"Saved learned filter artifact: {args.output}")
    print(json.dumps({"train_metrics": metrics, "stats": filt.stats()}, indent=2, sort_keys=True))


def _query_command(args: argparse.Namespace) -> None:
    filt = LearnedFilter.load(args.filter)

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


def _evaluate_command(args: argparse.Namespace) -> None:
    filt = LearnedFilter.load(args.filter)
    positives = list(load_kmers(args.kmer_file, deduplicate=True))
    dataset = build_training_dataset(
        positives,
        negative_count=args.negative_count,
        negative_mutation_rate=args.negative_mutation_rate,
        random_seed=args.seed,
    )
    _, _, test_set = split_dataset(dataset)

    metrics = filt.evaluate_heldout(test_set)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learned filter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train learned filter from positive k-mers")
    train.add_argument("--kmer-file", required=True, help="Positive k-mer file")
    train.add_argument("--output", required=True, help="Base output path for artifacts")
    train.add_argument("--k", type=int, default=None, help="k-mer length; default inferred from first k-mer")
    train.add_argument("--threshold", type=float, default=0.5, help="Classifier decision threshold")
    train.add_argument("--backup-fpr", type=float, default=1e-3, help="Backup Bloom filter FPR")
    train.add_argument("--negative-count", type=int, default=None, help="Negative samples for training")
    train.add_argument("--negative-mutation-rate", type=float, default=0.2, help="Negative mutation rate")
    train.add_argument(
        "--model-backend",
        default="ngram_sgd",
        choices=[
            "composition_logistic",
            "dna_ngram_sgd",
            "ngram_nb",
            "ngram_sgd",
            "prefix_set",
            "position_logistic",
        ],
        help="Classifier backend",
    )
    train.add_argument("--total-fpr", type=float, default=None, help="Overall learned-filter FPR target")
    train.add_argument("--model-fpr", type=float, default=None, help="Model FPR budget used for threshold tuning")
    train.add_argument("--prefilter-fpr", type=float, default=None, help="Optional sandwich prefilter Bloom FPR")
    train.add_argument(
        "--refit-full",
        action="store_true",
        help="Refit the classifier on all generated training data after validation threshold tuning",
    )
    train.add_argument("--ngram-features", type=int, default=4096, help="Hashed n-gram feature count")
    train.add_argument("--ngram-min", type=int, default=3, help="Minimum character n-gram length")
    train.add_argument("--ngram-max", type=int, default=5, help="Maximum character n-gram length")
    train.add_argument("--seed", type=int, default=0, help="Random seed")
    train.add_argument("--deduplicate", action="store_true", help="Deduplicate input positives")
    train.set_defaults(func=_train_command)

    query = subparsers.add_parser("query", help="Query a trained learned filter")
    query.add_argument("--filter", required=True, help="Base path used at training output")
    qgroup = query.add_mutually_exclusive_group(required=True)
    qgroup.add_argument("--key", help="Single query k-mer")
    qgroup.add_argument("--query-file", help="Batch query file")
    query.add_argument("--output", default=None, help="Optional output file for batch decisions")
    query.set_defaults(func=_query_command)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate classifier metrics on synthetic held-out split")
    evaluate.add_argument("--filter", required=True, help="Base path used at training output")
    evaluate.add_argument("--kmer-file", required=True, help="Positive k-mer file")
    evaluate.add_argument("--negative-count", type=int, default=None, help="Negative samples")
    evaluate.add_argument("--negative-mutation-rate", type=float, default=0.2, help="Negative mutation rate")
    evaluate.add_argument("--seed", type=int, default=0, help="Random seed")
    evaluate.set_defaults(func=_evaluate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
