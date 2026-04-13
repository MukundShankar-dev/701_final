"""Data loading and synthetic dataset utilities for AMQ experiments."""

from data.fasta_loader import iter_fasta_records, iter_fasta_sequences, load_fasta_sequences
from data.io_utils import append_csv_rows, load_json, load_kmers, save_json, save_kmers, write_csv_rows
from data.kmers import canonical_kmer, generate_kmers, load_kmers_from_fasta, reverse_complement
from data.synthetic import (
    generate_negative_kmers,
    generate_synthetic_contigs,
    generate_synthetic_kmer_set,
    generate_synthetic_sequence,
    kmers_from_synthetic_contigs,
)

__all__ = [
    "append_csv_rows",
    "canonical_kmer",
    "generate_kmers",
    "generate_negative_kmers",
    "generate_synthetic_contigs",
    "generate_synthetic_kmer_set",
    "generate_synthetic_sequence",
    "iter_fasta_records",
    "iter_fasta_sequences",
    "kmers_from_synthetic_contigs",
    "load_fasta_sequences",
    "load_json",
    "load_kmers",
    "load_kmers_from_fasta",
    "reverse_complement",
    "save_json",
    "save_kmers",
    "write_csv_rows",
]
