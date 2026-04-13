"""FASTA loading utilities for genomic sequence datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_fasta_records(path: str | Path, uppercase: bool = True) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` records from a FASTA file.

    The parser supports multiline sequence blocks and streams the file line-by-line
    to keep memory usage modest for large inputs.

    Args:
        path: Path to FASTA file.
        uppercase: Whether to normalize emitted sequences to uppercase.

    Yields:
        Tuples of ``(header_without_gt, sequence)``.

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
        ValueError: If sequence content appears before any header line.
    """
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA path does not exist: {fasta_path}")

    header: str | None = None
    seq_chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(seq_chunks)
                    yield header, sequence.upper() if uppercase else sequence
                header = line[1:].strip()
                seq_chunks = []
                continue

            if header is None:
                raise ValueError(
                    "Encountered FASTA sequence content before header line starting with '>'"
                )
            seq_chunks.append(line)

    if header is not None:
        sequence = "".join(seq_chunks)
        yield header, sequence.upper() if uppercase else sequence


def iter_fasta_sequences(path: str | Path, uppercase: bool = True) -> Iterator[str]:
    """Yield sequence strings from FASTA records."""
    for _, sequence in iter_fasta_records(path=path, uppercase=uppercase):
        yield sequence


def load_fasta_sequences(path: str | Path, uppercase: bool = True) -> list[str]:
    """Materialize all FASTA sequences into a list.

    For large datasets prefer ``iter_fasta_sequences`` to avoid high memory usage.
    """
    return list(iter_fasta_sequences(path=path, uppercase=uppercase))
