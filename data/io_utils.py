"""I/O helpers for k-mer files and benchmark artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Iterator


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_kmers(path: str | Path, kmers: Iterable[str]) -> int:
    """Save one k-mer per line and return the number written."""
    out_path = Path(path)
    _ensure_parent(out_path)

    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for kmer in kmers:
            k = kmer.strip()
            if not k:
                continue
            handle.write(f"{k}\n")
            count += 1
    return count


def load_kmers(path: str | Path, deduplicate: bool = False) -> Iterator[str]:
    """Stream one-k-mer-per-line files."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"k-mer file not found: {in_path}")

    seen: set[str] | None = set() if deduplicate else None
    with in_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            kmer = raw.strip().upper()
            if not kmer:
                continue
            if seen is not None:
                if kmer in seen:
                    continue
                seen.add(kmer)
            yield kmer


def save_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    """Write JSON to disk with deterministic key ordering."""
    out_path = Path(path)
    _ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)


def load_json(path: str | Path) -> Any:
    """Read JSON payload from disk."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"JSON file not found: {in_path}")
    with in_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str] | None = None,
) -> int:
    """Write rows to CSV and return row count.

    If ``fieldnames`` is not supplied, columns are inferred from the first row.
    """
    out_path = Path(path)
    _ensure_parent(out_path)

    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration:
        if fieldnames is None:
            raise ValueError("Cannot infer CSV fieldnames from empty rows")
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return 0

    if fieldnames is None:
        fieldnames = list(first.keys())

    count = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(first))
        count += 1
        for row in iterator:
            writer.writerow(dict(row))
            count += 1

    return count


def append_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
) -> int:
    """Append rows to CSV and return appended row count."""
    out_path = Path(path)
    _ensure_parent(out_path)
    file_exists = out_path.exists()

    count = 0
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
            count += 1
    return count
