"""Memory measurement helpers for benchmark reporting."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import psutil


def estimate_python_object_size_bytes(obj: Any) -> int:
    """Best-effort Python object size estimate.

    Note: ``sys.getsizeof`` does not recursively include nested object sizes.
    Use this estimate with caution in comparative analysis.
    """
    return int(sys.getsizeof(obj))


def process_rss_bytes() -> int:
    """Return current process RSS in bytes."""
    proc = psutil.Process(os.getpid())
    return int(proc.memory_info().rss)


def serialized_size_bytes(path: str | Path) -> int:
    """Return on-disk size in bytes for serialized artifact."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Serialized artifact not found: {p}")
    return int(p.stat().st_size)
