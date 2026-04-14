"""Cuckoo filter implementation with fingerprint-based storage."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

from benchmarking.interfaces import AMQFilter, FilterBuildMetadata, SupportsDeletion


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _approx_false_positive_rate(bucket_size: int, fingerprint_bits: int) -> float:
    """Approximate Cuckoo filter FPR using ``(2*b)/2^f``."""
    return min(1.0, (2.0 * bucket_size) / (2 ** fingerprint_bits))


class CuckooFilter(AMQFilter, SupportsDeletion):
    """Fingerprint-based Cuckoo filter with two candidate buckets per key."""

    FILTER_NAME = "cuckoo"

    def __init__(
        self,
        capacity: int,
        *,
        bucket_size: int = 4,
        fingerprint_bits: int = 12,
        max_relocations: int = 500,
        random_seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be > 0, got {bucket_size}")
        if fingerprint_bits <= 1:
            raise ValueError(f"fingerprint_bits must be > 1, got {fingerprint_bits}")
        if max_relocations <= 0:
            raise ValueError(f"max_relocations must be > 0, got {max_relocations}")

        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.max_relocations = max_relocations
        self.random_seed = random_seed

        self._fingerprint_mask = (1 << fingerprint_bits) - 1

        target_load = 0.95
        approx_buckets = math.ceil(capacity / (bucket_size * target_load))
        self.bucket_count = _next_power_of_two(max(2, approx_buckets))

        self._buckets: list[list[int]] = [[] for _ in range(self.bucket_count)]
        self._inserted_count = 0
        self._insertion_failures = 0
        self._build_time_seconds: float | None = None
        self._built = False
        self._rng = random.Random(random_seed)

    def _hash64(self, payload: bytes) -> int:
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

    def _fingerprint(self, key: str) -> int:
        raw = self._hash64(key.encode("utf-8")) & self._fingerprint_mask
        return raw if raw != 0 else 1

    def _index1(self, key: str) -> int:
        return self._hash64(f"i1:{key}".encode("utf-8")) & (self.bucket_count - 1)

    def _index2(self, index1: int, fingerprint: int) -> int:
        fp_hash = self._hash64(f"fp:{fingerprint}".encode("utf-8"))
        return (index1 ^ fp_hash) & (self.bucket_count - 1)

    def _insert_fingerprint(self, fp: int, i1: int, i2: int) -> bool:
        b1 = self._buckets[i1]
        if len(b1) < self.bucket_size:
            b1.append(fp)
            return True

        b2 = self._buckets[i2]
        if len(b2) < self.bucket_size:
            b2.append(fp)
            return True

        index = i1 if self._rng.random() < 0.5 else i2
        current_fp = fp

        for _ in range(self.max_relocations):
            bucket = self._buckets[index]
            evict_pos = self._rng.randrange(self.bucket_size)
            bucket[evict_pos], current_fp = current_fp, bucket[evict_pos]

            index = self._index2(index, current_fp)
            alt_bucket = self._buckets[index]
            if len(alt_bucket) < self.bucket_size:
                alt_bucket.append(current_fp)
                return True

        return False

    def insert(self, key: str) -> bool:
        """Insert one key into the Cuckoo filter."""
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(i1, fp)

        inserted = self._insert_fingerprint(fp, i1, i2)
        if inserted:
            self._inserted_count += 1
        else:
            self._insertion_failures += 1
        return inserted

    def build(self, keys: Sequence[str]) -> None:
        """Build or rebuild filter from sequence of keys.

        Raises:
            RuntimeError: If not all keys could be inserted.
        """
        self._buckets = [[] for _ in range(self.bucket_count)]
        self._inserted_count = 0
        self._insertion_failures = 0
        self._rng = random.Random(self.random_seed)

        start = time.perf_counter()
        for key in keys:
            self.insert(key)
        self._build_time_seconds = time.perf_counter() - start
        self._built = True

        if self._insertion_failures > 0:
            raise RuntimeError(
                "Cuckoo filter build failed to insert all keys; "
                f"failures={self._insertion_failures}, inserted={self._inserted_count}, "
                f"capacity={self.capacity}, load_factor={self.load_factor():.4f}"
            )

    def contains(self, key: str) -> bool:
        """Return membership query result for one key."""
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(i1, fp)

        return fp in self._buckets[i1] or fp in self._buckets[i2]

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        """Return membership query results for many keys."""
        return [self.contains(key) for key in keys]

    def delete(self, key: str) -> bool:
        """Delete a key fingerprint from either candidate bucket."""
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(i1, fp)

        if fp in self._buckets[i1]:
            self._buckets[i1].remove(fp)
            self._inserted_count = max(0, self._inserted_count - 1)
            return True
        if fp in self._buckets[i2]:
            self._buckets[i2].remove(fp)
            self._inserted_count = max(0, self._inserted_count - 1)
            return True
        return False

    def load_factor(self) -> float:
        """Return current table occupancy as a fraction."""
        total_slots = self.bucket_count * self.bucket_size
        return self._inserted_count / max(1, total_slots)

    def memory_usage_bytes(self) -> int:
        """Approximate raw fingerprint table bytes.

        Note: This excludes Python object/list overhead, which is significant.
        """
        total_slots = self.bucket_count * self.bucket_size
        return math.ceil(total_slots * self.fingerprint_bits / 8)

    def save(self, path: str | Path) -> None:
        """Serialize filter to JSON artifact."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "filter_name": self.FILTER_NAME,
            "capacity": self.capacity,
            "bucket_size": self.bucket_size,
            "fingerprint_bits": self.fingerprint_bits,
            "max_relocations": self.max_relocations,
            "random_seed": self.random_seed,
            "bucket_count": self.bucket_count,
            "inserted_count": self._inserted_count,
            "insertion_failures": self._insertion_failures,
            "build_time_seconds": self._build_time_seconds,
            "buckets": self._buckets,
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CuckooFilter:
        """Load filter state from serialized JSON payload."""
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Cuckoo filter artifact not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("filter_name") != cls.FILTER_NAME:
            raise ValueError(
                f"Unexpected filter artifact type: {payload.get('filter_name')}"
            )

        filt = cls(
            capacity=int(payload["capacity"]),
            bucket_size=int(payload["bucket_size"]),
            fingerprint_bits=int(payload["fingerprint_bits"]),
            max_relocations=int(payload["max_relocations"]),
            random_seed=int(payload.get("random_seed", 0)),
        )
        filt.bucket_count = int(payload["bucket_count"])
        filt._buckets = [list(map(int, bucket)) for bucket in payload["buckets"]]
        filt._inserted_count = int(payload.get("inserted_count", 0))
        filt._insertion_failures = int(payload.get("insertion_failures", 0))
        filt._build_time_seconds = payload.get("build_time_seconds")
        filt._built = True
        return filt

    def stats(self) -> dict[str, Any]:
        """Return implementation statistics and metadata."""
        approx_fpr = _approx_false_positive_rate(self.bucket_size, self.fingerprint_bits)
        metadata = FilterBuildMetadata(
            filter_name=self.FILTER_NAME,
            parameters={
                "capacity": self.capacity,
                "bucket_size": self.bucket_size,
                "fingerprint_bits": self.fingerprint_bits,
                "max_relocations": self.max_relocations,
                "bucket_count": self.bucket_count,
                "random_seed": self.random_seed,
            },
            inserted_keys=self._inserted_count,
            target_false_positive_rate=approx_fpr,
            actual_memory_usage_bytes=self.memory_usage_bytes(),
            build_time_seconds=self._build_time_seconds,
        )
        out = metadata.to_dict()
        out.update(
            {
                "load_factor": self.load_factor(),
                "insertion_failures": self._insertion_failures,
                "fingerprint_only_false_positive_rate": 2 ** (-self.fingerprint_bits),
                "built": self._built,
            }
        )
        return out
