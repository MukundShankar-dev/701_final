"""Deterministic bit-array Bloom filter implementation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

from benchmarking.interfaces import AMQFilter, FilterBuildMetadata


def _optimal_num_bits(expected_items: int, false_positive_rate: float) -> int:
    """Compute optimal bit-array size ``m`` for a Bloom filter."""
    if expected_items <= 0:
        raise ValueError(f"expected_items must be > 0, got {expected_items}")
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError(
            f"false_positive_rate must be in (0, 1), got {false_positive_rate}"
        )
    m = -expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)
    return max(8, math.ceil(m))


def _optimal_num_hashes(num_bits: int, expected_items: int) -> int:
    """Compute optimal number of hash functions ``k`` for a Bloom filter."""
    k = (num_bits / expected_items) * math.log(2)
    return max(1, round(k))


class BloomFilter(AMQFilter):
    """Bit-array Bloom filter using deterministic double hashing."""

    FILTER_NAME = "bloom"

    def __init__(
        self,
        expected_items: int,
        false_positive_rate: float,
        *,
        hash_seed: int = 0,
    ) -> None:
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        self.hash_seed = hash_seed

        self.num_bits = _optimal_num_bits(expected_items, false_positive_rate)
        self.num_hashes = _optimal_num_hashes(self.num_bits, expected_items)
        self._bytes = bytearray((self.num_bits + 7) // 8)
        self._inserted_count = 0
        self._built = False
        self._build_time_seconds: float | None = None

    @staticmethod
    def _hashes(key: str, hash_seed: int, num_hashes: int, modulus: int) -> list[int]:
        payload = f"{hash_seed}:{key}".encode("utf-8")

        h1 = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        h2 = int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little")

        if h2 == 0:
            h2 = 0x9E3779B185EBCA87

        return [((h1 + i * h2) % modulus) for i in range(num_hashes)]

    def _set_bit(self, bit_index: int) -> None:
        byte_index = bit_index // 8
        bit_offset = bit_index % 8
        self._bytes[byte_index] |= 1 << bit_offset

    def _get_bit(self, bit_index: int) -> bool:
        byte_index = bit_index // 8
        bit_offset = bit_index % 8
        return bool(self._bytes[byte_index] & (1 << bit_offset))

    def add(self, key: str) -> None:
        """Insert one key into the Bloom filter."""
        for bit_index in self._hashes(
            key,
            hash_seed=self.hash_seed,
            num_hashes=self.num_hashes,
            modulus=self.num_bits,
        ):
            self._set_bit(bit_index)
        self._inserted_count += 1

    def build(self, keys: Sequence[str]) -> None:
        """Build or rebuild the filter from a sequence of keys."""
        self._bytes = bytearray((self.num_bits + 7) // 8)
        self._inserted_count = 0

        start = time.perf_counter()
        for key in keys:
            self.add(key)
        self._build_time_seconds = time.perf_counter() - start
        self._built = True

    def contains(self, key: str) -> bool:
        """Return membership query result for a single key."""
        for bit_index in self._hashes(
            key,
            hash_seed=self.hash_seed,
            num_hashes=self.num_hashes,
            modulus=self.num_bits,
        ):
            if not self._get_bit(bit_index):
                return False
        return True

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        """Return membership query results for a batch of keys."""
        return [self.contains(key) for key in keys]

    def memory_usage_bytes(self) -> int:
        """Return in-memory footprint estimate for internal bit-array bytes."""
        return len(self._bytes)

    def save(self, path: str | Path) -> None:
        """Serialize filter as JSON with hex-encoded bit-array payload."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "filter_name": self.FILTER_NAME,
            "expected_items": self.expected_items,
            "false_positive_rate": self.false_positive_rate,
            "hash_seed": self.hash_seed,
            "num_bits": self.num_bits,
            "num_hashes": self.num_hashes,
            "inserted_count": self._inserted_count,
            "build_time_seconds": self._build_time_seconds,
            "bitarray_hex": bytes(self._bytes).hex(),
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BloomFilter:
        """Load filter state from serialized JSON payload."""
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Bloom filter artifact not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("filter_name") != cls.FILTER_NAME:
            raise ValueError(
                f"Unexpected filter artifact type: {payload.get('filter_name')}"
            )

        filt = cls(
            expected_items=int(payload["expected_items"]),
            false_positive_rate=float(payload["false_positive_rate"]),
            hash_seed=int(payload.get("hash_seed", 0)),
        )
        filt.num_bits = int(payload["num_bits"])
        filt.num_hashes = int(payload["num_hashes"])

        raw = bytes.fromhex(payload["bitarray_hex"])
        filt._bytes = bytearray(raw)
        filt._inserted_count = int(payload.get("inserted_count", 0))
        filt._build_time_seconds = payload.get("build_time_seconds")
        filt._built = True

        return filt

    def stats(self) -> dict[str, Any]:
        """Return filter statistics and derived metadata."""
        bits_per_item = (
            self.num_bits / max(1, self._inserted_count)
            if self._inserted_count > 0
            else self.num_bits / max(1, self.expected_items)
        )
        metadata = FilterBuildMetadata(
            filter_name=self.FILTER_NAME,
            parameters={
                "expected_items": self.expected_items,
                "false_positive_rate": self.false_positive_rate,
                "num_bits": self.num_bits,
                "num_hashes": self.num_hashes,
                "hash_seed": self.hash_seed,
            },
            inserted_keys=self._inserted_count,
            target_false_positive_rate=self.false_positive_rate,
            actual_memory_usage_bytes=self.memory_usage_bytes(),
            build_time_seconds=self._build_time_seconds,
        )
        out = metadata.to_dict()
        out.update(
            {
                "bits_per_item": bits_per_item,
                "built": self._built,
            }
        )
        return out
