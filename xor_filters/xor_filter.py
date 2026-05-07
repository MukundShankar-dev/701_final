"""XOR filter facade with optional native backend and pure-Python backend.

The pure-Python backend implements a static peel-based XOR filter:
1) hash each key to three positions,
2) peel the resulting 3-uniform hypergraph,
3) assign fingerprints in reverse peel order.

This is not intended to compete with native implementations for speed, but it
is a real XOR-filter construction and is suitable for framework-level
experiments when a native package is unavailable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from benchmarking.interfaces import AMQFilter, FilterBuildMetadata

_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    """SplitMix64 finalizer for deriving independent-looking hash words."""
    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    return (z ^ (z >> 31)) & _MASK64


@dataclass(slots=True)
class _NativeBackendHandle:
    module_name: str
    class_name: str


class _PythonXorBackend:
    """Deterministic peel-based XOR filter backend."""

    def __init__(
        self,
        fingerprint_bits: int,
        *,
        hash_seed: int = 0,
        size_factor: float = 1.23,
        max_retries: int = 64,
    ) -> None:
        if fingerprint_bits <= 1:
            raise ValueError("fingerprint_bits must be > 1")
        if size_factor <= 1.0:
            raise ValueError("size_factor must be > 1")
        if max_retries <= 0:
            raise ValueError("max_retries must be > 0")

        self.fingerprint_bits = fingerprint_bits
        self.hash_seed = hash_seed
        self.size_factor = size_factor
        self.max_retries = max_retries
        self.target_fpr = 2 ** (-fingerprint_bits)

        self._fingerprint_mask = (1 << fingerprint_bits) - 1
        self._fingerprints: list[int] = []
        self._array_length = 0
        self._inserted_count = 0
        self._build_seed = hash_seed
        self._build_attempts = 0
        self._effective_size_factor = size_factor

    def _hash_key(self, key: str, seed: int | None = None) -> int:
        build_seed = self._build_seed if seed is None else seed
        payload = f"{build_seed}:{key}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

    def _fingerprint_from_hash(self, key_hash: int) -> int:
        fp = _mix64(key_hash ^ 0xA5A5A5A5A5A5A5A5) & self._fingerprint_mask
        return fp if fp != 0 else 1

    def _locations_from_hash(self, key_hash: int, array_length: int | None = None) -> tuple[int, int, int]:
        m = self._array_length if array_length is None else array_length
        if m < 3:
            raise ValueError("array_length must be >= 3")

        x0 = _mix64(key_hash)
        x1 = _mix64(key_hash ^ 0xD6E8FEB86659FD93)
        x2 = _mix64(key_hash ^ 0xA0761D6478BD642F)

        p0 = x0 % m
        p1 = x1 % m
        if p1 == p0:
            p1 = (p1 + 1 + ((x1 >> 32) % (m - 1))) % m

        p2 = x2 % m
        if p2 == p0 or p2 == p1:
            while p2 == p0 or p2 == p1:
                p2 = (p2 + 1) % m

        return p0, p1, p2

    def _try_build_once(
        self,
        keys: Sequence[str],
        *,
        seed: int,
        array_length: int,
    ) -> tuple[list[int], list[tuple[int, int, int]], list[tuple[int, int]]] | None:
        counts = [0] * array_length
        edge_xors = [0] * array_length
        key_hashes: list[int] = []
        locations: list[tuple[int, int, int]] = []

        for edge_index, key in enumerate(keys):
            key_hash = self._hash_key(key, seed)
            locs = self._locations_from_hash(key_hash, array_length)
            key_hashes.append(key_hash)
            locations.append(locs)
            for loc in locs:
                counts[loc] += 1
                edge_xors[loc] ^= edge_index

        queue = [idx for idx, count in enumerate(counts) if count == 1]
        peeled = [False] * len(keys)
        stack: list[tuple[int, int]] = []

        while queue:
            loc = queue.pop()
            if counts[loc] != 1:
                continue

            edge_index = edge_xors[loc]
            if edge_index < 0 or edge_index >= len(keys) or peeled[edge_index]:
                continue

            peeled[edge_index] = True
            stack.append((edge_index, loc))

            for neighbor in locations[edge_index]:
                counts[neighbor] -= 1
                edge_xors[neighbor] ^= edge_index
                if counts[neighbor] == 1:
                    queue.append(neighbor)

        if len(stack) != len(keys):
            return None

        fingerprints = [0] * array_length
        for edge_index, assigned_loc in reversed(stack):
            wanted = self._fingerprint_from_hash(key_hashes[edge_index])
            accum = 0
            for loc in locations[edge_index]:
                if loc != assigned_loc:
                    accum ^= fingerprints[loc]
            fingerprints[assigned_loc] = wanted ^ accum

        return fingerprints, locations, stack

    def build(self, keys: Sequence[str]) -> None:
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise ValueError("keys must be non-empty")

        n = len(unique_keys)
        last_length = 0
        for attempt in range(self.max_retries):
            # Increase the array modestly after several failed peel attempts.
            factor = self.size_factor + 0.03 * (attempt // 8)
            array_length = max(3, math.ceil(factor * n))
            if array_length == last_length:
                array_length += 1
            last_length = array_length

            seed = self.hash_seed + attempt * 0x9E3779B1
            built = self._try_build_once(
                unique_keys,
                seed=seed,
                array_length=array_length,
            )
            if built is None:
                continue

            fingerprints, _, _ = built
            self._fingerprints = fingerprints
            self._array_length = array_length
            self._inserted_count = n
            self._build_seed = seed
            self._build_attempts = attempt + 1
            self._effective_size_factor = factor
            return

        raise RuntimeError(
            "Failed to construct XOR filter after "
            f"{self.max_retries} attempts; try increasing size_factor"
        )

    def contains(self, key: str) -> bool:
        if not self._fingerprints:
            return False
        key_hash = self._hash_key(key)
        loc0, loc1, loc2 = self._locations_from_hash(key_hash)
        observed = self._fingerprints[loc0] ^ self._fingerprints[loc1] ^ self._fingerprints[loc2]
        return observed == self._fingerprint_from_hash(key_hash)

    def memory_usage_bytes(self) -> int:
        return math.ceil(len(self._fingerprints) * self.fingerprint_bits / 8)

    def to_payload(self) -> dict[str, Any]:
        return {
            "fingerprint_bits": self.fingerprint_bits,
            "hash_seed": self.hash_seed,
            "build_seed": self._build_seed,
            "size_factor": self.size_factor,
            "effective_size_factor": self._effective_size_factor,
            "max_retries": self.max_retries,
            "array_length": self._array_length,
            "inserted_count": self._inserted_count,
            "build_attempts": self._build_attempts,
            "fingerprints": self._fingerprints,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "_PythonXorBackend":
        inst = cls(
            fingerprint_bits=int(payload["fingerprint_bits"]),
            hash_seed=int(payload.get("hash_seed", 0)),
            size_factor=float(payload.get("size_factor", 1.23)),
            max_retries=int(payload.get("max_retries", 64)),
        )
        inst._build_seed = int(payload.get("build_seed", inst.hash_seed))
        inst._effective_size_factor = float(payload.get("effective_size_factor", inst.size_factor))
        inst._array_length = int(payload.get("array_length", 0))
        inst._inserted_count = int(payload.get("inserted_count", 0))
        inst._build_attempts = int(payload.get("build_attempts", 0))
        inst._fingerprints = [int(v) for v in payload.get("fingerprints", [])]
        return inst


class XorFilter(AMQFilter):
    """Static build-once/query-many XOR filter facade."""

    FILTER_NAME = "xor"

    def __init__(
        self,
        *,
        fingerprint_bits: int = 8,
        backend: str = "auto",
        hash_seed: int = 0,
        size_factor: float = 1.23,
        max_retries: int = 64,
    ) -> None:
        self.fingerprint_bits = fingerprint_bits
        self.requested_backend = backend
        self.hash_seed = hash_seed
        self.size_factor = size_factor
        self.max_retries = max_retries

        self._backend_name = "uninitialized"
        self._native_handle: _NativeBackendHandle | None = None
        self._native_obj: Any | None = None
        self._python_backend = _PythonXorBackend(
            fingerprint_bits=fingerprint_bits,
            hash_seed=hash_seed,
            size_factor=size_factor,
            max_retries=max_retries,
        )

        self._inserted_count = 0
        self._build_time_seconds: float | None = None
        self._built = False

    def _try_build_native(self, keys: Sequence[str]) -> bool:
        """Best-effort native backend discovery."""
        candidates: list[tuple[str, str]] = [
            ("pyxorfilter", "Xor8"),
            ("xorfilter", "Xor8"),
            ("xorfilter", "XorFilter"),
        ]

        for module_name, class_name in candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            cls = getattr(module, class_name, None)
            if cls is None:
                continue

            try:
                obj = cls(keys)
                if hasattr(obj, "contains") or hasattr(obj, "__contains__"):
                    self._native_obj = obj
                    self._native_handle = _NativeBackendHandle(module_name, class_name)
                    self._backend_name = f"native:{module_name}.{class_name}"
                    return True
            except Exception:
                pass

            try:
                obj = cls()
                build_fn = getattr(obj, "build", None)
                if callable(build_fn):
                    build_fn(keys)
                    if hasattr(obj, "contains") or hasattr(obj, "__contains__"):
                        self._native_obj = obj
                        self._native_handle = _NativeBackendHandle(module_name, class_name)
                        self._backend_name = f"native:{module_name}.{class_name}"
                        return True
            except Exception:
                pass

        return False

    def build(self, keys: Sequence[str]) -> None:
        """Build or rebuild the static filter from keys."""
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise ValueError("keys must be non-empty")

        start = time.perf_counter()

        built_native = False
        if self.requested_backend == "auto":
            built_native = self._try_build_native(unique_keys)
        elif self.requested_backend == "native":
            built_native = self._try_build_native(unique_keys)
            if not built_native:
                raise RuntimeError(
                    "No compatible native XOR backend found. "
                    "Install and configure a supported package or use backend='fallback'."
                )

        if not built_native:
            self._python_backend = _PythonXorBackend(
                fingerprint_bits=self.fingerprint_bits,
                hash_seed=self.hash_seed,
                size_factor=self.size_factor,
                max_retries=self.max_retries,
            )
            self._python_backend.build(unique_keys)
            self._native_handle = None
            self._native_obj = None
            self._backend_name = "python:xor"

        self._inserted_count = len(unique_keys)
        self._build_time_seconds = time.perf_counter() - start
        self._built = True

    def contains(self, key: str) -> bool:
        """Return membership decision for one key."""
        if self._native_obj is not None:
            contains_fn = getattr(self._native_obj, "contains", None)
            if callable(contains_fn):
                return bool(contains_fn(key))
            return bool(key in self._native_obj)

        return self._python_backend.contains(key)

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        """Return membership decisions for many keys."""
        return [self.contains(key) for key in keys]

    def memory_usage_bytes(self) -> int:
        """Return approximate raw fingerprint-array memory in bytes."""
        if self._native_obj is not None:
            return math.ceil(self._inserted_count * self.fingerprint_bits / 8)
        return self._python_backend.memory_usage_bytes()

    def save(self, path: str | Path) -> None:
        """Serialize filter artifact to JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "filter_name": self.FILTER_NAME,
            "fingerprint_bits": self.fingerprint_bits,
            "requested_backend": self.requested_backend,
            "backend_name": self._backend_name,
            "hash_seed": self.hash_seed,
            "size_factor": self.size_factor,
            "max_retries": self.max_retries,
            "inserted_count": self._inserted_count,
            "build_time_seconds": self._build_time_seconds,
            "python_payload": self._python_backend.to_payload(),
            "native_handle": (
                {
                    "module_name": self._native_handle.module_name,
                    "class_name": self._native_handle.class_name,
                }
                if self._native_handle is not None
                else None
            ),
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "XorFilter":
        """Load serialized XOR facade artifact."""
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"XOR filter artifact not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("filter_name") != cls.FILTER_NAME:
            raise ValueError(
                f"Unexpected filter artifact type: {payload.get('filter_name')}"
            )

        filt = cls(
            fingerprint_bits=int(payload["fingerprint_bits"]),
            backend=str(payload.get("requested_backend", "auto")),
            hash_seed=int(payload.get("hash_seed", 0)),
            size_factor=float(payload.get("size_factor", 1.23)),
            max_retries=int(payload.get("max_retries", 64)),
        )
        filt._backend_name = str(payload.get("backend_name", "python:xor"))
        filt._inserted_count = int(payload.get("inserted_count", 0))
        filt._build_time_seconds = payload.get("build_time_seconds")

        python_payload = payload.get("python_payload")
        if isinstance(python_payload, dict):
            filt._python_backend = _PythonXorBackend.from_payload(python_payload)
        else:
            raise ValueError("XOR artifact does not contain a Python XOR payload")

        filt._built = True
        return filt

    def stats(self) -> dict[str, Any]:
        """Return implementation metadata and run statistics."""
        parameters: dict[str, Any] = {
            "fingerprint_bits": self.fingerprint_bits,
            "backend": self._backend_name,
            "requested_backend": self.requested_backend,
        }
        if self._native_obj is None:
            parameters.update(
                {
                    "array_length": self._python_backend._array_length,
                    "size_factor": self._python_backend._effective_size_factor,
                    "build_attempts": self._python_backend._build_attempts,
                    "hash_seed": self._python_backend._build_seed,
                }
            )

        metadata = FilterBuildMetadata(
            filter_name=self.FILTER_NAME,
            parameters=parameters,
            inserted_keys=self._inserted_count,
            target_false_positive_rate=2 ** (-self.fingerprint_bits),
            actual_memory_usage_bytes=self.memory_usage_bytes(),
            build_time_seconds=self._build_time_seconds,
        )
        out = metadata.to_dict()
        out.update(
            {
                "built": self._built,
                "note": (
                    "pure-Python peel-based XOR backend active"
                    if self._native_obj is None
                    else "native backend active"
                ),
            }
        )
        return out
