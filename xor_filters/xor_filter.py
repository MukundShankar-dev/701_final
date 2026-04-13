"""XOR filter facade with optional native backend and Python fallback.

A fully correct and high-performance XOR filter is typically implemented in
native code. This module provides:
1) a thin wrapper path for optional external backends, and
2) a deterministic Python fallback backend for framework integration/testing.

The fallback is *not* a true XOR filter data structure; it approximates the
AMQ behavior with fixed-size fingerprints to keep interface compatibility.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from benchmarking.interfaces import AMQFilter, FilterBuildMetadata


@dataclass(slots=True)
class _NativeBackendHandle:
    module_name: str
    class_name: str


class _FingerprintSetFallback:
    """Static fingerprint-set fallback backend.

    This is not a true XOR filter; it serves as a deterministic placeholder
    backend until a native XOR filter package is selected.
    """

    def __init__(self, fingerprint_bits: int) -> None:
        if fingerprint_bits <= 1:
            raise ValueError("fingerprint_bits must be > 1")
        self.fingerprint_bits = fingerprint_bits
        self._mask = (1 << fingerprint_bits) - 1
        self._fps: set[int] = set()

    def _fp(self, key: str) -> int:
        raw = int.from_bytes(
            hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little"
        )
        out = raw & self._mask
        return out if out != 0 else 1

    def build(self, keys: Sequence[str]) -> None:
        self._fps = {self._fp(key) for key in keys}

    def contains(self, key: str) -> bool:
        return self._fp(key) in self._fps

    def to_payload(self) -> dict[str, Any]:
        return {
            "fingerprint_bits": self.fingerprint_bits,
            "fingerprints": sorted(self._fps),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> _FingerprintSetFallback:
        inst = cls(fingerprint_bits=int(payload["fingerprint_bits"]))
        inst._fps = set(map(int, payload["fingerprints"]))
        return inst

    @property
    def size(self) -> int:
        return len(self._fps)


class XorFilter(AMQFilter):
    """Static build-once/query-many XOR filter facade."""

    FILTER_NAME = "xor"

    def __init__(
        self,
        *,
        fingerprint_bits: int = 8,
        backend: str = "auto",
    ) -> None:
        self.fingerprint_bits = fingerprint_bits
        self.requested_backend = backend

        self._backend_name = "uninitialized"
        self._native_handle: _NativeBackendHandle | None = None
        self._native_obj: Any | None = None
        self._fallback = _FingerprintSetFallback(fingerprint_bits=fingerprint_bits)

        self._inserted_count = 0
        self._build_time_seconds: float | None = None
        self._built = False

    def _try_build_native(self, keys: Sequence[str]) -> bool:
        """Best-effort native backend discovery.

        Supported probes are intentionally conservative. If no known compatible
        package API is found, the function returns False.
        """
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

            # Probe two common patterns:
            # pattern A: ctor accepts keys
            # pattern B: empty ctor plus .build(keys)
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
        start = time.perf_counter()

        built_native = False
        if self.requested_backend == "auto":
            built_native = self._try_build_native(keys)
        elif self.requested_backend == "native":
            built_native = self._try_build_native(keys)
            if not built_native:
                raise RuntimeError(
                    "No compatible native XOR backend found. "
                    "Install and configure a supported package or use backend='fallback'."
                )

        if not built_native:
            self._fallback.build(keys)
            self._native_handle = None
            self._native_obj = None
            self._backend_name = "fallback:fingerprint_set"

        self._inserted_count = len(keys)
        self._build_time_seconds = time.perf_counter() - start
        self._built = True

    def contains(self, key: str) -> bool:
        """Return membership decision for one key."""
        if self._native_obj is not None:
            contains_fn = getattr(self._native_obj, "contains", None)
            if callable(contains_fn):
                return bool(contains_fn(key))
            return bool(key in self._native_obj)

        return self._fallback.contains(key)

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        """Return membership decisions for many keys."""
        return [self.contains(key) for key in keys]

    def memory_usage_bytes(self) -> int:
        """Return approximate in-memory usage in bytes.

        For native backend this is estimated and may be optimistic.
        """
        if self._native_obj is not None:
            return int(self._inserted_count * self.fingerprint_bits / 8)
        return int(max(1, self._fallback.size) * self.fingerprint_bits / 8)

    def save(self, path: str | Path) -> None:
        """Serialize filter artifact to JSON.

        Native backend artifacts are not currently persisted directly; saving in
        native mode captures wrapper metadata only and requires rebuild on load.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "filter_name": self.FILTER_NAME,
            "fingerprint_bits": self.fingerprint_bits,
            "requested_backend": self.requested_backend,
            "backend_name": self._backend_name,
            "inserted_count": self._inserted_count,
            "build_time_seconds": self._build_time_seconds,
            "fallback_payload": self._fallback.to_payload(),
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
    def load(cls, path: str | Path) -> XorFilter:
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
        )
        filt._backend_name = str(payload.get("backend_name", "fallback:fingerprint_set"))
        filt._inserted_count = int(payload.get("inserted_count", 0))
        filt._build_time_seconds = payload.get("build_time_seconds")
        filt._fallback = _FingerprintSetFallback.from_payload(payload["fallback_payload"])
        filt._built = True
        return filt

    def stats(self) -> dict[str, Any]:
        """Return implementation metadata and run statistics."""
        metadata = FilterBuildMetadata(
            filter_name=self.FILTER_NAME,
            parameters={
                "fingerprint_bits": self.fingerprint_bits,
                "backend": self._backend_name,
                "requested_backend": self.requested_backend,
            },
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
                    "fallback backend is not a true XOR filter; use native backend "
                    "for publication-quality XOR comparisons"
                    if self._backend_name.startswith("fallback")
                    else "native backend active"
                ),
            }
        )
        return out
