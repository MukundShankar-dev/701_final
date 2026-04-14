"""Learned filter prototype combining classifier and backup Bloom filter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from benchmarking.interfaces import AMQFilter, FilterBuildMetadata
from learned_filters.backup_filter import BackupBloomFilter
from learned_filters.dataset import LabeledKmerDataset, build_training_dataset, split_dataset
from learned_filters.model import KmerLogisticModel


class LearnedFilter(AMQFilter):
    """Two-stage learned filter:

    1) classifier predicts likely positives,
    2) if classifier predicts negative, consult backup Bloom filter.
    """

    FILTER_NAME = "learned"

    def __init__(
        self,
        *,
        k: int,
        model_threshold: float = 0.5,
        backup_false_positive_rate: float = 1e-3,
        random_seed: int = 0,
    ) -> None:
        self.k = k
        self.model_threshold = model_threshold
        self.backup_false_positive_rate = backup_false_positive_rate
        self.random_seed = random_seed

        self.model = KmerLogisticModel(
            k=k,
            random_seed=random_seed,
            threshold=model_threshold,
        )
        self.backup_filter: BackupBloomFilter | None = None

        self._inserted_count = 0
        self._build_time_seconds: float | None = None
        self._last_eval: dict[str, float] = {}
        self._built = False

    def train(
        self,
        positive_kmers: Sequence[str],
        *,
        negative_count: int | None = None,
        negative_mutation_rate: float = 0.2,
    ) -> dict[str, float]:
        """Train classifier, create backup filter from model false negatives."""
        start = time.perf_counter()

        dataset = build_training_dataset(
            positive_kmers=positive_kmers,
            negative_count=negative_count,
            negative_mutation_rate=negative_mutation_rate,
            random_seed=self.random_seed,
        )
        train_set, val_set, _ = split_dataset(dataset)

        self.model.fit(train_set.kmers, train_set.labels)

        val_positive = [k for k, y in zip(val_set.kmers, val_set.labels, strict=True) if y == 1]
        val_negative = [k for k, y in zip(val_set.kmers, val_set.labels, strict=True) if y == 0]

        tuning = self.model.tune_threshold(
            positive_kmers=val_positive,
            negative_kmers=val_negative,
            target_model_fpr=self.backup_false_positive_rate,
        )
        self.model_threshold = self.model.threshold

        eval_metrics = self.model.evaluate(val_set.kmers, val_set.labels)

        positives = [k.upper() for k in positive_kmers]
        preds = self.model.predict(positives)
        model_false_negatives = [k for k, pred in zip(positives, preds, strict=True) if pred == 0]

        self.backup_filter = BackupBloomFilter.build(
            model_false_negatives,
            false_positive_rate=self.backup_false_positive_rate,
            hash_seed=self.random_seed,
        )

        self._inserted_count = len(positives)
        self._build_time_seconds = time.perf_counter() - start
        self._last_eval = {
            **eval_metrics,
            **tuning,
            "model_false_negative_count": float(len(model_false_negatives)),
            "model_false_negative_rate": float(len(model_false_negatives) / max(1, len(positives))),
        }
        self._built = True
        return dict(self._last_eval)

    def build(self, keys: Sequence[str]) -> None:
        """Build learned filter from positive keys only."""
        self.train(positive_kmers=keys)

    def contains(self, key: str) -> bool:
        """Query learned filter inference path."""
        pred = int(self.model.predict([key])[0])
        if pred == 1:
            return True
        if self.backup_filter is None:
            return False
        return self.backup_filter.contains(key)

    def batch_contains(self, keys: Sequence[str]) -> list[bool]:
        query_list = list(keys)
        if not query_list:
            return []

        preds = self.model.predict(query_list)
        out: list[bool] = []
        for key, pred in zip(query_list, preds, strict=True):
            if int(pred) == 1:
                out.append(True)
            elif self.backup_filter is not None:
                out.append(self.backup_filter.contains(key))
            else:
                out.append(False)
        return out

    def memory_usage_bytes(self) -> int:
        # Rough estimate: sklearn model params are not trivial to size precisely.
        backup_bytes = self.backup_filter.bloom.memory_usage_bytes() if self.backup_filter else 0
        model_estimate = 8 * (self.k * 4 + 1 + 6)
        return int(model_estimate + backup_bytes)

    def evaluate_heldout(self, dataset: LabeledKmerDataset) -> dict[str, float]:
        """Evaluate classifier-only metrics on held-out dataset."""
        return self.model.evaluate(dataset.kmers, dataset.labels)

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model_path = out_path.with_suffix(".model.pkl")
        backup_path = out_path.with_suffix(".backup.json")
        meta_path = out_path.with_suffix(".meta.json")

        self.model.save(model_path)
        if self.backup_filter is not None:
            self.backup_filter.save(backup_path)

        payload: dict[str, Any] = {
            "filter_name": self.FILTER_NAME,
            "k": self.k,
            "model_threshold": self.model_threshold,
            "backup_false_positive_rate": self.backup_false_positive_rate,
            "random_seed": self.random_seed,
            "inserted_count": self._inserted_count,
            "build_time_seconds": self._build_time_seconds,
            "last_eval": self._last_eval,
            "model_path": model_path.name,
            "backup_path": backup_path.name,
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedFilter":
        base = Path(path)
        meta_path = base.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Learned filter metadata not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        inst = cls(
            k=int(payload["k"]),
            model_threshold=float(payload["model_threshold"]),
            backup_false_positive_rate=float(payload["backup_false_positive_rate"]),
            random_seed=int(payload.get("random_seed", 0)),
        )

        model_path = meta_path.parent / payload["model_path"]
        backup_path = meta_path.parent / payload["backup_path"]

        inst.model = KmerLogisticModel.load(model_path)
        if backup_path.exists():
            inst.backup_filter = BackupBloomFilter.load(backup_path)

        inst._inserted_count = int(payload.get("inserted_count", 0))
        inst._build_time_seconds = payload.get("build_time_seconds")
        inst._last_eval = dict(payload.get("last_eval", {}))
        inst._built = True
        return inst

    def stats(self) -> dict[str, Any]:
        metadata = FilterBuildMetadata(
            filter_name=self.FILTER_NAME,
            parameters={
                "k": self.k,
                "model_threshold": self.model_threshold,
                "backup_false_positive_rate": self.backup_false_positive_rate,
            },
            inserted_keys=self._inserted_count,
            target_false_positive_rate=self.backup_false_positive_rate,
            actual_memory_usage_bytes=self.memory_usage_bytes(),
            build_time_seconds=self._build_time_seconds,
        )
        out = metadata.to_dict()
        out.update({
            "built": self._built,
            "model_eval": self._last_eval,
            "has_backup_filter": self.backup_filter is not None,
        })
        return out
