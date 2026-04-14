"""Feature extraction and classifier wrappers for learned filters."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass(slots=True)
class KmerFeatureExtractor:
    """Simple explicit feature extractor for DNA k-mers.

    Features:
    1) Position-wise one-hot encoding for A/C/G/T.
    2) GC content.
    3) Motif counts for short dinucleotide motifs.
    """

    k: int
    motifs: tuple[str, ...] = ("CG", "GC", "AT", "TA", "AA", "TT")

    def _encode_one_hot(self, kmer: str) -> list[float]:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        out = [0.0] * (self.k * 4)
        for i, base in enumerate(kmer):
            idx = mapping.get(base)
            if idx is None:
                continue
            out[i * 4 + idx] = 1.0
        return out

    def _gc_content(self, kmer: str) -> float:
        if not kmer:
            return 0.0
        gc = sum(1 for c in kmer if c in {"G", "C"})
        return gc / len(kmer)

    def _motif_features(self, kmer: str) -> list[float]:
        if len(kmer) < 2:
            return [0.0] * len(self.motifs)
        counts = [kmer.count(motif) for motif in self.motifs]
        denom = max(1, len(kmer) - 1)
        return [c / denom for c in counts]

    def transform(self, kmers: Sequence[str]) -> np.ndarray:
        """Transform k-mers into a dense feature matrix."""
        rows: list[list[float]] = []
        for kmer in kmers:
            kk = kmer.upper()
            if len(kk) != self.k:
                raise ValueError(f"Expected k-mer length {self.k}, got {len(kk)} for {kmer}")
            row = self._encode_one_hot(kk)
            row.append(self._gc_content(kk))
            row.extend(self._motif_features(kk))
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


class KmerLogisticModel:
    """Thin wrapper around scikit-learn LogisticRegression for k-mers."""

    def __init__(
        self,
        k: int,
        *,
        random_seed: int = 0,
        max_iter: int = 400,
        C: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        self.k = k
        self.threshold = threshold
        self.extractor = KmerFeatureExtractor(k=k)
        self.model = LogisticRegression(
            random_state=random_seed,
            max_iter=max_iter,
            C=C,
            solver="lbfgs",
        )
        self._trained = False

    def fit(self, kmers: Sequence[str], labels: Sequence[int]) -> None:
        X = self.extractor.transform(kmers)
        y = np.asarray(labels, dtype=np.int32)
        if X.shape[0] != y.shape[0]:
            raise ValueError("kmers and labels length mismatch")
        self.model.fit(X, y)
        self._trained = True

    def predict_proba(self, kmers: Sequence[str]) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model is not trained")
        X = self.extractor.transform(kmers)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, kmers: Sequence[str]) -> np.ndarray:
        probs = self.predict_proba(kmers)
        return (probs >= self.threshold).astype(np.int32)

    def evaluate(self, kmers: Sequence[str], labels: Sequence[int]) -> dict[str, float]:
        y_true = np.asarray(labels, dtype=np.int32)
        probs = self.predict_proba(kmers)
        preds = (probs >= self.threshold).astype(np.int32)

        positives = y_true == 1
        negatives = y_true == 0
        true_positive_rate = float(np.mean(preds[positives] == 1)) if np.any(positives) else 0.0
        false_positive_rate = float(np.mean(preds[negatives] == 1)) if np.any(negatives) else 0.0

        out: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "avg_precision": float(average_precision_score(y_true, probs)),
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "threshold": float(self.threshold),
        }
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        except ValueError:
            out["roc_auc"] = float("nan")
        return out

    def tune_threshold(
        self,
        positive_kmers: Sequence[str],
        negative_kmers: Sequence[str],
        *,
        target_model_fpr: float,
        candidate_count: int = 101,
    ) -> dict[str, float]:
        """Tune threshold from validation probabilities.

        Strategy:
        1) Evaluate candidate thresholds from probability quantiles.
        2) Prefer thresholds meeting ``model_fpr <= target_model_fpr``.
        3) Within feasible thresholds, minimize model false negative rate.
        4) If infeasible, minimize model false positive rate.
        """
        if not 0.0 <= target_model_fpr <= 1.0:
            raise ValueError("target_model_fpr must be in [0, 1]")
        if candidate_count < 3:
            raise ValueError("candidate_count must be >= 3")

        if not positive_kmers or not negative_kmers:
            return {
                "selected_threshold": float(self.threshold),
                "model_fpr": float("nan"),
                "model_fnr": float("nan"),
                "target_model_fpr": float(target_model_fpr),
                "used_fallback_selection": 1.0,
            }

        pos_probs = self.predict_proba(positive_kmers)
        neg_probs = self.predict_proba(negative_kmers)

        combined = np.concatenate([pos_probs, neg_probs])
        quantiles = np.linspace(0.0, 1.0, candidate_count)
        candidates = np.unique(np.quantile(combined, quantiles))
        candidates = np.concatenate(([0.0], candidates, [1.0]))

        feasible: list[tuple[float, float, float]] = []
        all_scores: list[tuple[float, float, float]] = []

        for thr in candidates:
            model_fpr = float(np.mean(neg_probs >= thr))
            model_fnr = float(np.mean(pos_probs < thr))

            all_scores.append((model_fpr, model_fnr, float(thr)))
            if model_fpr <= target_model_fpr:
                feasible.append((model_fpr, model_fnr, float(thr)))

        if feasible:
            # Minimize backup size proxy (FNR) while satisfying model-FPR target.
            chosen_fpr, chosen_fnr, chosen_thr = min(feasible, key=lambda t: (t[1], t[0], -t[2]))
            used_fallback = 0.0
        else:
            chosen_fpr, chosen_fnr, chosen_thr = min(all_scores, key=lambda t: (t[0], t[1], -t[2]))
            used_fallback = 1.0

        self.threshold = float(chosen_thr)
        return {
            "selected_threshold": float(chosen_thr),
            "model_fpr": float(chosen_fpr),
            "model_fnr": float(chosen_fnr),
            "target_model_fpr": float(target_model_fpr),
            "used_fallback_selection": used_fallback,
        }

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "k": self.k,
            "threshold": self.threshold,
            "extractor": self.extractor,
            "model": self.model,
            "trained": self._trained,
        }
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "KmerLogisticModel":
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {in_path}")
        with in_path.open("rb") as handle:
            payload = pickle.load(handle)

        inst = cls(k=int(payload["k"]), threshold=float(payload["threshold"]))
        inst.extractor = payload["extractor"]
        inst.model = payload["model"]
        inst._trained = bool(payload["trained"])
        return inst
