"""Feature extraction and classifier wrappers for learned filters."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize


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


@dataclass(slots=True)
class KmerCompositionExtractor:
    """Small dense feature set for fast composition-based classifiers."""

    k: int

    def transform(self, kmers: Sequence[str]) -> np.ndarray:
        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        rows: list[list[float]] = []
        for kmer in kmers:
            kk = kmer.upper()
            if len(kk) != self.k:
                raise ValueError(f"Expected k-mer length {self.k}, got {len(kk)} for {kmer}")

            bases = [base_to_idx.get(base, -1) for base in kk]
            mono = [0.0] * 4
            dinuc = [0.0] * 16
            invalid = 0.0
            longest_run = 0
            current_run = 0
            previous = ""

            for idx, base in enumerate(kk):
                code = bases[idx]
                if code < 0:
                    invalid += 1.0
                else:
                    mono[code] += 1.0

                if base == previous:
                    current_run += 1
                else:
                    current_run = 1
                    previous = base
                longest_run = max(longest_run, current_run)

                if idx > 0 and bases[idx - 1] >= 0 and code >= 0:
                    dinuc[bases[idx - 1] * 4 + code] += 1.0

            mono_denom = max(1, self.k)
            dinuc_denom = max(1, self.k - 1)
            gc_content = (mono[1] + mono[2]) / mono_denom

            row = [v / mono_denom for v in mono]
            row.extend(v / dinuc_denom for v in dinuc)
            row.extend([
                gc_content,
                longest_run / mono_denom,
                invalid / mono_denom,
            ])
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


@dataclass(slots=True)
class DnaNgramVectorizer:
    """Fast sparse DNA n-gram vectorizer using 2-bit rolling encodings."""

    k: int
    ngram_range: tuple[int, int] = (3, 5)
    normalize_rows: bool = True

    def __post_init__(self) -> None:
        lo, hi = self.ngram_range
        if lo <= 0 or hi < lo:
            raise ValueError(f"Invalid ngram_range: {self.ngram_range}")
        if hi > self.k:
            raise ValueError(f"ngram_range upper bound {hi} exceeds k={self.k}")

    @property
    def n_features(self) -> int:
        lo, hi = self.ngram_range
        return sum(4**n for n in range(lo, hi + 1))

    def _offsets(self) -> dict[int, int]:
        out: dict[int, int] = {}
        offset = 0
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            out[n] = offset
            offset += 4**n
        return out

    def transform(self, kmers: Sequence[str]) -> sparse.csr_matrix:
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        offsets = self._offsets()
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        lo, hi = self.ngram_range
        for row_idx, kmer in enumerate(kmers):
            kk = kmer.upper()
            if len(kk) != self.k:
                raise ValueError(f"Expected k-mer length {self.k}, got {len(kk)} for {kmer}")

            codes = [base_map.get(base, -1) for base in kk]
            if any(code < 0 for code in codes):
                continue

            for n in range(lo, hi + 1):
                code = 0
                mask = (1 << (2 * n)) - 1
                for i, base_code in enumerate(codes):
                    code = ((code << 2) | base_code) & mask
                    if i + 1 >= n:
                        rows.append(row_idx)
                        cols.append(offsets[n] + code)
                        data.append(1.0)

        mat = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(len(kmers), self.n_features),
            dtype=np.float32,
        )
        mat.sum_duplicates()
        if self.normalize_rows:
            normalize(mat, norm="l2", copy=False)
        return mat


class PrefixSetClassifier:
    """Classifier-like prefix gate used as an experimental learned-filter backend."""

    def __init__(self, *, k: int, prefix_length: int) -> None:
        if not 0 < prefix_length <= k:
            raise ValueError(f"prefix_length must be in [1, {k}], got {prefix_length}")
        self.k = k
        self.prefix_length = prefix_length
        self.prefixes: set[str] = set()
        self.classes_ = np.asarray([0, 1], dtype=np.int32)

    def fit(self, X: Sequence[str], y: Sequence[int]) -> "PrefixSetClassifier":
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        self.prefixes = {
            kmer[: self.prefix_length]
            for kmer, label in zip(X, y, strict=True)
            if int(label) == 1
        }
        return self

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        probs = np.zeros((len(X), 2), dtype=np.float32)
        for i, kmer in enumerate(X):
            positive = kmer[: self.prefix_length] in self.prefixes
            probs[i, 1] = 1.0 if positive else 0.0
            probs[i, 0] = 1.0 - probs[i, 1]
        return probs

    def memory_usage_bytes(self) -> int:
        return sys.getsizeof(self.prefixes) + sum(sys.getsizeof(prefix) for prefix in self.prefixes)


class KmerLogisticModel:
    """Classifier wrapper for k-mers.

    Backends:
    - ``position_logistic``: original dense position-wise one-hot logistic model.
    - ``composition_logistic``: compact dense composition logistic model.
    - ``dna_ngram_sgd``: sparse DNA n-gram logistic model trained with SGD.
    - ``ngram_sgd``: sklearn text HashingVectorizer backend kept for comparison.
    - ``ngram_nb``: sparse hashed n-gram Multinomial Naive Bayes baseline.
    - ``prefix_set``: non-parametric prefix gate for experimental comparison.
    """

    def __init__(
        self,
        k: int,
        *,
        random_seed: int = 0,
        model_backend: str = "ngram_sgd",
        max_iter: int = 40,
        C: float = 1.0,
        alpha: float = 1e-5,
        threshold: float = 0.5,
        ngram_range: tuple[int, int] = (3, 5),
        ngram_features: int = 4096,
    ) -> None:
        self.k = k
        self.threshold = threshold
        self.random_seed = random_seed
        self.model_backend = model_backend
        self.max_iter = max_iter
        self.C = C
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.ngram_features = ngram_features

        self.extractor: KmerFeatureExtractor | None = None
        self.composition_extractor: KmerCompositionExtractor | None = None
        self.vectorizer: HashingVectorizer | DnaNgramVectorizer | None = None

        if model_backend == "position_logistic":
            self.extractor = KmerFeatureExtractor(k=k)
            self.model: Any = LogisticRegression(
                random_state=random_seed,
                max_iter=max_iter,
                C=C,
                solver="lbfgs",
            )
        elif model_backend == "composition_logistic":
            self.composition_extractor = KmerCompositionExtractor(k=k)
            self.model = LogisticRegression(
                random_state=random_seed,
                max_iter=max_iter,
                C=C,
                solver="lbfgs",
                class_weight="balanced",
            )
        elif model_backend == "dna_ngram_sgd":
            self.vectorizer = DnaNgramVectorizer(
                k=k,
                ngram_range=ngram_range,
            )
            self.model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=max_iter,
                tol=1e-3,
                random_state=random_seed,
                class_weight="balanced",
            )
        elif model_backend == "ngram_nb":
            self.vectorizer = HashingVectorizer(
                analyzer="char",
                ngram_range=ngram_range,
                n_features=ngram_features,
                alternate_sign=False,
                lowercase=False,
                norm=None,
                dtype=np.float32,
            )
            self.model = MultinomialNB(alpha=alpha)
        elif model_backend == "ngram_sgd":
            self.vectorizer = HashingVectorizer(
                analyzer="char",
                ngram_range=ngram_range,
                n_features=ngram_features,
                alternate_sign=False,
                lowercase=False,
                norm="l2",
                dtype=np.float32,
            )
            self.model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=max_iter,
                tol=1e-3,
                random_state=random_seed,
                class_weight="balanced",
            )
        elif model_backend == "prefix_set":
            self.model = PrefixSetClassifier(k=k, prefix_length=max(1, min(k, ngram_range[0])))
        else:
            raise ValueError(
                "Unsupported model_backend; expected 'composition_logistic', 'dna_ngram_sgd', "
                "'ngram_sgd', 'ngram_nb', 'prefix_set', or 'position_logistic', "
                f"got {model_backend!r}"
            )
        self._trained = False

    def _normalize_kmers(self, kmers: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        for kmer in kmers:
            kk = kmer.upper()
            if len(kk) != self.k:
                raise ValueError(f"Expected k-mer length {self.k}, got {len(kk)} for {kmer}")
            normalized.append(kk)
        return normalized

    def _transform(self, kmers: Sequence[str]) -> Any:
        normalized = self._normalize_kmers(kmers)
        if self.model_backend == "prefix_set":
            return normalized

        if self.model_backend == "position_logistic":
            if self.extractor is None:
                raise RuntimeError("Position feature extractor is not initialized")
            return self.extractor.transform(normalized)

        if self.model_backend == "composition_logistic":
            if self.composition_extractor is None:
                raise RuntimeError("Composition feature extractor is not initialized")
            return self.composition_extractor.transform(normalized)

        if self.vectorizer is None:
            raise RuntimeError("N-gram vectorizer is not initialized")
        return self.vectorizer.transform(normalized)

    def fit(self, kmers: Sequence[str], labels: Sequence[int]) -> None:
        X = self._transform(kmers)
        y = np.asarray(labels, dtype=np.int32)
        row_count = len(X) if self.model_backend == "prefix_set" else X.shape[0]
        if row_count != y.shape[0]:
            raise ValueError("kmers and labels length mismatch")
        self.model.fit(X, y)
        self._trained = True

    def predict_proba(self, kmers: Sequence[str]) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model is not trained")
        X = self._transform(kmers)
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

        avg_precision = (
            float(average_precision_score(y_true, probs))
            if np.any(positives)
            else float("nan")
        )

        out: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "avg_precision": avg_precision,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "threshold": float(self.threshold),
        }
        if np.any(positives) and np.any(negatives):
            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        else:
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

    def memory_usage_bytes(self) -> int:
        """Return a compact estimate of learned model parameters."""
        if hasattr(self.model, "memory_usage_bytes"):
            return int(self.model.memory_usage_bytes())

        total = 0
        for attr in ("coef_", "intercept_", "classes_"):
            value = getattr(self.model, attr, None)
            if hasattr(value, "nbytes"):
                total += int(value.nbytes)
        # HashingVectorizer is stateless; include a small fixed accounting term
        # for backend metadata so tiny models do not report zero bytes.
        return max(total, 64)

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "k": self.k,
            "threshold": self.threshold,
            "random_seed": self.random_seed,
            "model_backend": self.model_backend,
            "max_iter": self.max_iter,
            "C": self.C,
            "alpha": self.alpha,
            "ngram_range": self.ngram_range,
            "ngram_features": self.ngram_features,
            "extractor": self.extractor,
            "composition_extractor": self.composition_extractor,
            "vectorizer": self.vectorizer,
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

        inst = cls(
            k=int(payload["k"]),
            threshold=float(payload["threshold"]),
            random_seed=int(payload.get("random_seed", 0)),
            model_backend=str(payload.get("model_backend", "position_logistic")),
            max_iter=int(payload.get("max_iter", 400)),
            C=float(payload.get("C", 1.0)),
            alpha=float(payload.get("alpha", 1e-5)),
            ngram_range=tuple(payload.get("ngram_range", (3, 5))),
            ngram_features=int(payload.get("ngram_features", 4096)),
        )
        inst.extractor = payload.get("extractor")
        inst.composition_extractor = payload.get("composition_extractor")
        inst.vectorizer = payload.get("vectorizer")
        inst.model = payload["model"]
        inst._trained = bool(payload["trained"])
        return inst
