"""Experiment configuration dataclasses and loaders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.io_utils import load_json, save_json


@dataclass(slots=True)
class ExperimentConfig:
    dataset_name: str
    dataset_path: str
    k: int
    canonicalize: bool
    filter_type: str
    filter_params: dict[str, Any] = field(default_factory=dict)
    positive_query_count: int = 10_000
    negative_query_count: int = 10_000
    random_seed: int = 0
    output_directory: str = "benchmarking/results"
    repetitions: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        save_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        payload = load_json(path)
        return cls.from_dict(payload)
