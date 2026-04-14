"""Benchmarking package for AMQ experimental framework."""

from benchmarking.experiment_config import ExperimentConfig

__all__ = ["ExperimentConfig", "run_and_save", "run_single_benchmark"]


def run_single_benchmark(*args, **kwargs):
	"""Lazy import wrapper to avoid package import cycles."""
	from benchmarking.benchmark_runner import run_single_benchmark as _run_single_benchmark

	return _run_single_benchmark(*args, **kwargs)


def run_and_save(*args, **kwargs):
	"""Lazy import wrapper to avoid package import cycles."""
	from benchmarking.benchmark_runner import run_and_save as _run_and_save

	return _run_and_save(*args, **kwargs)
