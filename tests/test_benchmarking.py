from pathlib import Path

from benchmarking.benchmark_runner import run_and_save, run_single_benchmark
from benchmarking.experiment_config import ExperimentConfig
from data.io_utils import save_kmers


def test_benchmark_metrics_tiny_dataset(tmp_path: Path) -> None:
    keys = ["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA", "CCCCCC", "GGGGGG"]
    data_path = tmp_path / "tiny.kmers"
    save_kmers(data_path, keys)

    cfg = ExperimentConfig(
        dataset_name="tiny",
        dataset_path=str(data_path),
        k=6,
        canonicalize=False,
        filter_type="bloom",
        filter_params={"false_positive_rate": 1e-3},
        positive_query_count=10,
        negative_query_count=10,
        random_seed=0,
        output_directory=str(tmp_path / "out"),
        repetitions=1,
    )

    result = run_single_benchmark(cfg)
    assert 0.0 <= result.true_positive_rate <= 1.0
    assert 0.0 <= result.false_positive_rate <= 1.0
    assert result.memory_per_kmer_bytes > 0


def test_run_ids_do_not_overwrite_across_param_sweeps(tmp_path: Path) -> None:
    keys = ["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA", "CCCCCC", "GGGGGG"]
    data_path = tmp_path / "tiny.kmers"
    save_kmers(data_path, keys)

    out_dir = tmp_path / "out"

    cfg_loose = ExperimentConfig(
        dataset_name="tiny",
        dataset_path=str(data_path),
        k=6,
        canonicalize=False,
        filter_type="bloom",
        filter_params={"false_positive_rate": 1e-2},
        positive_query_count=8,
        negative_query_count=8,
        random_seed=0,
        output_directory=str(out_dir),
        repetitions=1,
    )
    cfg_strict = ExperimentConfig(
        dataset_name="tiny",
        dataset_path=str(data_path),
        k=6,
        canonicalize=False,
        filter_type="bloom",
        filter_params={"false_positive_rate": 1e-4},
        positive_query_count=8,
        negative_query_count=8,
        random_seed=0,
        output_directory=str(out_dir),
        repetitions=1,
    )

    run_and_save(cfg_loose)
    run_and_save(cfg_strict)

    json_files = sorted(out_dir.glob("*.json"))
    # Two separate parameter configurations should produce two JSON artifacts.
    assert len(json_files) == 2
