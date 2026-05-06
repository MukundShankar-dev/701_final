# AMQ Benchmarking Framework for Genomic k-mers

This repository provides a modular, reproducible Python framework for benchmarking approximate membership query (AMQ) structures on genomic k-mer sets.

Implemented filter families:
- Bloom filters
- Cuckoo filters
- XOR filter facade (native-backend hook + deterministic Python fallback)
- Learned filter prototype (classifier + backup Bloom filter)

## Project Goals

The framework is designed for reproducible experiments comparing:
- memory per k-mer
- empirical false positive rate
- query speed
- timing-based cache behavior proxies

It supports genomic-like data workflows including FASTA-based k-mer extraction and synthetic dataset generation.

For real bacterial FASTA downloads and Jellyfish-based k-mer extraction, see
`JELLYFISH_DATA_WORKFLOW.md`.

## Repository Layout

- bloom_filters/: Bloom filter implementation, builder, CLI
- cuckoo_filters/: Cuckoo filter implementation, builder, CLI
- xor_filters/: XOR facade, builder, CLI
- learned_filters/: learned filter prototype and training pipeline
- data/: FASTA loader, k-mer extraction, synthetic data, I/O helpers
- benchmarking/: shared interfaces, config, metrics, runner, result serialization
- scripts/: convenience scripts for data generation, builds, and sweep runs
- tests/: pytest suite

## macOS Setup (Python 3.11+)

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run tests:

```bash
pytest -q
```

## Quickstart

### 1) Generate synthetic k-mers

```bash
python scripts/generate_synthetic_data.py \
	--output data/datasets/synth_k21.txt \
	--k 21 \
	--num-contigs 8 \
	--contig-length 50000 \
	--gc-bias 0.52 \
	--seed 7
```

### 2) Build each filter from same input

```bash
python scripts/build_all.py \
	--kmer-file data/datasets/synth_k21.txt \
	--output-dir data/datasets/artifacts_k21 \
	--k 21 \
	--seed 7
```

### 3) Run sweep benchmarks (k, filter family, FPR)

```bash
python scripts/run_benchmarks.py \
	--dataset data/datasets/synth_k21.txt \
	--dataset-name synth_k21 \
	--output-dir benchmarking/results \
	--seed 7 \
	--repetitions 3
```

By default, the script infers k from the dataset file and runs all filter families.
For a fixed k dataset like synth_k21, this avoids invalid k sweeps automatically.

### 4) Plot benchmark results

```bash
python scripts/plot_results.py \
	--results-dir benchmarking/results \
	--output-dir benchmarking/results/plots
```

This generates cross-k plots using the nearest target FPR per filter/k around
reference 1e-3, plus a dedicated achieved-vs-target FPR sweep plot.

## CLI Examples by Subsystem

### Bloom

```bash
python -m bloom_filters.cli build \
	--kmer-file data/datasets/synth_k21.txt \
	--output data/datasets/bloom_k21.json \
	--fpr 1e-3
```

### Cuckoo

```bash
python -m cuckoo_filters.cli build \
	--kmer-file data/datasets/synth_k21.txt \
	--output data/datasets/cuckoo_k21.json \
	--fingerprint-bits 12
```

### XOR

```bash
python -m xor_filters.cli build \
	--kmer-file data/datasets/synth_k21.txt \
	--output data/datasets/xor_k21.json \
	--fingerprint-bits 8 \
	--backend auto
```

### Learned

```bash
python -m learned_filters.cli train \
	--kmer-file data/datasets/synth_k21.txt \
	--output data/datasets/learned_k21 \
	--k 21 \
	--backup-fpr 1e-3
```

### Benchmark Runner CLI

```bash
python -m benchmarking.cli run --config path/to/experiment_config.json
```

Use this when you want strict, explicit control for one experiment configuration.
Use scripts/run_benchmarks.py when you want automatic multi-filter sweeps.

## Which commands are required?

- scripts/generate_synthetic_data.py: required to create input k-mers (unless you already have them).
- scripts/run_benchmarks.py: enough to run all implemented filters on that dataset.
- scripts/build_all.py: optional convenience step to build and inspect standalone artifacts.
- per-filter module CLIs (python -m bloom_filters.cli, etc.): optional, useful for targeted debugging.
- python -m benchmarking.cli run --config ...: optional single-config runner for reproducible custom experiments.

## Experiment Config Format

Use JSON matching benchmarking.ExperimentConfig:

```json
{
	"dataset_name": "synth_k21",
	"dataset_path": "data/datasets/synth_k21.txt",
	"k": 21,
	"canonicalize": false,
	"filter_type": "bloom",
	"filter_params": {
		"false_positive_rate": 0.001
	},
	"positive_query_count": 10000,
	"negative_query_count": 10000,
	"random_seed": 7,
	"output_directory": "benchmarking/results/synth_k21/bloom",
	"repetitions": 3
}
```

## Benchmark Outputs

For each run:
- one JSON result file in output directory

Across runs:
- aggregate_results.csv (appended rows with comparable schema)

Key fields include:
- build_time_seconds
- memory_usage_bytes
- memory_per_kmer_bytes
- true_positive_rate
- false_positive_rate
- throughput_qps
- avg/p50/p95/p99 latency (microseconds)
- cache proxy metrics (sequential/random/repeated query throughput)

Default plot outputs:
- false_positive_rate_by_filter.png
- throughput_by_filter.png
- memory_per_kmer_by_filter.png
- build_time_by_filter.png
- fpr_vs_target.png

## Notes on Pure-Python Limitations

- XOR fallback backend is a placeholder AMQ-compatible implementation, not a mathematically faithful XOR filter construction.
- Hardware cache counters are not collected directly; cache_proxy.py reports timing-based proxies.
- Python memory accounting is approximate; compare metrics consistently across methods and include serialized artifact sizes when possible.

## Reproducibility Guidance

- Always set seeds for data generation and filter construction.
- Keep k-mer input files immutable once generated for fair comparisons.
- Run all methods on identical key/query sets when comparing FPR and throughput.
- Repeat runs and inspect variance across repetitions.

## Suggested Next Optimization Steps

- Add a native XOR backend package and wire it through xor_filters/xor_filter.py.
- Add optional C/C++ or Rust-backed Cuckoo/Bloom variants behind same interfaces.
- Extend result aggregation with confidence intervals and statistical tests.
- Add plotting notebooks or scripts for publication-ready figures.
