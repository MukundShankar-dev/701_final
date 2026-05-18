# AMQ Filters for Genomic k-mers

This repository implements and benchmarks approximate membership query (AMQ) filters on genomic k-mer sets:

- Bloom filter
- Cuckoo filter
- XOR filter
- Learned filter with classifier plus backup Bloom filter

The main benchmark metrics are memory per inserted k-mer, empirical false positive rate, query throughput, build time, latency, and cache-proxy throughput.

## Setup

Use Python 3.10+ from the project environment:

```bash
cd "701_final"

python -m pip install -r requirements.txt
```

If you are using the course conda environment:

```bash
conda activate 701
```

The commands below assume they are run from the repository root.

## Quick Sanity Check

Run tests:

```bash
python -m pytest
```

Expected result:

```text
14 passed
```

## Data Layout

Synthetic k-mer files:

```text
data/datasets/synth_k15.txt
data/datasets/synth_k21.txt
data/datasets/synth_k31.txt
```

Real bacterial datasets:

```text
data/datasets/real/diverse_bacteria_4/k15/manifest.tsv
data/datasets/real/diverse_bacteria_4/k21/manifest.tsv
data/datasets/real/diverse_bacteria_4/k31/manifest.tsv
```

Each real-data manifest points to one-k-mer-per-line `kmers.txt` files. The benchmark runners only need `kmers.txt`; Jellyfish `counts.tsv`, `genome.jf`, and metadata files are useful for traceability but are not required to run benchmarks.

## Regenerate Synthetic Data

The final synthetic datasets were generated from 8 contigs of length 50,000 with GC bias 0.52 and seed 7.

```bash
python scripts/generate_synthetic_data.py \
  --output data/datasets/synth_k15.txt \
  --k 15 \
  --num-contigs 8 \
  --contig-length 50000 \
  --gc-bias 0.52 \
  --seed 7

python scripts/generate_synthetic_data.py \
  --output data/datasets/synth_k21.txt \
  --k 21 \
  --num-contigs 8 \
  --contig-length 50000 \
  --gc-bias 0.52 \
  --seed 7

python scripts/generate_synthetic_data.py \
  --output data/datasets/synth_k31.txt \
  --k 31 \
  --num-contigs 8 \
  --contig-length 50000 \
  --gc-bias 0.52 \
  --seed 7
```

You usually do not need to regenerate these unless the files are missing or you intentionally want a fresh dataset.

## Run Synthetic Benchmarks

This runs Bloom, Cuckoo, XOR, and learned filters on all three synthetic k values and target FPRs `1e-2`, `1e-3`, and `1e-4`.

The first command uses `--overwrite` to replace old synthetic results. The next two append k=21 and k=31 into the same output tree.

```bash
python scripts/run_benchmarks.py \
  --dataset data/datasets/synth_k15.txt \
  --dataset-name synth_k15 \
  --output-dir benchmarking/results \
  --overwrite \
  --seed 7 \
  --repetitions 3 \
  --filters bloom,cuckoo,xor,learned \
  --fprs 1e-2,1e-3,1e-4

python scripts/run_benchmarks.py \
  --dataset data/datasets/synth_k21.txt \
  --dataset-name synth_k21 \
  --output-dir benchmarking/results \
  --seed 7 \
  --repetitions 3 \
  --filters bloom,cuckoo,xor,learned \
  --fprs 1e-2,1e-3,1e-4

python scripts/run_benchmarks.py \
  --dataset data/datasets/synth_k31.txt \
  --dataset-name synth_k31 \
  --output-dir benchmarking/results \
  --seed 7 \
  --repetitions 3 \
  --filters bloom,cuckoo,xor,learned \
  --fprs 1e-2,1e-3,1e-4
```

Synthetic benchmark outputs are written under:

```text
benchmarking/results/k15/<filter>/
benchmarking/results/k21/<filter>/
benchmarking/results/k31/<filter>/
```

Each run writes a JSON file and updates `aggregate_results.csv`.

## Plot Synthetic Results

```bash
python scripts/plot_results.py \
  --results-dir benchmarking/results \
  --output-dir benchmarking/results/plots
```

Plots are written to:

```text
benchmarking/results/plots/
```

Use these synthetic plots in the report or slides:

```text
throughput_by_filter.png
memory_per_kmer_by_filter.png
false_positive_rate_by_filter.png
build_time_by_filter.png
fpr_vs_target.png
```

## Run Real Bacterial Benchmarks

The real datasets are much larger. The recommended final run is target FPR `1e-3` only, across all four organisms, all three k values, and all four filters:

```bash
python scripts/run_real_bacteria_benchmarks.py \
  --output-root benchmarking/final_results/real/diverse_bacteria_4 \
  --seed 7 \
  --repetitions 1 \
  --filters bloom,cuckoo,xor,learned \
  --fprs 1e-3 \
  --rerun-completed \
  --stop-on-error
```

This runs 48 benchmark tasks:

```text
12 datasets x 4 filters x 1 target FPR
```

Progress is printed to the terminal and also logged at:

```text
benchmarking/final_results/real/diverse_bacteria_4/run_real_bacteria_benchmarks.log
```

Real benchmark outputs are written under:

```text
benchmarking/final_results/real/diverse_bacteria_4/<dataset_id>/k<k>/<filter>/
```

For example:

```text
benchmarking/final_results/real/diverse_bacteria_4/ecoli_k31/k31/xor/
```

## Plot Real Results

For the final real-data plots, use only the fresh `1e-3` reference-FPR rows:

```bash
python scripts/plot_real_bacteria_results.py \
  --results-dir benchmarking/final_results/real/diverse_bacteria_4 \
  --output-dir benchmarking/final_results/real/diverse_bacteria_4/plots \
  --reference-fpr 1e-3 \
  --only-reference-fpr
```

Plots and summary CSVs are written to:

```text
benchmarking/final_results/real/diverse_bacteria_4/plots/
```

Use these real plots in the report or slides:

```text
throughput_by_filter.png
memory_per_kmer_by_filter.png
false_positive_rate_by_filter.png
build_time_by_filter.png
fpr_vs_target.png
```

The real-data summary tables are:

```text
real_cross_k_summary.csv
real_fpr_sweep_summary.csv
```

## Run One Small Benchmark

For a quick smoke test:

```bash
python scripts/run_benchmarks.py \
  --dataset data/datasets/synth_k15.txt \
  --dataset-name synth_k15_smoke \
  --output-dir benchmarking/smoke_results \
  --overwrite \
  --seed 7 \
  --repetitions 1 \
  --filters bloom,cuckoo,xor \
  --fprs 1e-3
```

Plot it:

```bash
python scripts/plot_results.py \
  --results-dir benchmarking/smoke_results \
  --output-dir benchmarking/smoke_results/plots
```

## Run a Learned-Backend Variant

The retained final results use the default learned backend:

```text
ngram_sgd
```

Other learned backends are implemented and can be run with `--learned-backend`:

```text
composition_logistic
dna_ngram_sgd
ngram_nb
ngram_sgd
prefix_set
position_logistic
```

Example:

```bash
python scripts/run_benchmarks.py \
  --dataset data/datasets/synth_k15.txt \
  --dataset-name synth_k15_dna_ngram_sgd \
  --output-dir benchmarking/learned_backend_experiments \
  --overwrite \
  --seed 7 \
  --repetitions 1 \
  --filters learned \
  --fprs 1e-3 \
  --learned-backend dna_ngram_sgd
```

Use a separate output directory for learned-backend experiments so they do not overwrite the final reported results.

## Build Filter Artifacts Directly

To build and save filter objects from one k-mer file:

```bash
python scripts/build_all.py \
  --kmer-file data/datasets/synth_k15.txt \
  --output-dir artifacts/synth_k15 \
  --k 15 \
  --seed 7
```

This writes serialized filter artifacts and prints a JSON summary.

## Final Report Draft

The current writeup draft is:

```text
writeup.md
```

It contains methodology, dataset construction, filter implementation details, synthetic results, real-data results, and learned-filter analysis.

## Important Notes

- Use `--overwrite` only when you intentionally want to replace an output directory.
- Real learned-filter runs are slow. Most runtime is model feature extraction/training, not the Bloom backup.
- The final real-data plots should be generated with `--only-reference-fpr` so old non-reference JSONs do not affect the figures.
- If Matplotlib complains about cache directories, the plotting scripts set a temporary writable cache automatically.
