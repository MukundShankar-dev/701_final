# Jellyfish Data Workflow

This project benchmarks approximate membership query filters on k-mer sets.
Jellyfish does not provide genome data itself; it counts k-mers from FASTA or
FASTQ files. We use NCBI Datasets to pull bacterial genome FASTA files, then
use Jellyfish to extract k-mers from those FASTA files.

## 1. Install Tools On macOS

Use a separate conda environment for genomics command-line tools. This keeps the
project Python `.venv` focused on the benchmark code.

```bash
brew install --cask miniforge
conda init "$(basename "${SHELL}")"
```

Restart the terminal, then create and activate the tool environment:

```bash
conda create -n jellyfish-env -c conda-forge -c bioconda kmer-jellyfish ncbi-datasets-cli
conda activate jellyfish-env
```

Verify the tools:

```bash
jellyfish --version
datasets --version
```

References:
- Jellyfish: https://github.com/gmarcais/Jellyfish
- Bioconda package: https://bioconda.github.io/recipes/kmer-jellyfish/README.html
- NCBI Datasets CLI: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/datasets/

## 2. Pull FASTA Files From NCBI

Create an accession list for the selected bacterial genomes:

```bash
mkdir -p data/raw

cat > data/raw/diverse_bacteria_20_accessions.txt <<'EOF'
GCF_000005845.2
GCF_000009045.1
GCF_000006765.1
GCF_000195955.2
GCF_000006745.1
GCF_000008525.1
GCF_000008565.1
GCF_000008545.1
GCF_000008625.1
GCF_000008725.1
GCF_000008685.2
GCF_000006885.1
GCF_000006945.2
GCF_000196035.1
GCF_000009205.2
GCF_000146165.2
GCF_000025985.1
GCF_000009085.1
GCF_000027305.1
GCF_000009065.1
EOF
```

Download genome FASTA files and NCBI metadata:

```bash
conda activate jellyfish-env

datasets download genome accession \
  --inputfile data/raw/diverse_bacteria_20_accessions.txt \
  --include genome \
  --filename data/raw/diverse_bacteria_20.zip

unzip -o data/raw/diverse_bacteria_20.zip \
  -d data/raw/diverse_bacteria_20
```

Check that FASTA files were downloaded:

```bash
find data/raw/diverse_bacteria_20 -name "*_genomic.fna" | sort
find data/raw/diverse_bacteria_20 -name "*_genomic.fna" | wc -l
```

NCBI puts organism names and genome metadata in:

```text
data/raw/diverse_bacteria_20/ncbi_dataset/data/assembly_data_report.jsonl
```

Make a readable metadata table:

```bash
jq -r '[.accession, .organism.organismName, .assemblyInfo.assemblyName, .assemblyStats.totalSequenceLength, .assemblyStats.gcPercent] | @tsv' \
  data/raw/diverse_bacteria_20/ncbi_dataset/data/assembly_data_report.jsonl \
| column -t -s $'\t' \
> data/raw/diverse_bacteria_20/metadata_table.txt
```

For a script-friendly TSV with a header:

```bash
{
  printf "accession\torganism\tassembly_name\ttotal_sequence_length\tgc_percent\n"
  jq -r '[.accession, .organism.organismName, .assemblyInfo.assemblyName, .assemblyStats.totalSequenceLength, .assemblyStats.gcPercent] | @tsv' \
    data/raw/diverse_bacteria_20/ncbi_dataset/data/assembly_data_report.jsonl
} > data/raw/diverse_bacteria_20/metadata_table.tsv
```

## 3. Extract One Genome As A Smoke Test

This example extracts 21-mers from the E. coli K-12 MG1655 FASTA.

```bash
conda activate jellyfish-env

FASTA="data/raw/diverse_bacteria_20/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna"
OUT_DIR="data/datasets/real/diverse_bacteria_20/k21/escherichia_coli_k12_mg1655__GCF_000005845.2"

mkdir -p "${OUT_DIR}"

jellyfish count \
  -m 21 \
  -s 100M \
  -t 4 \
  -o "${OUT_DIR}/genome.jf" \
  "${FASTA}"

jellyfish dump -c "${OUT_DIR}/genome.jf" \
  > "${OUT_DIR}/counts.tsv"

awk '{print $1}' "${OUT_DIR}/counts.tsv" \
  > "${OUT_DIR}/kmers.txt"
```

Verify the extracted benchmark input:

```bash
wc -l "${OUT_DIR}/kmers.txt"
head "${OUT_DIR}/kmers.txt"
```

The files mean:

```text
genome.jf   Jellyfish binary count database
counts.tsv  dumped k-mer counts, one k-mer plus count per line
kmers.txt   benchmark input, one k-mer per line
```

## 4. Keep Organism And FASTA Traceability

Use an organized output directory per organism and k value:

```text
data/datasets/real/diverse_bacteria_20/
  k15/
    organism_slug__GCF_accession/
      genome.jf
      counts.tsv
      kmers.txt
      metadata.tsv
  k21/
    organism_slug__GCF_accession/
      ...
  k31/
    organism_slug__GCF_accession/
      ...
```

For each generated k-mer dataset, keep a small `metadata.tsv` next to
`kmers.txt`:

```bash
cat > "${OUT_DIR}/metadata.tsv" <<EOF
field	value
accession	GCF_000005845.2
organism	Escherichia coli str. K-12 substr. MG1655
k	21
fasta_path	${FASTA}
jellyfish_db	${OUT_DIR}/genome.jf
counts_path	${OUT_DIR}/counts.tsv
kmers_path	${OUT_DIR}/kmers.txt
EOF
```

This lets benchmark results be traced back to the exact genome assembly and
FASTA file.

## 5. Run A Benchmark On Extracted K-mers

For a quick smoke test, run one filter and one FPR target:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_benchmarks.py \
  --dataset data/datasets/real/diverse_bacteria_20/k21/escherichia_coli_k12_mg1655__GCF_000005845.2/kmers.txt \
  --dataset-name ecoli_k21 \
  --output-dir benchmarking/final_results/real/ecoli_k21_smoke \
  --seed 7 \
  --repetitions 1 \
  --filters bloom \
  --fprs 1e-3
```

For fuller CPU-only runs, start with non-learned filters:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_benchmarks.py \
  --dataset data/datasets/real/diverse_bacteria_20/k21/escherichia_coli_k12_mg1655__GCF_000005845.2/kmers.txt \
  --dataset-name ecoli_k21 \
  --output-dir benchmarking/final_results/real/ecoli_k21 \
  --seed 7 \
  --repetitions 1 \
  --filters bloom,cuckoo,xor \
  --fprs 1e-3
```

Full-size learned-filter runs can be much slower because they train a model on
the k-mer set. Use sampled datasets while developing.

## 6. Optional: Make A Sampled Development Dataset

Use a smaller k-mer file for fast iteration:

```bash
python3.11 - <<'PY'
import random
from pathlib import Path

src = Path("data/datasets/real/diverse_bacteria_20/k21/escherichia_coli_k12_mg1655__GCF_000005845.2/kmers.txt")
dst = src.with_name("kmers_100k.txt")

random.seed(7)
with src.open() as handle:
    kmers = [line for line in handle if line.strip()]

sample = random.sample(kmers, min(100_000, len(kmers)))

with dst.open("w") as handle:
    handle.writelines(sample)

print(f"Wrote {len(sample)} k-mers to {dst}")
PY
```

Then benchmark `kmers_100k.txt` until the workflow is stable.
