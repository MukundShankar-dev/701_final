# Interim Project Report

**Team Members:** Anirudh Poruri, Mukund Shankar  
**Designated submitter for this team report:** Mukund Shankar


## Project goal and work accomplished so far

Our goal is to build a reproducible experimental framework for approximate membership query (AMQ) data structures on genomic k-mer sets and use it to compare practical tradeoffs between memory usage per k-mer, empirical false positive rate (FPR), and query throughput. For the final report, we aim to produce a fair and controlled comparison of Bloom filters, Cuckoo filters, XOR-style static filters, and learned-filter hybrids under shared datasets and query workloads, and to identify settings in which each method is preferable for genomics-style applications.

So far, we have implemented a working Python benchmarking codebase with a shared AMQ-style interface, reusable dataset loading and synthetic dataset generation utilities, and a unified benchmarking harness that produces JSON and CSV outputs for repeated runs. We implemented Bloom filters and Cuckoo filters directly in Python and built a learned-filter prototype using logistic regression (via `scikit-learn`) together with a backup Bloom filter. We are currently running experiments on synthetic k-mer datasets with controlled GC bias and seeded randomness while preparing a Jellyfish-based pipeline for extracting real k-mers from bacterial genomes.

In addition, we implemented command-line scripts for dataset generation and parameter sweeps, basic plotting utilities for interpreting benchmark outputs, and `pytest` tests validating expected AMQ behavior (e.g., no false negatives for Bloom filters on inserted keys, correct insert/query/delete behavior for Cuckoo filters, and sanity checks on metric computations). We are currently integrating a full XOR-filter backend so that static-filter comparisons are scientifically meaningful and directly comparable to Bloom and Cuckoo results.


## Relevant literature and next steps


### Most relevant literature/resources

The four most relevant references guiding our approach are:

- **Graf & Lemire (2020), _Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters_**  
  This paper motivates static AMQ structures as strong baselines for memory-efficient membership testing and informs our planned XOR-filter evaluation.

- **Fan et al. (2014), _Cuckoo Filter: Practically Better Than Bloom_**  
  This provides the primary deletion-capable AMQ baseline and informs the parameter choices and performance expectations for our dynamic-filter comparisons.

- **Pandey et al. (2018), _Mantis: A Fast, Small, and Exact Large-Scale Sequence-Search Index_**  
  This anchors the genomics-scale setting of our project and illustrates practical constraints and expectations for large k-mer indexing systems.

- **Mitzenmacher (2018), _A Model for Learned Bloom Filters and Optimizing by Sandwiching_**  
  This provides the theoretical framework for learned-filter hybrids and guides the structure and evaluation of our learned-filter baseline.

In addition, we are using the **Jellyfish k-mer counter** as our primary tool for extracting real genomic k-mer datasets and **Limasset (2026), _ZOR filters_** as a reference for recent developments in deterministic static AMQ structures.


### Next steps to end of semester

Our next step is to integrate real bacterial-genome datasets into the benchmarking pipeline using Jellyfish-generated k-mer files while keeping synthetic datasets as controlled baselines. We then plan to run parameter sweeps over k-mer length (`k = 15, 21, 31`), dataset size (from approximately `100k` to `1M+` k-mers), and target false positive rates (`10^-2`, `10^-3`, `10^-4`). We also plan to improve feature design and threshold tuning for the learned-filter prototype and compare learned filters against Bloom filters under both matched-memory and matched-FPR settings.

We are currently integrating a full XOR-filter backend so that static-filter comparisons can be included alongside dynamic AMQs under the same evaluation pipeline. By the final deadline, our objective is to produce a reproducible experimental comparison with clear plots, tables, and discussion of fairness assumptions and methodological limitations.


## Most relevant question

What is the fairest comparison protocol across dynamic AMQ structures (Bloom and Cuckoo filters), static AMQ structures (XOR-style filters), and learned-filter hybrids? In particular, should methods primarily be compared under matched memory budgets, matched target false positive rates, or both as separate evaluation settings?