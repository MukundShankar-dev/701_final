# Interim Project Report

**Team Members:** Anirudh Poruri, Mukund Shankar  
**Designated submitter for this team report:** Mukund Shankar


## Project goal and work accomplished so far

Our goal is to build a reproducible experimental framework for approximate membership query (AMQ) structures on genomic k-mers, and use it to report practical tradeoffs across memory per k-mer, empirical false positive rate (FPR), and query speed. For the final report, we want a fair comparison of methods under the same datasets and query workloads, with enough detail to justify when one method is preferable over another in genomics-style settings.

So far, we have a working Python benchmark codebase with a shared AMQ-style interface, common data loading/generation utilities, and a single benchmarking harness that produces JSON and CSV outputs for repeated runs. We implemented Bloom and Cuckoo filters ourselves in Python, and we built a learned-filter prototype using `scikit-learn` (logistic regression) plus a backup Bloom filter. We are currently running experiments on synthetic k-mer datasets (with controlled GC bias and seeded randomness) because we have not yet completed our Jellyfish-based real-data pipeline. We also added command-line scripts for dataset generation and benchmark sweeps, plus basic plotting and `pytest` tests to validate expected behavior (e.g., Bloom no false negatives on inserted keys, Cuckoo insert/query/delete behavior, and metric sanity checks on small datasets).

## Relevant literature and next steps

### Most relevant literature/resources

The four most relevant references we are using are: **Graf & Lemire (2020), _Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters_** (https://arxiv.org/abs/1912.08258), which motivates the static AMQ direction we want to evaluate; **Fan et al. (2014), _Cuckoo Filter: Practically Better Than Bloom_** (https://doi.org/10.1145/2674005.2674994), which gives the design and practical tradeoff baseline for deletion-capable AMQs; **Pandey et al. (2018), _Mantis: A Fast, Small, and Exact Large-Scale Sequence-Search Index_** (https://doi.org/10.1016/j.cels.2018.05.021), which anchors our genomics problem setting and scale expectations; and **Mitzenmacher (2018), _A Model for Learned Bloom Filters and Optimizing by Sandwiching_** (https://papers.nips.cc/paper/7328-a-model-for-learned-bloom-filters-and-optimizing-by-sandwiching), which is directly relevant to our learned-filter baseline and how to reason about its guarantees. As additional practical resources, we are also using **Limasset (2026), _ZOR filters_** (https://arxiv.org/abs/2602.03525) for deterministic static-filter ideas, and **Jellyfish** (https://github.com/gmarcais/Jellyfish) for future real k-mer extraction/counting workflows.

### Next steps to end of semester

Next, we plan to connect real bacterial-genome datasets into the same pipeline (likely through Jellyfish-generated k-mer files), keep synthetic datasets as controlled baselines, and run full sweeps over `k` (15/21/31), target FPR (`1e-2`, `1e-3`, `1e-4`), and dataset size (starting around `100k` and scaling toward `1M+` k-mers). We also want to improve learned-filter feature design and threshold tuning, then compare learned-vs-Bloom under both matched memory budget and matched target FPR. If time permits, we will replace our current XOR scaffold with a true backend so those results are scientifically meaningful. By the final deadline, our objective is a reproducible experimental comparison with clear plots/tables and explicit discussion of fairness assumptions and limitations.

## Most relevant question

What is the fairest comparison protocol across dynamic AMQs (Bloom/Cuckoo), static AMQs (XOR-style), and learned+backup hybrids: should we primarily match methods by memory budget, by target FPR, or report both equally as first-class evaluation settings?
