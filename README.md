# RAG Optimization

## Overview

This project investigates methods for improving retrieval quality in a retrieval-augmented generation (RAG) pipeline over research papers.

The work starts with a dense-retrieval baseline, diagnoses its failure modes, and evaluates three optimization strategies: structure-aware chunking, hybrid dense and lexical retrieval, and cross-encoder reranking.

The repository contains the retrieval implementation, experiment scripts, evaluation metrics, tests, experiment outputs, a decision log, and the required JSONL submission files.

> **New here?** Read [`GUIDE.md`](GUIDE.md) for a narrative walk-through of how the pipeline works, why each optimization was chosen, and what the numbers mean.

## Objectives

- Establish a reproducible retrieval baseline.
- Diagnose why relevant evidence is missed or ranked too low.
- Evaluate individual retrieval optimizations on the development split.
- Select and evaluate a final retrieval configuration.
- Compare baseline and final performance on held-out data.
- Validate and package the required JSONL outputs.
- Document experimental decisions, failure cases, limitations, and results.

## Current Progress

### Completed

- Implemented dense retrieval using `bge-small-en-v1.5`.
- Implemented fixed-size and structure-aware chunking.
- Implemented BM25 lexical retrieval.
- Implemented reciprocal-rank fusion for hybrid retrieval.
- Implemented optional cross-encoder reranking.
- Implemented Recall@k, MRR@10, and gold-evidence coverage evaluation.
- Added retrieval failure classification and diagnostic scripts.
- Ran baseline and development-set optimization experiments.
- Recorded the initial diagnosis and optimization rationale in `decisions.md`.
- Generated the five development JSONL outputs and two held-out JSONL outputs.
- Validated all seven required JSONL files successfully.
- Ran a limited answer-generation check using `scripts/answer20.py`.
- Completed the ten-case failure table (`results/failure_table.md`).
- Populated the ablation, development/held-out, and answer-scoring tables below.
- Scored the 20 answer questions with normalized answer F1 (`scripts/score_answers.py`).
- Added `requirements.txt`, `requirements.lock.txt`, and fresh-environment setup instructions.

### Remaining work

- Confirm the Git push completed successfully.
- Finalize the README and submission review.

## Repository Structure

```text
rag-opt/
├── configs/                  # Configuration files, if applicable
├── notebooks/                # Notebooks, if applicable
├── results/
│   ├── dev_baseline.jsonl
│   ├── dev_A.jsonl
│   ├── dev_B.jsonl
│   ├── dev_C.jsonl
│   ├── dev_final.jsonl
│   ├── heldout_baseline.jsonl
│   ├── heldout_final.jsonl
│   ├── failure_table.md
│   └── experiments/          # Experiment summaries and diagnostics
├── scripts/                  # Experiment and validation scripts
├── src/ragopt/
│   ├── analysis/             # Failure analysis
│   ├── chunking/              # Chunking implementations
│   ├── bm25.py                # Lexical retrieval
│   ├── dense.py               # Dense retrieval
│   ├── embed.py               # Embedding utilities
│   ├── fusion.py              # Rank fusion
│   ├── metrics.py             # Evaluation metrics
│   ├── pipeline.py            # Retrieval pipeline
│   └── rerank.py              # Cross-encoder reranking
├── tests/                    # Automated tests
├── decisions.md               # Experimental decision log
├── pyproject.toml             # Python project metadata
├── .gitignore
```

## Baseline Configuration

The recorded baseline uses:

- Fixed-size chunks of 1,000 characters.
- A 200-character overlap.
- Concatenated paper text.
- Normalized `bge-small-en-v1.5` embeddings.
- Exact inner-product search over the papers in the selected split.
- The configured BGE query prefix.
- Top-10 retrieved chunks for evaluation.

## Optimization Experiments

### A: Structure-aware chunking

Uses document structure to create chunks and includes paper titles and section headings in the text used for embedding.

**Motivation:** Preserve useful document boundaries and improve contextual identification of relevant evidence.

### B: Hybrid retrieval

Combines dense retrieval and BM25 using reciprocal-rank fusion.

**Motivation:** Improve retrieval for exact terms, rare words, named entities, dataset names, and evaluation metrics that lexical matching can identify effectively.

### C: Cross-encoder reranking

Reranks a candidate pool using a cross-encoder, configured by default as `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Motivation:** Improve the ordering of candidate chunks when relevant evidence is retrieved but ranked too low.

### Final configuration

The final pipeline combines the selected changes based on development-set experiments. The configuration and selection rationale should be documented alongside the measured results in `decisions.md` and the experiment summaries.

## Evaluation Methodology

The project reports:

- **Recall@5:** Gold evidence coverage among the five highest-ranked chunks.
- **Recall@10:** Gold evidence coverage among the ten highest-ranked chunks.
- **MRR@10:** Rewards retrieving qualifying gold evidence near the top of the ranking.

A gold paragraph counts as covered when retrieved source spans cover at least 80% of its tokens.

Two coverage modes are implemented:

- **Union coverage:** Combines spans from multiple retrieved chunks.
- **Single-chunk coverage:** Requires one chunk to cover at least 80% of the gold paragraph on its own.

Both modes are reported because the task specification does not clearly settle which interpretation is required.

## Initial Baseline Findings

The recorded development baseline covers 261 questions with gold evidence.

| Metric | Union coverage | Single-chunk coverage |
|---|---:|---:|
| Recall@5 | 0.153 | 0.128 |
| Recall@10 | 0.212 | 0.174 |
| MRR@10 | 0.124 | 0.111 |

The initial failure diagnosis reported:

| Category | Share |
|---|---:|
| Success | 16.1% |
| Wrong paper | 40.2% |
| Incomplete multi-evidence | 11.5% |
| Boundary split | 10.0% |
| Ranked too low | 8.8% |
| Not found | 13.4% |

A diagnostic run that restricted retrieval to the target paper raised union Recall@10 from 0.212 to 0.651. This suggests that retrieving from the wrong paper is an important source of failure. The restriction is diagnostic evidence, not a claim that the target paper is known during ordinary retrieval.

## Experimental Protocol

The development split is used to compare and select configurations. The held-out split is reserved for baseline evaluation and one selected final evaluation.

The documented decision rule is to retain an optimization when it improves development Recall@10 without hurting MRR@10. Expensive or LLM-based changes require evidence of improvement rather than an assumption that added complexity is beneficial.

Experiment summaries and per-question diagnostics are stored in `results/experiments/`.

## Required JSONL Outputs

| File | Description |
|---|---|
| `results/dev_baseline.jsonl` | Development baseline |
| `results/dev_A.jsonl` | Experiment A |
| `results/dev_B.jsonl` | Experiment B |
| `results/dev_C.jsonl` | Experiment C |
| `results/dev_final.jsonl` | Selected final development output |
| `results/heldout_baseline.jsonl` | Held-out baseline |
| `results/heldout_final.jsonl` | Held-out final output |

The most recent recorded validation passed for all seven files. The development files each contained 295 records, and the held-out files each contained 182 records. These seven files are the submission; no archive is required.

## Results

All retrieval metrics use the 80%-token-coverage rule on the target paper only. "Union" merges the source spans of all top-k chunks; "Single" requires one chunk alone to cover 80% of a gold paragraph. Both are reported because the brief does not state which interpretation is required. Numbers are reproduced from the saved JSONL outputs with `scripts/score_answers.py`. Questions whose gold evidence consists only of figure/table references (no paragraph) cannot be scored for paragraph coverage and are excluded: 261 of 295 development and 155 of 182 held-out questions have paragraph-level gold.

### Independent ablation (development, 261 questions with gold evidence)

| Configuration | R@5 | R@10 | MRR@10 | Median retrieval latency | LLM calls/q |
|---|---|---:|---:|---:|---:|---:|
| Baseline (fixed chunks + dense) | 0.153 | 0.212 | 0.124 | 13 ms | 0 |
| Baseline + A (structure + context) | 0.195 | 0.287 | 0.146 | 14 ms | 0 |
| Baseline + B (dense + BM25, RRF) | 0.161 | 0.255 | 0.141 | 386 ms | 0 |
| Baseline + C (cross-encoder top-30) | 0.191 | 0.247 | 0.151 | 4,386 ms | 0 |
| Final (A + B + C) | 0.246 | 0.354 | 0.230 | 5,027 ms | 0 |

Union coverage. LLM calls are zero for development because generation is only required for the 20 held-out questions.

- A alone improves R@10 by +0.075 (structure-aware chunks fix boundary splits and title/section context reduces paper confusion).
- B alone improves R@10 by +0.043 but raises latency ~30x (BM25 over the whole corpus).
- C alone improves MRR@10 the most (+0.027) but is the most expensive (~4.4 s/query).
- The final combination is best on all three metrics: +0.142 R@10 and +0.106 MRR@10 over baseline.

### Final development vs held-out

| Split | Configuration | Coverage | R@5 | R@10 | MRR@10 |
|---|---|---|---|---:|---:|---:|
| Development | Baseline | Union | 0.153 | 0.212 | 0.124 |
| Development | Final | Union | 0.246 | 0.354 | 0.230 |
| Development | Baseline | Single | 0.128 | 0.174 | 0.111 |
| Development | Final | Single | 0.238 | 0.345 | 0.227 |
| Held-out | Baseline | Union | 0.169 | 0.234 | 0.140 |
| Held-out | Final | Union | 0.259 | 0.339 | 0.221 |
| Held-out | Baseline | Single | 0.158 | 0.212 | 0.135 |
| Held-out | Final | Single | 0.247 | 0.310 | 0.216 |

Held-out n = 155 questions with gold evidence (of 182). The final system's gains transfer to held-out: union R@10 +0.105 and MRR@10 +0.081 over baseline, consistent with (slightly smaller than) the development gains. The held-out split was not used for tuning; only the baseline and the single chosen final were run on it.

### Answer generation (20 held-out questions)

Answers were generated for the fixed 20-question sample with `gemini-3.5-flash-lite` (temperature 0) using identical grounded prompts for baseline and final. Scoring uses QASPER-style normalized answer F1: token-level F1 over lowercased, punctuation-stripped, article-removed answers, max over annotator references.

| System | Answer F1 (all 20) | Answer F1 (19 answerable) | Abstained | Gold-unanswerable | False-answer rate |
|---|---|---:|---:|---:|---:|---:|
| Baseline | 0.157 | 0.113 | 9/20 | 1 | 0.0 (n=1) |
| Final | 0.203 | 0.161 | 9/20 | 1 | 0.0 (n=1) |

The 20-question sample contains **1** gold-unanswerable question; both systems correctly abstained on it, so the false-answer rate is 0.0 but is exploratory (n=1) and not a reliable ranking signal. Both systems answer 11 and abstain 9 of the 20 questions, but the final system's answers score higher on F1 (0.203 vs 0.157). The sample is far too small for a conclusive answer-quality claim.

### Latency and cost

Retrieval latency is the median of single-query timings (warm models, uncached reranker) applied per record. Generation latency is measured on the 20 answered questions and includes the answerer's 4.5 s rate-limit sleep.

| Scope | Baseline | Final |
|---|---:|---:|
| Retrieval median (dev) | 13 ms | 5,027 ms |
| Retrieval median (held-out) | 14 ms | 5,350 ms |
| Retrieval + generation median (20 qs) | 5,239 ms | 10,770 ms |
| LLM calls (held-out, total) | 20 | 20 |

The cross-encoder reranker is the dominant cost: it accounts for the ~5 s/query final retrieval versus 13 ms for the dense baseline. All 182 held-out questions incur retrieval cost, but only the 20 selected questions incur an LLM call (1 call each).

## Reflection

**What worked.** Structure-aware chunking (A) was the single largest source of improvement, fixing boundary splits and, via title/section context, reducing paper confusion. Fusing self-implemented BM25 with dense retrieval (B) added lexical recall for rare/exact terms (dataset names, "BERTbase", "Elasticsearch") that dense embeddings miss. Cross-encoder reranking (C) fixed "ranked too low" cases by reordering the top-30 candidates. Together the final system lifts union R@10 from 0.212 to 0.354 on development and from 0.234 to 0.339 on held-out.

**What did not help.** B alone is a marginal improvement (+0.043 R@10) at a ~30x latency cost, and C alone improves ordering but barely recall (+0.035 R@10). Tested ablations that were not selected include BM25-only retrieval, structure-without-context, and larger rerank pools (50 vs 30), none of which beat the final settings.

**Quality vs cost trade-off.** The final system is ~380x slower per query than the dense baseline (5.0 s vs 13 ms) because of the cross-encoder. For this offline, retrieval-first task that cost is acceptable and the gains are substantial; for a latency-sensitive production setting, structure chunking plus hybrid retrieval without reranking (roughly +0.107 R@10 over baseline at ~386 ms) is a cheaper alternative worth considering. LLM cost is tightly controlled by the task design: answers are generated for only 20 questions, one call each.

## Scope and unbuilt experiments

The required optimizations (A, B, C) plus the full evaluation and reporting are complete. A wider "extended bank" of techniques was planned in `decisions.md` but **not implemented or measured** — including approximate nearest-neighbor indexes (**IVF, HNSW**), MMR/dedup, HyDE, multi-query, LLM contextual enrichment, and small-to-big retrieval. Only exact dense search (`FlatIPIndex`) is used. These were deprioritized once the minimum complete submission was finished; for IVF/HNSW specifically, exact search is already microseconds at this corpus size, so approximate indexing would give ~0 accuracy gain. See [GUIDE.md §8](GUIDE.md#8-planned-but-not-built-honest-scope-statement).

## Setup and Validation

Requires Python 3.10 or newer (the pinned environment in `requirements.lock.txt` needs 3.12+ and is tested on 3.13).

### Fresh clone (Windows PowerShell), from the repository root

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run the test suite (28 tests, no network or model download required)
python -m pytest

# Validate the required JSONL outputs
python scripts\validate.py results\dev_*.jsonl results\heldout_*.jsonl
```

### Fresh clone (macOS / Linux), from the repository root

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
python scripts/validate.py 'results/dev_*.jsonl' 'results/heldout_*.jsonl'
```

`requirements.txt` installs the direct dependencies at their lower bounds and the package itself in editable mode, so `import ragopt` works from any directory. To reproduce the exact verified dependency versions instead:

```powershell
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
```

### What the first real run downloads

Tests and validation run offline. Running the retrieval experiments needs network access on first use:

- `allenai/qasper` validation split (pinned revisionvia `datasets` — cached by Hugging Face.
- `BAAI/bge-small-en-v1.5` encoder and `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker via `sentence-transformers` — cached by Hugging Face, plus the project's own embedding/score cache in `.cache/` (gitignored, rebuildable).
- `scripts/answer20.py` additionally requires `$env:GEMINI_API_KEY` and `pip install google-genai` (included in `requirements.txt`).

No API keys are needed for tests, experiments, or validation.

## Security and Repository Hygiene

- Do not commit API keys, tokens, credentials, or private environment files.
- Keep virtual environments, caches, and temporary files out of Git.
- Review ZIP archives before committing them, especially if they contain datasets, model weights, caches, or duplicate deliverables.
- Keep the required source code, documentation, experiment results, and submission files available for reproducibility.

## Next Steps

1. Review repository file sizes and remove unnecessary archives.
2. Complete the failure-case table.
3. Extract and report the final development and held-out metrics.
4. Verify answer-generation quality and the full test suite.
5. Confirm the Git remote and push status.
6. Perform a final submission checklist against the assignment requirements.