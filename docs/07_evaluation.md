# Step 7: Evaluation - Metrics and Analysis

## Retrieval Metrics

### Gold-Evidence Coverage (80% Token Rule)
A gold paragraph is "covered" when retrieved source spans contain **at least 80% of its word tokens**.

```text
coverage(paragraph) = covered_tokens / total_tokens  >=  0.8
```

Two readings implemented (both reported):

| Mode | Description |
|---|---|
| **union** | Merge spans of all top-k chunks, then check 80% **(headline number)** |
| **single** | One chunk alone must cover 80% |

### Recall@k
```
Recall@k = covered gold paragraphs / all gold paragraphs  (top-k)
```

### MRR@10
```
MRR@10 = 1 / (first rank at which ANY gold paragraph is covered)
```

### Development vs Held-Out Results

| Split | Config | Coverage | R@5 | R@10 | MRR@10 |
|---|---|---|---|---:|---:|---:|
| Dev | Baseline | Union | 0.153 | 0.212 | 0.124 |
| Dev | Final | Union | 0.246 | 0.354 | 0.230 |
| Held-out | Baseline | Union | 0.169 | 0.234 | 0.140 |
| Held-out | Final | Union | 0.259 | 0.339 | 0.221 |

**Key insight**: Gains transfer to held-out (+0.105 recall), since held-out was never tuned.

### Questions Excluded from Metrics
- 261 of 295 dev questions have paragraph-level gold
- 155 of 182 held-out questions have paragraph-level gold
- Rest have only figure/table evidence (`FLOAT SELECTED …`), no paragraph ID, no tokens

## Answer Generation Metrics (20 Questions)

### Normalized Answer F1
- SQuAD-style token F1: lowercase, strip punctuation + articles, token F1, max over annotators
- Sample: 19 answerable + 1 gold-unanswerable out of 20

### False-Answer Rate
- Non-abstained answers on gold-unanswerable questions
- "undefined" if no gold-unanswerable questions in sample
- Currently: 0.0 (n=1) — exploratory, not reliable

### Answer Results

| System | Answer F1 (all 20) | Abstained | Gold-unanswerable | False-answer rate |
|---|---|---:|---:|---:|---:|
| Baseline | 0.157 | 9/20 | 1 | 0.0 (n=1) |
| Final | 0.203 | 9/20 | 1 | 0.0 (n=1) |

### Latency and Cost

| Scope | Baseline | Final |
|---|---:|---:|
| Retrieval median (dev) | 13 ms | 5,027 ms |
| Retrieval median (held-out) | 14 ms | 5,350 ms |
| Retrieval + generation median (20 qs) | 5,239 ms | 10,770 ms |
| LLM calls (held-out total) | 20 | 20 |

**The cross-encoder dominates**: ~5 s/query vs 13 ms for dense only. All 182 held-out questions incur retrieval; only the 20 selected incur an LLM call.

## Ablation Table (from decisions.md)

| Configuration | R@5 | R@10 | MRR@10 | Latency | LLM calls/q |
|---|---|---|---|---|---|
| Baseline (fixed + dense) | 0.153 | 0.212 | 0.124 | 13 ms | 0 |
| + A (structure + context) | 0.195 | 0.287 | 0.146 | 14 ms | 0 |
| + B (dense + BM25, RRF) | 0.161 | 0.255 | 0.141 | 386 ms | 0 |
| + C (cross-encoder top-30) | 0.191 | 0.247 | 0.151 | 4,386 ms | 0 |
| **Final (A + B + C)** | **0.246** | **0.354** | **0.230** | 5,027 ms | 0 |

## How to Read the Table Aloud
> "A is the biggest single win and nearly free; B helps a little but costs 30×; C helps ordering but is expensive. Combining all three wins on every metric — **+0.142 recall and +0.106 MRR** over baseline."

## Failure Analysis → Optimization Mapping

| Failure (share) | Chosen optimization | Stage | Theory |
|---|---|---|---|
| wrong_paper (40.2%) | **A** structure + title/section context, **B** BM25 hybrid | ingestion + retrieval | disambiguate papers by context; match rare exact terms |
| boundary_split (10.0%) | **A** structure-aware chunking | ingestion | snap chunk edges to paragraph/section boundaries |
| ranked_too_low (8.8%) | **C** cross-encoder rerank | selection/ranking | joint query-doc attention reorders the pool |
| not_found (13.4%) | **B** BM25 lexical retrieval | retrieval | exact rare tokens that dense embeddings dilute |
| incomplete_multi (11.5%) | **A** + **C** | ingestion + ranking | keep facets intact, surface more than one gold paragraph |
---
