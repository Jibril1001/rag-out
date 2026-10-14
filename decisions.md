# Decision log (pre-registration)

Written after the baseline diagnosis and BEFORE running any optimization experiment.

## Baseline (fixed before looking at results)
- 1,000-char chunks, 200 overlap, concatenated paper text, bge-small-en-v1.5 (normalised),
  exact inner-product search over ALL papers of the split, top-10, bge query prefix.
- Dev result: R@5 0.153 | R@10 0.212 | MRR@10 0.124 (union coverage), 0.128 | 0.174 | 0.111 (single).
- Coverage rule has two readings (union of top-k spans vs. one chunk alone); BOTH are reported until clarified.

## Diagnosis (dev, 261 questions with gold evidence)
| outcome | share |
|---|---|
| success | 16.1% |
| wrong_paper (no top-10 chunk from the target paper) | 40.2% |
| incomplete_multi_evidence | 11.5% |
| boundary_split | 10.0% |
| ranked_too_low (gold in top-50, not top-10) | 8.8% |
| not_found | 13.4% |

- Restricting search to the target paper (diagnostic only) lifts R@10 0.212 -> 0.651: most loss is paper confusion.
- Chunking ceiling of the baseline chunks (perfect ranker): union R@10 1.00, single R@10 0.69.
- 3.0 of the 10 retrieved chunks overlap another retrieved chunk.

## Chosen optimizations
- **A (ingestion)**: structure-aware chunks + title/section heading in the EMBEDDED text only.
  Targets wrong_paper (40.2%) and boundary_split (10.0%).
- **B (retrieval)**: dense + BM25 (own implementation) fused with RRF.
  Targets wrong_paper and not_found (13.4%): rare terms, named entities, dataset/metric names.
- **C (selection)**: cross-encoder rerank over the top-30 pool.
  Targets ranked_too_low (8.8%) and within-paper ordering.

## Extended bank (run as extra ablations)
MMR/dedup (redundancy), HyDE, multi-query, LLM contextual enrichment, small-to-big,
paper-level routing, Flat vs IVF vs HNSW.

## Rules
- All selection is done on DEV. Held-out is run only for the baseline and the single official final.
- An optimization is kept only if it improves dev R@10 without hurting MRR@10; LLM-based ones must also beat noise (paired bootstrap).
