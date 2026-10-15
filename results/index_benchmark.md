# Index benchmark: exact vs IVF vs HNSW (from scratch, standalone)

- Corpus: **2376** baseline chunks (fixed 1000/200) over the 80 dev papers, dim 384;
  **261** gold questions; embeddings from `.cache` (0 re-encoded).
- Method: every index returns top-10; agreement@10 vs exact search, plus end-to-end
  union R@10/MRR@10 (the report's metric). IVF = spherical k-means (seed 42, 25 iters);
  HNSW = simplified from-scratch layered graph (no search-time neighbor heuristic).
  **This file is not part of the submitted pipeline - it exists to test the
  exact-vs-approximate decision with measurements.**

```
index                              params                             build s  query us   agree    R@10     MRR
exact FlatIP (shipped)             O(N*d) matrix multiply                0.00    1626.0    1.00   0.212   0.124
IVF (k-means, from scratch)        nlist=49, nprobe=1                    0.21     121.7    0.40   0.132   0.074
IVF (k-means, from scratch)        nlist=49, nprobe=4                    0.21     350.9    0.67   0.167   0.105
IVF (k-means, from scratch)        nlist=49, nprobe=16                   0.21    3063.4    0.95   0.210   0.123
IVF (k-means, from scratch)        nlist=49, nprobe=64                   0.21    5872.6    1.00   0.212   0.124
HNSW (layered graph, from scratch) M=16, efConstruction=200, efSearch=16   15.21    1135.3    0.66   0.176   0.098
HNSW (layered graph, from scratch) M=16, efConstruction=200, efSearch=64   15.21    2930.1    0.86   0.190   0.110
HNSW (layered graph, from scratch) M=16, efConstruction=200, efSearch=256   15.21    9955.3    0.96   0.196   0.119
```

## Reading

- **agree < 1.00 is accuracy lost against exact** - approximate indexes can only tie
  exact here, never beat it (same embeddings, same chunks, exact within probed cells).
- **Two exact numbers, both real:** the table's 1.63 ms/query is
  *live single-question* serving (small-GEMM per-call overhead); the pipeline evaluates
  batched - one GEMM + one top-k for all 261 queries at once - which measures
  **115 us/query amortized**. That µs-scale figure is the one quoted in the
  talk; the 13 ms measured dense retrieval is dominated by the query-embedding model call
  either way.
- **Speed only ever came from dropping accuracy:** IVF nprobe=1/4 and HNSW ef=16 beat
  exact's latency only by not scanning enough candidates (agree 0.40/0.67/0.66 -> R@10
  0.132/0.167/0.176 vs exact 0.212). Set equal to exact (IVF nprobe=64) it ties every
  metric but is 3.6x slower; HNSW never matched
  exact's top-10 at any efSearch.
- **Build cost is real:** IVF k-means 0.21 s, HNSW graph 15.2 s,
  exact ~0 s (the matrix already exists).
- **Conclusion at N=2376:** shipping exact was right - measured, not assumed. IVF/HNSW
  become worth testing at roughly 10-100x this corpus size.
