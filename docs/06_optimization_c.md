# Step 6: Optimization C - Cross-Encoder Reranking

## What Optimization C Solves
- **ranked_too_low** (8.8%): Right chunk is in candidate pool but ranked outside top-10
- **Within-paper ordering**: Better ordering of chunks within the same paper

### C.1 Cross-Encoder Reranking

#### The Concept
- **Bi-encoder** (dense): Query and document encoded separately → never "see" each other
- **Cross-encoder**: Concatenates `[query, doc]` and runs through a transformer jointly → finer relevance judgment
- **Cost**: Must score every (query, candidate) pair live — no precomputation

#### What I Built (`src/ragopt/rerank.py`)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (truncated at 512 tokens)
- Applied **only to reorder the top-30 candidates**
- Everything after the pool keeps its original order

#### Why Top-30?
- The right passage is already in the candidate pool (retrieved by dense + BM25)
- Cross-encoder just lifts it into the top-10
- Cheaper than reranking the entire corpus

#### Why C Works
- The right passage is already retrieved; the cross-encoder just reorders it
- Fixes "ranked too low": Gold covered in top-50 but not top-10
- Improves MRR the most (+0.027 over baseline)

#### Numbers
- C alone: R@10 0.247 (+0.035 over baseline); MRR improves the most (+0.027)
- Cost: ~4,386 ms median latency (dominated by cross-encoder inference)
- Only top-30 reranked — tail untouched

#### Configuration
```python
cfg = Config(name="C", rerank_pool=30, reranker="cross-encoder/ms-marco-MiniLM-L-6-v2")
scorer = CrossEncoderScorer(cfg.reranker)
idx = retrieve(cfg, chunks, queries, emb, scorer)
```

#### The `rerank_pool` Parameter
- Determines how many top candidates are reranked
- `rerank_pool=30`: Rerank top-30, rest keep original order
- Larger pool = more accurate but more expensive
- `rerank_pool=0`: Reranking disabled

### C.2 How Reranking Fixes Failures

#### "ranked_too_low" Case (051df74d)
- Exact passage present in candidates but ranked #11+
- Cross-encoder scores [query, passage] pairs and reorders
- The exact passage moves into top-10

#### Example from Failure Table
- Question: "How did they obtain the OSG dataset?"
- Gold passage: "crawling and pre-processing an OSG web forum"
- Before rerank: ranked outside the top-10 (dense score favors a generic paper)
- After rerank: moved into the top-10 (cross-encoder recognizes the exact match)

### C.3 Corrective RAG Connection

#### Corrective RAG adds another checkpoint:
1. Inspect retrieval quality
2. Decide whether to proceed, retrieve again, or abstain
3. Prompted citations useful but not proof; separate verification may be needed

#### Keep in mind
- Retrieved documents should be treated as **untrusted data**, not instructions
- Citation should resolve to original passage, not just model-produced citation number

---
