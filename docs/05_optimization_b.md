# Step 5: Optimization B - Dense + BM25 Hybrid Retrieval

## What Optimization B Solves
- **wrong_paper** (40.2%): Dense embeddings rank thematically similar wrong papers
- **not_found** (13.4%): Rare exact terms (dataset names, model names) not surfaced by dense retrieval

### B.1 Dense Retrieval (The Baseline)

#### How It Works
- **Bi-encoder**: `bge-small-en-v1.5` maps query and document independently to vectors
- **Inner product = cosine similarity** (vectors are L2-normalized)
- **FlatIPIndex**: Exact search — one matrix multiply `query @ documentsᵀ`, top-k via `np.argpartition`
- **Query prefix**: `"Represent this sentence for searching relevant passages: "`

#### Strengths
- Captures **semantic similarity**: "prevent overfitting" ↔ "regularization through dropout"
- **Precomputable**: Document embeddings cached to `.cache/` — one pass per split

#### Weaknesses
- Misses **rare exact terms**: dataset names (`BERTbase`), error codes, uncommon terminology
- Cannot match on exact word overlap when paraphrase is involved

#### Latency
- **13 ms** median retrieval (dev) — document embeddings precomputed, only query encoding + matrix multiply

### B.2 BM25: Lexical Retrieval from Scratch

#### Implementation (from `src/ragopt/bm25.py`)
The brief forbids `rank_bm25` or equivalent, so BM25 was implemented from scratch:

1. **`tokenize`**: Lowercase, alphanumeric only, stopword list removed
2. **`fit()`**: Build inverted index (term → postings with term frequency), compute document lengths and average length (`avgdl`)
3. **`idf`**: `ln(1 + (N − df + 0.5)/(df + 0.5))` — inverse document frequency
4. **`scores()`**: Okapi BM25 scoring:
   ```
   score = idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · dl/avgdl))
   ```
   With `k1 = 1.2`, `b = 0.75`

#### Strengths
- **Exact token matching**: Rare terms, dataset names (`BERTbase`), error codes, numerical settings
- **No meaning needed**: Helpful when the question and document use different phrasing but share key tokens

#### Weaknesses
- **No notion of meaning/paraphrase**: Can't connect "prevent overfitting" with "regularization through dropout"
- **Stopword filtering**: May remove useful context words

#### Latency
- **~386 ms** median retrieval (dev) — BM25 pass adds significant cost over dense only

### B.3 Reciprocal Rank Fusion (RRF)

#### How It Works
Fuses dense and BM25 rankings without requiring comparable score scales:
```
score(d) = Σ 1/(k + rank_i(d)),  k = 60
```
- Each ranker contributes equally (default weights [1.0, 1.0])
- Only cares about **ranks**, not raw scores
- Robust when the two rankers disagree

#### Why RRF Helps
- Dense misses rare terms; BM25 misses paraphrases
- RRF combines both "instincts" without score calibration
- Especially helps "wrong_paper" and "not_found" failures

#### Example
If dense ranks paper A #1 and BM25 ranks paper A #5, RRF gives:
```
score = 1/(60+1) + 1/(60+5) = 1/61 + 1/65 ≈ 0.0164 + 0.0154 = 0.0318
```
Paper A gets more credit than if only one ranker were used.

#### Numbers
- B alone (dense + BM25 + RRF): R@10 0.255 (+0.043 over baseline)
- Cost: ~386 ms median latency (vs 13 ms for dense only)

### B.4 Hybrid Retrieval in Practice

#### Configuration
```python
cfg = Config(name="hybrid_fixed", retriever="hybrid", topn=50, rrf_k=60)
chunks = build_chunks(cfg, papers)  # structure-aware or fixed
idx = retrieve(cfg, chunks, queries, emb, scorer)
```
- `retriever="hybrid"`: Enables both dense and BM25
- `rrf_k=60`: RRF parameter (default 60)
- `topn=50`: Candidates returned (first 10 are the answer)

#### Pipeline Flow
```
Query → Query embedding (dense) + BM25 tokenization
       → Dense scores (matrix multiply) + BM25 scores (inverted index)
       → RRF fusion → Re-ranked candidate list
       → Top-10 chunks returned
```

---
