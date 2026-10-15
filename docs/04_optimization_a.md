# Step 4: Optimization A - Structure-Aware Chunking + Title Context

## What Optimization A Solves
- **boundary_split** (10.0% of dev failures): Chunk edges cut gold paragraphs
- **wrong_paper** (40.2% of dev failures): Papers with overlapping topics confused

### A.1 Structure-Aware Chunking

#### How It Works
1. **Group paragraphs by section** — the abstract is its own section; other paragraphs grouped by their section heading
2. **Split long paragraphs** at sentence boundaries (never mid-sentence)
3. **Merge short paragraphs** in the same section: if `merge_below=300`, units shorter than 300 chars merge into the next unit (as long as total ≤ max_chars=1000)
4. **Never cross section boundaries** — each chunk stays within one section
5. **`context=True` option**: Prepend `title | section heading` to `embed_text` only
   - Returned text stays original paragraph (so `source_spans` remain exact)

#### Example
- **Input**: Paper with sections "Abstract", "Introduction", "Methods", "Results"
- **Output**: Chunks that respect section boundaries; each chunk's `embed_text` may include "PaperTitle | SectionName"

#### Why It Works
- **Boundary snapping**: Gold paragraphs stay inside single chunks → can reach 80% token coverage
- **Title/heading context**: The encoder gets a disambiguating signal — "this chunk is from the FacTweet paper's Experiments section"

#### Numbers
- A alone lifts R@10 from 0.212 → **0.287** (+0.075)
- Nearly free: 14 ms vs 13 ms latency
- Fixes boundary_split failures by snapping chunk edges to paragraph/section boundaries

#### Configuration
```python
cfg = Config(name="struct_ctx", chunker="structure", context=True)
chunks = build_chunks(cfg, papers)
```
- `chunker="structure"`: Enables structure-aware chunking
- `context=True`: Prepends title + section heading to embedded text only
- `max_chars=1000`: Max chars per chunk
- `merge_below=300`: Merge units shorter than 300 chars

---

### A.2 Title/Section Context in Embeddings

#### The Idea
When `context=True`, each chunk's `embed_text` becomes:
```
"{paper_title}\n{section_heading}\n{chunk_text}"
```
While the returned `text` stays the original paragraph (preserving `source_spans`).

#### Why It Helps
- The encoder gets a **disambiguating signal**: "this passage is about the FacTweet paper's activation function experiments"
- Attacks the **wrong_paper** problem (40.2% of failures) directly
- Nearly zero cost: 14 ms vs 13 ms latency

#### Example
```
Chunk text: "We use dropout on attention coefficients."
Embedded text: "FacTweet\nExperiments\nWe use dropout on attention coefficients."
```
The encoder sees the paper title and section heading, helping it distinguish between papers that discuss similar topics.

---
