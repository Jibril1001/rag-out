# Step 2: Chunking and Evidence Preparation

## Why Chunking Matters
Chunking is **not just cutting text** — it determines what a search result can explain on its own. A well-designed chunk keeps related information together, while poor chunking splits related content across multiple chunks, making it impossible for any single chunk to reach the 80% token coverage threshold.

## How to read the numbers: R@10 (union)

Every `R@10 (union)` number in this file uses one rule (defined in `src/ragopt/metrics.py`):

1. Take the retriever's **top-10 chunks**; only chunks from the **target paper** count.
2. A gold paragraph is **covered** when the merged (`union`) source spans of those chunks
   contain **≥80% of its word tokens** — several small chunks can cover one paragraph
   together. (The other mode, `single`, requires one chunk alone to reach 80%.)
3. **R@10** = covered gold paragraphs ÷ total gold paragraphs for that question,
   averaged over the questions that have paragraph gold (261 dev / 155 held-out).

So `R@10 (union): 0.212` means *"on average, 21% of a question's gold paragraphs are
≥80% present within the top 10 results."* How to judge a value: **0.21 baseline is
weak → 0.35 (our final) is a solid +67% relative gain → ~0.65 is the realistic ceiling**
(searching only inside the correct paper gives 0.651) **→ 1.00 is the theoretical
ceiling** if chunking never splits evidence (0.69 in single mode).

## Baseline: Fixed-Window Chunking

### Configuration
- **1,000-character chunks** with **200-character overlap**
- Sliding window over concatenated paper text
- Each chunk records exact `source_spans` back into original paragraphs

### How It Works
1. Concatenate all paper text (with `\n\n` separators between paragraphs)
2. Slide a 1,000-char window with 200-char stride
3. For each window, record which original paragraphs (`source_spans`) are included
4. The chunk `text` is the concatenated segment; `embed_text` is the same by default

### Why It Fails (Theory)
- **Boundary split**: A gold paragraph can be split across two windows, so no single chunk reaches 80% coverage
- **Unrelated headings**: Fixed windows may mix unrelated content (e.g., a section heading) into a chunk
- **Paper confusion**: Query against all papers can confuse two papers discussing the same topic

### Numbers from Baseline
- R@10 (union): 0.212
- Boundary split failures: 10.0% of dev questions
- 3.0 of 10 retrieved chunks overlap another chunk

---

## Optimization A: Structure-Aware Chunking

### Configuration Options
| Parameter | Default | Description |
|---|---|---|
| `chunker` | `"fixed"` | Choose `"structure"` for optimization A |
| `max_chars` | `1000` | Max chars per chunk (structure mode) |
| `merge_below` | `300` | Merge units shorter than this into next unit |
| `context` | `False` | Prepend title + section heading to embedded text |

### How Structure-Aware Chunking Works
1. **Group paragraphs by section** (including "abstract" as its own section)
2. **Split long paragraphs** at sentence boundaries (never mid-sentence)
3. **Merge short paragraphs** in the same section up to `merge_below` (300 chars)
4. **Never cross section boundaries** — each chunk stays within one section
5. **When `context=True`**: Prepend `title | section heading` to `embed_text` only
   - Returned text stays original (so `source_spans` remain exact)

### Why A Works (Theory)
Two effects:
1. **Boundary snapping**: Gold paragraphs stay inside single chunks, reaching 80% coverage
2. **Title/heading context**: Disambiguates papers that talk about the same topic

### Numbers from Optimization A
- R@10 (union): 0.287 (+0.075 over baseline)
- Nearly free: 14 ms vs 13 ms latency
- Fixes boundary_split (10.0% of failures)

### Small-to-Big Retrieval (Planned but Not Built)
- Index short units (2-3 sentences) for precision
- But return the containing parent paragraph for context
- Separates the best unit for *finding* from the best unit for *explaining*

**Why it would have helped here:** a short snippet matches a query precisely, so the
right paper climbs higher in the ranking (fewer `wrong_paper` failures); and once that
snippet hits, returning its full parent paragraph covers the gold paragraph in one span
at ~100%, which directly feeds the ≥80% R@10 rule above. Optimization A already covers
part of this — it merges short paragraphs (≤300 chars) and never splits mid-sentence —
so `s2b` was left at the cut line. Status: logged as `s2b` in `decisions.md` and
`PRESENTATION.md` §9.5; ingestion stage, zero LLM cost; never built.

### Other Chunking Options (Planned but Not Built)
- **MMR + dedup**: Drop chunks whose spans overlap >50% with a higher-ranked one
- **LLM contextual enrichment**: Prepend one-sentence "what this chunk refers to" to embedded text (cost: 1 call/chunk at ingestion)
- **Semantic chunking**: Cut when sentence meaning changes (adds processing, uneven sizes)

---
