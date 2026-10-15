# Step 1: Dataset and Fixed Setup

## QASPER Dataset and Split Configuration

### Dataset Overview
The **QASPER validation split** from HuggingFace (`allenai/qasper`) — a dataset of information-seeking questions and answers anchored in research papers, with detailed evidence annotations collected from human annotators.

### Raw Data Structure (Full Example)
Here is a sample raw QASPER parquet record showing the complete `full_text` field structure:

```json
{
  "section_name": [
    "Introduction",
    "Related Work",
    "Proposed Method",
    "Proposed Method ::: Polarity Function",
    "Proposed Method ::: Discourse Relation-Based Event Pairs",
    "Proposed Method ::: Discourse Relation-Based Event Pairs ::: AL (Automatically Labeled Pairs)",
    "Proposed Method ::: Discourse Relation-Based Event Pairs ::: CA (Cause Pairs)",
    "Proposed Method ::: Discourse Relation-Based Event Pairs ::: CO (Concession Pairs)",
    "Proposed Method ::: Loss Functions",
    "Experiments",
    "Experiments ::: Dataset",
    "Experiments ::: Dataset ::: AL, CA, and CO",
    "Experiments ::: Dataset ::: ACP (ACP Corpus)",
    "Experiments ::: Model Configurations",
    "Experiments ::: Results and Discussion",
    "Conclusion",
    "Acknowledgments",
    "Appendices ::: Seed Lexicon ::: Positive Words",
    "Appendices ::: Seed Lexicon ::: Negative Words",
    "Appendices ::: Settings of Encoder ::: BiGRU",
    "Appendices ::: Settings of Encoder ::: BERT"
  ],
  "paragraphs": [
    [
      "Affective events BIBREF0 are events that typically affect people in positive or negative ways..."
    ],
    [
      "Learning affective events is challenging because..."
    ],
    [
      "In this paper, we propose a simple and effective method..."
    ],
    [
      "We trained the models using a Japanese web corpus..."
    ],
    [
      ""
    ],
    [
      "",
      "Our goal is to learn the polarity function p(x)..."
    ],
    [
      "Our method requires a very small seed lexicon..."
    ],
    [
      "The seed lexicon matches (1) the latter event but (2) not the former event..."
    ],
    [
      "The seed lexicon matches neither the former nor the latter event..."
    ],
    [
      "The seed lexicon matches neither the former nor the latter event..."
    ],
    [
      "Using AL, CA, and CO data, we optimize the parameters..."
    ],
    [
      ""
    ],
    [
      ""
    ],
    [
      "As a raw corpus, we used a Japanese web corpus..."
    ],
    [
      "We used the latest version of the ACP Corpus..."
    ],
    [
      "As for Encoder, we compared two types of neural networks..."
    ],
    [
      "",
      "Table TABREF23 shows accuracy..."
    ],
    [
      "We thank Nobuhiro Kaji for providing the ACP Corpus..."
    ],
    [
      "喜ぶ (rejoice), 嬉しい (be glad), ..."
    ],
    [
      "怒る (get angry), 悲しい (be sad), ..."
    ],
    [
      "The dimension of the embedding layer was 256..."
    ],
    [
      "We used a Japanese BERT model pretrained..."
    ]
  ]
}
```

### Split Configuration
- **80 development papers** (`dev` split): Used for tuning and debugging optimizations
- **50 held-out papers** (`heldout` split): Used for final comparison only (never seen during tuning)
- **Total papers in QASPER validation split**: **281 papers** (pinned HF revision `0fbdb8ed…`); this project uses 130 of them via the seeded shuffle
- **How the split works**: Not "first 130 rows" — the split is **reproducible and shuffled**:
  1. All paper IDs from the QASPER validation split are collected
  2. IDs are sorted, then shuffled with `random.Random(42)` (seed 42)
  3. First 80 shuffled IDs → **dev** split
  4. Next 50 shuffled IDs → **held-out** split
  5. Remaining IDs are excluded from this project (not used)
- **Seed 42**: Guarantees the same dev/held-out split every time
- **Disjoint splits**: dev and held-out share no papers; tuning on dev cannot leak into held-out score

### Which Splits Were Used

| Split | Purpose | Questions | Papers | With paragraph gold |
|-------|---------|-----------|--------|---------------------|
| **dev** | Tuning, debugging, all optimizations | 295 | 80 | 261 (88.5%) |
| **held-out** | Final evaluation only (no tuning) | 182 | 50 | 155 (85.2%) |
| **Combined** | Total dataset used in this project | 477 | 130 | — |

> **Note**: No separate `test` split was used. The held-out 50 papers serve as the final test evaluation. The 20 answer questions are sampled from held-out only.

### The 295 + 182 Question Distribution

| Metric | dev | held-out |
|--------|---:|---: |
| Total questions | 295 | 182 |
| With paragraph gold | 261 | 155 |
| No paragraph gold (excluded from Recall/MRR) | 34 | 27 |
| └ of which float/table evidence only | 10 | 11 |
| Unanswerable (majority annotators) | 19 | 12 |

> **34 dev (11.5%)** and **27 held-out (14.8%)** questions have **no paragraph-level gold**
> (unanswerable, or evidence that is only `FLOAT SELECTED …` / unmatched text) and are
> **excluded from Recall/MRR** because there is no paragraph to measure token coverage
> against. Of these, only 10 (dev) and 11 (held-out) are purely figure/table evidence.

### Clarification on Row Counts

The QASPER **validation** split (pinned HF revision `0fbdb8ed…`) contains **281 papers**.
This project uses **130** of them — 80 dev + 50 held-out selected by the seeded shuffle —
and leaves the remaining **151** unused.

| Split | Papers | Questions | Notes |
|-------|--------|-----------|-------|
| **dev** | 80 | 295 | Used for tuning all optimizations |
| **held-out** | 50 | 182 | Used for final evaluation only |
| **Combined** | 130 | 477 | Total papers/questions used in this project |

> **About "281":** `281` is exactly the number of papers in the QASPER validation split.
> (`tests/test_data.py` independently uses `range(281)` for fake IDs when unit-testing
> `split_ids()` — a deliberate match, not a contradiction.)
>
> **About "888 / 416":** those are counts from *other* QASPER splits or other datasets;
> they are not part of the validation split used here.
>
> **The 130 papers are not the "first 130 rows".** `split_ids()` in `src/ragopt/data.py`
> sorts all 281 IDs, shuffles with `random.Random(42)`, then takes `[:80]` (dev) and
> `[80:130]` (held). Because the revision is pinned, the exact 130 papers — and the
> 80/50 ratio — are fixed and reproducible.

### JSONL File Row Counts

The actual JSONL output files contain one record per question:

| File | Records | Questions |
|------|---------|-----------|
| `results/dev_baseline.jsonl` | 295 | All dev questions |
| `results/heldout_baseline.jsonl` | 182 | All held-out questions |
| `results/dev_final.jsonl` | 295 | Dev with final retrieval |
| `results/heldout_final.jsonl` | 182 | Held-out with final retrieval |

Each record contains 10 retrieved chunks with source spans, as specified in the task brief.

### Full Text Structure: `section_name` + `paragraphs`

The QASPER dataset stores each paper's text as two parallel arrays:

| Field | Description |
|---|---|
| **`section_name`** | Array of section headings (37 entries in this example) |
| **`paragraphs`** | Parallel array of paragraph groups. Each entry is itself an array of paragraph texts. |

**Key observations from the example**:
- There are **37 section headings** and **37 paragraph groups** (parallel arrays)
- Some paragraph groups may be **empty** (`[]`) — e.g., the 5th group `"Proposed Method ::: Discourse Relation-Based Event Pairs"` has `[""]` (one empty string)
- Some groups have **multiple paragraphs** — e.g., the last group has 2 paragraphs: `[". 重大な失敗を犯したので、仕事をクビになった。", "Because [I] made a serious mistake, [I] got fired."]`
- Section names can contain **hierarchical info** with `:::` separators (e.g., `"Proposed Method ::: Discourse Relation-Based Event Pairs ::: AL"`)

### How Paragraph IDs (gold_pids) Map to This Structure

From `src/ragopt/data.py`, the mapping works like this:

```python
def map_evidence(paper: Paper) -> dict[str, str]:
    """Normalized paragraph text -> pid (first occurrence wins)."""
    out: dict[str, str] = {}
    for p in paper.paragraphs:
        out.setdefault(norm_ws(p.text), p.pid)
    return out

# Later, when building questions:
for ev in a["evidence"]:  # each answer's evidence list
    if ev.startswith(FLOAT_PREFIX):  # "FLOAT SELECTED"
        q.n_float_evidence += 1
        continue
    pid = lookup.get(norm_ws(ev))  # map text to pid
    if pid is None:
        q.n_unmatched_evidence += 1
    else:
        q.gold_pids.add(pid)  # add to gold set
```

**PID format**: `"abstract"` or `"s0:p4"` (section 0, paragraph 4, zero-based)

**Mapping example**:
- If an answer's evidence is `"Learning affective events is challenging because..."`
- And that text appears in `paragraphs[1]` (the second paragraph group)
- Then the PID is `"s1:p0"` (section 1, paragraph 0)
- If the evidence is `"The dimension of the embedding layer was 256."`
- And that appears in `paragraphs[34]` (the last group)
- Then the PID is `"s34:p0"`

### Key Rule: Target Paper Search
- **Queries search ALL papers** in the split — never filter to the target paper
- **Only evidence from the target paper counts as correct**
- This is the core design principle: "retrieval-first, measure-don't-assume"
- The `wrong_paper` failure mode (40.2%) occurs when dense retrieval ranks a *different* paper in the top-10

### Split Code (from data.py)

```python
def split_ids(paper_ids_all):
    """The task's exact reproducible split (80 dev / 50 held-out)."""
    ids = sorted(set(paper_ids_all))
    random.Random(SEED).shuffle(ids)  # Seed 42 for reproducibility
    return ids[:80], ids[80:130]  # 80 dev, 50 held-out

def answer_qids(held_questions_qids) -> set[str]:
    """The task's exact 20 answer question IDs (sampled from held-out)."""
    qids = sorted({str(q) for q in held_questions_qids})
    random.Random(SEED).shuffle(qids)  # Seed 42 for reproducibility
    return set(qids[:20])  # 20 fixed questions for answer generation
```

### Embedding Model
- **BAAI/bge-small-en-v1.5**: Bi-encoder, L2-normalized
- **Downloaded from HuggingFace hub** on first run (not pre-local in repo)
- Inner product on normalized vectors = cosine similarity
- Queries must be prefixed: `"Represent this sentence for searching relevant passages: "`
- Passages are embedded raw (no prefix)
- The model is cached after first download, so subsequent runs are fast

### Answer Generator
- **gemini-3.5-flash-lite**, temperature 0
- Used only for the 20 fixed held-out questions
- JSON reply format: `{"answer", "abstained"}`
- **Temperature 0** ensures baseline vs final differ only because of retrieval
- Responses cached on disk to avoid repeated LLM calls

### How the 20 Answer Questions Are Selected

```python
# From data.py: answer_qids selection
held = set(held_out_ids)  # 50 held-out paper IDs
# Collect all question qids from held-out papers
qids = sorted({str(q) for paper in ds if paper["id"] in held ...})
random.Random(42).shuffle(qids)  # seed 42 for reproducibility
ANSWER_QIDS = set(qids[:20])  # first 20 are the fixed questions
```

### Demo: Loading a Single Paper and Question

```python
from src.ragopt.data import load_all

dev, held, ANSWER_QIDS = load_all()

# Print first dev question
q = dev.questions[0]
print(f"QID: {q.qid}")
print(f"Paper: {q.paper_id}")
print(f"Question: {q.text}")
print(f"Gold PIDs: {q.gold_pids}")
print(f"Float evidence: {q.n_float_evidence}")
print(f"Unmatch: {q.n_unmatched_evidence}")
print(f"Unanswerable: {q.unanswerable}")
```

**Output example**:
```
QID: q001
Paper: paper-17
Question: What aspects have been compared between various language models?
Gold PIDs: {'s0:p3', 's0:p5'}
Float evidence: 0
Unmatch: 0
Unanswerable: False
```

### Where is the Gold PID?
Gold PIDs are **not** in the raw QASPER data directly — they are **computed** during data loading (`src/ragopt/data.py`, function `build_questions`):

1. Each answer has an `evidence` list referencing sections or paragraphs
2. The code maps evidence text to PIDs via `map_evidence(paper)` which creates a lookup from normalized paragraph text → PID
3. If evidence matches a paragraph text, that paragraph's PID (`"s0:p4"` or `"abstract"`) is added to the question's `gold_pids` set
4. Evidence starting with `FLOAT_PREFIX` (`"FLOAT SELECTED"`) are counted as figure/table evidence and **do not** get a paragraph PID

So the gold PIDs you see in the repo (`"s0:p4"`, `"abstract"`) are **derived** from the answer evidence annotations in the original QASPER dataset.

### Was the Embedding Model Local?
**No — the embedding model is downloaded from HuggingFace hub on first run.**

- Model: `BAAI/bge-small-en-v1.5`
- Downloaded via `sentence-transformers` or `datasets` library
- **Cached after first download** — subsequent runs use the cached local copy
- The repo does **NOT** pre-package the model weights (they would be ~200MB+)
- `requirements.lock.txt` pins the exact version used, but weights are downloaded separately

**Code to load it:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # downloads once
```

- The model is **not** in the git repo and is not version-controlled
- It's cached at `~/.cache/huggingface/...` after first download

### How the `full_text` Structure Informs Chunking

The `section_name` + `paragraphs` structure directly impacts chunking decisions:

| Feature | Impact on Chunking |
|---|---|
| **37 sections with hierarchical headings** | Optimization A (structure-aware) groups paragraphs by section, never crosses section boundaries |
| **Some empty paragraph groups** (`[]`) | Baseline fixed chunking may include empty space; structure-aware skips them |
| **Hierarchical section names** (`"A ::: B ::: C"` | Title/heading context in embeddings (Optimization A `context=True`) prepends disambiguating signals |
| **Paragraphs with varying lengths** | `merge_below=300` in structure-aware mode merges short paragraphs, splits long ones at sentence boundaries |
| **Section "abstract" is its own entry** | Abstract is treated as its own section; chunks never cross into it from body text |

### Summary: Data Flow From Raw QASPER to RAG Pipeline

```
Raw QASPER parquet
    │
    ├── id, title, abstract
    ├── full_text: {section_name, paragraphs[][]}
    │        │
    │        └─► data.py: build_paper() ──► Paper objects with Paragraph pids ("abstract", "s0:p4")
    │                    │
    │                    └─► data.py: map_evidence() ──► normalized_text → pid lookup
    │                                        │
    │                                        └─► data.py: build_questions()
    │                                               │
    │                                               ├───► gold_pids set (for metrics)
    │                                               ├───► n_float_evidence count
    │                                               └─► n_unmatched_evidence count
    │
    └─► Questions with qid, paper_id, text, gold_pids, unanswerable
          │
          ├──► Embedding model (bge-small-en-v1.5) ──► vectors
          │        │
          │        └► Retrieval (dense + BM25 + RRF + cross-encoder)
          │                 │
          │                 └► JSONL output (7 files)
          │
          └► Answer generation (20 questions, gemini-3.5-flash-lite)
```

---