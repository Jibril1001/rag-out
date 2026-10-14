"""Failure funnel + cross-paper diagnostic + draft failure table (DEV only, cached embeddings)."""
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ragopt.analysis.funnel import CATEGORIES, DESCRIPTIONS, chunks_overlap, classify
from ragopt.chunking.fixed import chunk_corpus_fixed
from ragopt.data import load_all
from ragopt.dense import top_k
from ragopt.embed import Embedder
from ragopt.metrics import GoldIndex, evaluate

dev, _, _ = load_all()
chunks = chunk_corpus_fixed(dev.papers, 1000, 200)
emb = Embedder(cache_dir=".cache")
D = emb.encode_docs([c.embed_text for c in chunks])
qs = dev.questions
Q = emb.encode_queries([q.text for q in qs])
assert emb.stats["encoded"] == 0, "expected everything to come from the cache"
gi = GoldIndex(dev.papers)

S = Q @ D.T                                           # (questions, chunks)
idx50, _ = top_k(S, 50)
paper_of = np.array([c.paper_id for c in chunks])

# ---- 1. failure funnel -------------------------------------------------------
labels, rows = {}, []
for i, q in enumerate(qs):
    if not q.gold_pids:
        continue
    top50 = [chunks[j] for j in idx50[i]]
    labels[q.qid] = classify(q, top50[:10], top50, gi)
    rows.append((i, q))
cnt = Counter(labels.values())
print("===== FAILURE FUNNEL (baseline, dev, union mode) =====")
for cat in CATEGORIES:
    print(f"{cat:28} {cnt[cat]:4}  {cnt[cat] / len(rows):6.1%}   {DESCRIPTIONS[cat]}")

# ---- 2. redundancy ---------------------------------------------------------------
red = []
for i, q in rows:
    top = [chunks[j] for j in idx50[i][:10]]
    red.append(sum(any(chunks_overlap(a, b) for b in top if b is not a) for a in top))
print(f"\nredundancy: on average {np.mean(red):.1f} of the 10 chunks overlap another top-10 chunk")

# ---- 3. cross-paper diagnostic (NEVER used for submissions) ---------------------
ranked = {}
for i, q in rows:
    masked = np.where(paper_of == q.paper_id, S[i], -np.inf)[None, :]
    ix, _ = top_k(masked, 10)
    ranked[q.qid] = [chunks[j] for j in ix[0]]
print("\n===== DIAGNOSTIC: search restricted to the target paper (not a valid system!) =====")
for mode in ("union", "single"):
    s, _ = evaluate(ranked, qs, gi, mode)
    print(f"[{mode:6}] R@5={s['R@5']:.3f}  R@10={s['R@10']:.3f}  MRR@10={s['MRR@10']:.3f}")

# ---- 4. draft failure table (10 questions, stratified over categories) -----------
rng = random.Random(42)
by_cat = defaultdict(list)
for i, q in rows:
    if labels[q.qid] != "success":
        by_cat[labels[q.qid]].append((i, q))
for v in by_cat.values():
    rng.shuffle(v)
picked, cats = [], [c for c in CATEGORIES if by_cat[c]]
while len(picked) < 10 and any(by_cat[c] for c in cats):
    for c in cats:
        if by_cat[c] and len(picked) < 10:
            picked.append(by_cat[c].pop())

out = ["# Draft failure table (fill in 'cause' and 'fix' by reading each row)\n"]
for i, q in picked:
    paper = dev.papers[q.paper_id]
    g = paper.by_pid()[sorted(q.gold_pids)[0]]
    top1 = chunks[idx50[i][0]]
    out += [f"## {q.qid[:8]} | {labels[q.qid]}",
            f"- **Question:** {q.text}",
            f"- **Target paper:** {paper.title}",
            f"- **Gold ({g.pid}, {g.section}):** {g.text[:220]!r}",
            f"- **Top-1 retrieved (paper: {dev.papers[top1.paper_id].title[:70]}):** {top1.text[:220]!r}",
            "- **Likely cause:** ", "- **Proposed fix:** ", ""]
Path("results").mkdir(exist_ok=True)
Path("results/failure_table_draft.md").write_text("\n".join(out), encoding="utf-8")
print("\nwrote results/failure_table_draft.md with", len(picked), "questions")
