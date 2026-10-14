r"""Pick 10 baseline failures (2 per category) and write results/failure_cases.md.

  python scripts\failure_cases.py
Read each case yourself and edit 'likely cause' / 'proposed fix' before submitting.
"""
import random
from pathlib import Path

from ragopt.analysis.funnel import classify
from ragopt.data import load_all
from ragopt.embed import Embedder
from ragopt.metrics import GoldIndex
from ragopt.pipeline import Config, build_chunks, retrieve

DRAFT = {
    "wrong_paper": ("Question is generic or shares vocabulary with other papers; dense search over all papers "
                    "has no paper signal.", "Add title/section context to the embedded text (A); add BM25 for exact terms (B)."),
    "boundary_split": ("Fixed 1000-char window cut the gold paragraph, so no single chunk covers it.",
                       "Structure-aware chunking that respects paragraph/section boundaries (A)."),
    "ranked_too_low": ("Gold chunk is in the top-50 but weak similarity scores rank it below the top 10.",
                       "Cross-encoder rerank of the top-30 candidates (C)."),
    "not_found": ("Question wording differs from the paper's wording (query mismatch); gold not in top-50.",
                  "Lexical match via BM25 fusion (B); query rewriting/HyDE if LLM cost is acceptable."),
    "incomplete_multi_evidence": ("Several gold paragraphs; top-10 is filled with near-duplicates/overlapping chunks "
                                  "from one place.", "Non-overlapping chunks (A); MMR/dedup selection."),
}

cfg = Config(name="baseline")
dev, _, _ = load_all()
chunks = build_chunks(cfg, dev.papers)
qs = dev.questions
idx = retrieve(cfg, chunks, [q.text for q in qs], Embedder(cache_dir=".cache"))
gi = GoldIndex(dev.papers)

by_cat = {}
for i, q in enumerate(qs):
    if not q.gold_pids:
        continue
    lab = classify(q, [chunks[j] for j in idx[i][:10]], [chunks[j] for j in idx[i]], gi)
    by_cat.setdefault(lab, []).append(i)

rng = random.Random(42)
snip = lambda t, n=260: " ".join(t.split())[:n] + ("..." if len(t) > n else "")
lines = ["# Failure analysis: 10 baseline cases (dev)\n"]
n = 0
for cat in DRAFT:
    for i in rng.sample(by_cat.get(cat, []), min(2, len(by_cat.get(cat, [])))):
        q, top = qs[i], chunks[idx[i][0]]
        gold = dev.papers[q.paper_id].by_pid()[sorted(q.gold_pids)[0]].text
        n += 1
        cause, fix = DRAFT[cat]
        lines += [f"## {n}. {q.qid} [{cat}]",
                  f"- **Question:** {q.text}",
                  f"- **Target paper:** {dev.papers[q.paper_id].title}",
                  f"- **Top-1 came from:** {dev.papers[top.paper_id].title}"
                  + ("" if top.paper_id != q.paper_id else " (correct paper)"),
                  f"- **Gold paragraph:** {snip(gold)}",
                  f"- **Top-1 chunk:** {snip(top.text)}",
                  f"- **Observed problem:** {cat.replace('_', ' ')}",
                  f"- **Likely cause (edit me):** {cause}",
                  f"- **Proposed fix (edit me):** {fix}\n"]
Path("results/failure_cases.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote results/failure_cases.md with", n, "cases;",
      {c: len(v) for c, v in by_cat.items()})