"""First real baseline run on the DEV split (fixed chunks + dense top-10)."""
import time

from ragopt.chunking.fixed import chunk_corpus_fixed
from ragopt.data import load_all
from ragopt.dense import FlatIPIndex
from ragopt.embed import Embedder
from ragopt.metrics import GoldIndex, evaluate

SIZE, OVERLAP, K = 1000, 200, 10

dev, _, _ = load_all()
chunks = chunk_corpus_fixed(dev.papers, SIZE, OVERLAP)      # ONE index over ALL dev papers
print(f"{len(chunks)} chunks over {len(dev.papers)} papers")

emb = Embedder(cache_dir=".cache")
t0 = time.time()
D = emb.encode_docs([c.embed_text for c in chunks])
print(f"doc embeddings {D.shape} in {time.time() - t0:.1f}s  {emb.stats}")

qs = dev.questions
Q = emb.encode_queries([q.text for q in qs])
idx, sc = FlatIPIndex(D).search(Q, K)                       # no filtering by target paper
ranked = {q.qid: [chunks[j] for j in idx[i]] for i, q in enumerate(qs)}

gi = GoldIndex(dev.papers)
for mode in ("union", "single"):
    s, _ = evaluate(ranked, qs, gi, mode)
    print(f"[{mode:6}] n={s['n']}  R@5={s['R@5']:.3f}  R@10={s['R@10']:.3f}  MRR@10={s['MRR@10']:.3f}")

gq = [(i, q) for i, q in enumerate(qs) if q.gold_pids]
top1 = sum(chunks[idx[i, 0]].paper_id == q.paper_id for i, q in gq) / len(gq)
any10 = sum(any(chunks[j].paper_id == q.paper_id for j in idx[i]) for i, q in gq) / len(gq)
frac10 = sum(sum(chunks[j].paper_id == q.paper_id for j in idx[i]) / K for i, q in gq) / len(gq)
print(f"top-1 chunk from target paper: {top1:.1%} | any of top-10: {any10:.1%} | mean share of top-10: {frac10:.1%}")
