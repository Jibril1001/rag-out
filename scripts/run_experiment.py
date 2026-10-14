"""Run one retrieval experiment and save per-question results.

Examples (dev):
  python scripts\run_experiment.py --name bm25_fixed --retriever bm25
  python scripts\run_experiment.py --name hybrid_fixed --retriever hybrid
  python scripts\run_experiment.py --name struct_ctx --chunker structure --context
  python scripts\run_experiment.py --name struct_ctx_hybrid --chunker structure --context --retriever hybrid
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ragopt.analysis.funnel import CATEGORIES, classify
from ragopt.data import load_all
from ragopt.embed import Embedder
from ragopt.metrics import GoldIndex, evaluate
from ragopt.pipeline import Config, build_chunks, retrieve
from ragopt.rerank import CrossEncoderScorer

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
ap.add_argument("--confirm-heldout", action="store_true",
                help="required for --split heldout (never tune on held-out!)")
ap.add_argument("--chunker", default="fixed", choices=["fixed", "structure"])
ap.add_argument("--size", type=int, default=1000)
ap.add_argument("--overlap", type=int, default=200)
ap.add_argument("--max-chars", type=int, default=1000)
ap.add_argument("--merge-below", type=int, default=300)
ap.add_argument("--context", action="store_true")
ap.add_argument("--retriever", default="dense", choices=["dense", "bm25", "hybrid"])
ap.add_argument("--pool", type=int, default=50)
ap.add_argument("--rrf-k", type=int, default=60)
ap.add_argument("--rerank-pool", type=int, default=0, help="cross-encoder reorders the top-N (0 = off)")
ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
a = ap.parse_args()
if a.split == "heldout" and not a.confirm_heldout:
    raise SystemExit("Refusing: held-out is only for the baseline and the ONE final. Pass --confirm-heldout.")

cfg = Config(a.name, a.chunker, a.size, a.overlap, a.max_chars, a.merge_below, a.context,
             a.retriever, a.pool, a.rrf_k,
             rerank_pool=a.rerank_pool, reranker=a.reranker)
dev, held, _ = load_all()
sp = dev if a.split == "dev" else held
chunks = build_chunks(cfg, sp.papers)
lens = [len(c.text) for c in chunks]
print(f"[{cfg.name}] {len(chunks)} chunks over {len(sp.papers)} papers (mean {sum(lens)/len(lens):.0f} chars)")

emb = Embedder(cache_dir=".cache")
t0 = time.time()
qs = sp.questions
scorer = CrossEncoderScorer(cfg.reranker, cache_dir=".cache") if cfg.rerank_pool else None
idx = retrieve(cfg, chunks, [q.text for q in qs], emb, scorer)
print(f"retrieval done in {time.time() - t0:.1f}s  embedder={emb.stats}" + (f"  reranker={scorer.stats}" if scorer else ""))

gi = GoldIndex(sp.papers)
ranked = {q.qid: [chunks[j] for j in idx[i][:10]] for i, q in enumerate(qs)}
summary, per_q = {}, {}
for mode in ("union", "single"):
    s, res = evaluate(ranked, qs, gi, mode)
    summary[mode] = s
    for r in res:
        per_q.setdefault(r.qid, {})[mode] = [r.r5, r.r10, r.rr]
    print(f"[{mode:6}] n={s['n']}  R@5={s['R@5']:.3f}  R@10={s['R@10']:.3f}  MRR@10={s['MRR@10']:.3f}")

gq = [(i, q) for i, q in enumerate(qs) if q.gold_pids]
labels = {q.qid: classify(q, [chunks[j] for j in idx[i][:10]], [chunks[j] for j in idx[i]], gi) for i, q in gq}
for qid, lab in labels.items():
    per_q[qid]["label"] = lab
cnt = Counter(labels.values())
print("funnel:", {c: cnt[c] for c in CATEGORIES})
summary["paper_top1"] = sum(chunks[idx[i][0]].paper_id == q.paper_id for i, q in gq) / len(gq)
summary["paper_share_top10"] = sum(sum(chunks[j].paper_id == q.paper_id for j in idx[i][:10]) / 10 for i, q in gq) / len(gq)
print(f"top-1 from target paper: {summary['paper_top1']:.1%} | share of top-10: {summary['paper_share_top10']:.1%}")

out = Path("results/experiments"); out.mkdir(parents=True, exist_ok=True)
(out / f"{a.split}_{cfg.name}.json").write_text(
    json.dumps({"config": cfg.to_dict(), "split": a.split, "n_chunks": len(chunks),
                "summary": summary, "per_question": per_q}, indent=1), encoding="utf-8")
print("saved", out / f"{a.split}_{cfg.name}.json")
