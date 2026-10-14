r"""Export one preset to results/<split>_<preset>.jsonl in the task's format (retrieval only).

  python scripts\export_jsonl.py --preset baseline --split dev
  python scripts\export_jsonl.py --preset final    --split heldout --confirm-heldout
Presets: baseline, A, B, C, final  (held-out allows only baseline and final).
Latency: median of single-query retrieval calls on a seeded sample of questions
(warm models, cross-encoder scores NOT cached), applied to every record.
"""
import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from ragopt.data import load_all
from ragopt.embed import Embedder
from ragopt.pipeline import Config, build_chunks, retrieve
from ragopt.rerank import CrossEncoderScorer

PRESETS = {
    "baseline": dict(chunker="fixed"),
    "A": dict(chunker="structure", context=True),
    "B": dict(retriever="hybrid"),
    "C": dict(rerank_pool=30),
    "final": dict(chunker="structure", context=True, retriever="hybrid", rerank_pool=30),
}

ap = argparse.ArgumentParser()
ap.add_argument("--preset", required=True, choices=list(PRESETS))
ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
ap.add_argument("--confirm-heldout", action="store_true")
ap.add_argument("--timing-sample", type=int, default=15)
a = ap.parse_args()
if a.split == "heldout":
    if a.preset not in ("baseline", "final") or not a.confirm_heldout:
        raise SystemExit("Held-out: only baseline/final, and pass --confirm-heldout.")

cfg = Config(name=a.preset, **PRESETS[a.preset])
dev, held, _ = load_all()
sp = dev if a.split == "dev" else held
qs = sp.questions
chunks = build_chunks(cfg, sp.papers)
emb = Embedder(cache_dir=".cache")
scorer = CrossEncoderScorer(cfg.reranker, cache_dir=".cache") if cfg.rerank_pool else None
idx = retrieve(cfg, chunks, [q.text for q in qs], emb, scorer)

# --- timing pass: single queries, warm models, fresh (empty) reranker cache ---
t_scorer = CrossEncoderScorer(cfg.reranker, cache_dir=tempfile.mkdtemp()) if cfg.rerank_pool else None
retrieve(cfg, chunks, ["warm up query"], emb, t_scorer)
sample = random.Random(42).sample(qs, min(a.timing_sample, len(qs)))
times = []
for q in sample:
    t0 = time.perf_counter()
    retrieve(cfg, chunks, [q.text], emb, t_scorer)
    times.append((time.perf_counter() - t0) * 1000)
lat = round(statistics.median(times))
print(f"retrieval latency median over {len(sample)} single queries: {lat} ms")

out = Path("results"); out.mkdir(exist_ok=True)
path = out / f"{a.split}_{a.preset}.jsonl"
with open(path, "w", encoding="utf-8") as f:
    for i, q in enumerate(qs):
        rec = {
            "question_id": q.qid, "target_paper_id": q.paper_id,
            "retrieved": [chunks[j].to_record(r + 1) for r, j in enumerate(idx[i][:10])],
            "answer": None, "abstained": None, "latency_ms": lat, "llm_calls": 0,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
Path("results/experiments").mkdir(parents=True, exist_ok=True)
Path(f"results/experiments/timing_{a.split}_{a.preset}.json").write_text(
    json.dumps({"median_ms": lat, "samples_ms": times}), encoding="utf-8")
print("wrote", path, f"({len(qs)} records)")
