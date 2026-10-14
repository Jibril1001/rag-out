"""Baseline chunking sanity check + chunking ceiling (no models needed)."""
import statistics as st
from collections import defaultdict

from ragopt.chunking.fixed import chunk_paper_fixed
from ragopt.data import load_all
from ragopt.headroom import oracle_ranking
from ragopt.metrics import GoldIndex, evaluate

SIZE, OVERLAP = 1000, 200

dev, held, _ = load_all()

for sp in (dev, held):
    gi = GoldIndex(sp.papers)
    by_paper = {pid: chunk_paper_fixed(p, SIZE, OVERLAP) for pid, p in sp.papers.items()}
    n_chunks = sum(len(v) for v in by_paper.values())
    lens = [len(c.text) for v in by_paper.values() for c in v]
    print(f"\n===== {sp.name}: baseline chunks (size={SIZE}, overlap={OVERLAP}) =====")
    print(f"chunks={n_chunks}  chars: mean={st.mean(lens):.0f} min={min(lens)}")

    for mode in ("union", "single"):
        ranked = {q.qid: oracle_ranking(by_paper[q.paper_id], q, gi, 10) for q in sp.questions if q.gold_pids}
        summary, res = evaluate(ranked, sp.questions, gi, mode)
        print(f"chunking ceiling [{mode:6}]: n={summary['n']}  "
              f"R@5={summary['R@5']:.3f}  R@10={summary['R@10']:.3f}  MRR@10={summary['MRR@10']:.3f}")

    # how many gold paragraphs does the baseline cut across chunk boundaries?
    whole = total = 0
    for q in sp.questions:
        for pid in q.gold_pids:
            total += 1
            if any(any(s.paragraph_id == pid and s.start == 0 and
                       s.end == len(sp.papers[q.paper_id].by_pid()[pid].text) for s in c.spans)
                   for c in by_paper[q.paper_id]):
                whole += 1
    print(f"gold paragraphs fully inside ONE chunk: {whole}/{total} = {whole/total:.1%}")
