"""Sanity-check the loader on the real QASPER data."""
import statistics as st
from ragopt.data import load_all

dev, held, ANSWER_QIDS = load_all()

for sp in (dev, held):
    paras = [p for paper in sp.papers.values() for p in paper.paragraphs]
    lens = [len(p.text) for p in paras]
    qs = sp.questions
    with_gold = [q for q in qs if q.gold_pids]
    print(f"\n===== {sp.name} =====")
    print(f"papers={len(sp.papers)}  paragraphs={len(paras)}  questions={len(qs)}")
    print(f"paragraph chars: median={st.median(lens):.0f}  mean={st.mean(lens):.0f}  "
          f"p90={sorted(lens)[int(.9*len(lens))]}  max={max(lens)}")
    print(f"questions with >=1 gold paragraph: {len(with_gold)}")
    print(f"  unanswerable (majority): {sum(q.unanswerable for q in qs)}")
    print(f"  no gold & not unanswerable: {sum((not q.gold_pids) and (not q.unanswerable) for q in qs)}")
    print(f"gold paragraphs/question (those with gold): "
          f"mean={st.mean(len(q.gold_pids) for q in with_gold):.2f}  "
          f"max={max(len(q.gold_pids) for q in with_gold)}")
    print(f"evidence strings: float/table={sum(q.n_float_evidence for q in qs)}  "
          f"UNMATCHED text={sum(q.n_unmatched_evidence for q in qs)}")
    print(f"questions touched by float evidence: {sum(q.n_float_evidence > 0 for q in qs)}")

print("\n===== the 20 answer questions =====")
heldq = {q.qid: q for q in held.questions}
sel = [heldq[i] for i in sorted(ANSWER_QIDS)]
print("unanswerable among the 20:", sum(q.unanswerable for q in sel))
print("with gold paragraphs:", sum(bool(q.gold_pids) for q in sel))

print("\n===== first 3 dev questions =====")
for q in dev.questions[:3]:
    print(q.qid[:8], "|", q.text, "| gold:", sorted(q.gold_pids), "| unans:", q.unanswerable)