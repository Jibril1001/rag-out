"""Why do some evidence strings fail to map to a paragraph? Classify them."""
import difflib
from collections import Counter
from ragopt.data import URL, FLOAT_PREFIX, build_paper, norm_ws, split_ids


def classify(ev: str, paper):
    """Return (category, detail) for one unmatched evidence string."""
    e = norm_ws(ev)
    paras = [(p.pid, norm_ws(p.text)) for p in paper.paragraphs]
    inside = [pid for pid, t in paras if e and e in t]
    if inside:
        return "substring_of_paragraph", inside
    contained = [pid for pid, t in paras if t and len(t) > 20 and t in e]
    if contained:
        return "spans_multiple_paragraphs", contained
    best_pid, best = None, 0.0
    for pid, t in paras:
        if difflib.SequenceMatcher(None, e, t).quick_ratio() < 0.6:
            continue
        r = difflib.SequenceMatcher(None, e, t).ratio()
        if r > best:
            best_pid, best = pid, r
    if best >= 0.85:
        return "near_match_(typo/unicode)", [(best_pid, round(best, 3))]
    return "no_match", [(best_pid, round(best, 3))]


def main():
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"validation": URL}, split="validation")
    dev_ids, held_ids = split_ids(ds["id"])
    keep = set(dev_ids) | set(held_ids)
    counts, examples = Counter(), {}
    for rec in ds:
        if rec["id"] not in keep:
            continue
        paper = build_paper(rec)
        exact = {norm_ws(p.text) for p in paper.paragraphs}
        for ans in rec["qas"]["answers"]:
            for a in ans["answer"]:
                for ev in a["evidence"]:
                    if ev.startswith(FLOAT_PREFIX) or norm_ws(ev) in exact:
                        continue
                    cat, detail = classify(ev, paper)
                    counts[cat] += 1
                    examples.setdefault(cat, []).append((rec["id"], ev, detail))
    print("unmatched evidence by category:", dict(counts))
    for cat, items in examples.items():
        print(f"\n=== {cat} ({len(items)}) ===")
        for pid, ev, detail in items[:4]:
            print(f"- paper {pid} | len={len(ev)} | candidates={detail}")
            print(f"  evidence: {ev[:160]!r}")


if __name__ == "__main__":
    main()
