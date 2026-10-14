r"""Check submission JSONL files against the task's rules.

  python scripts\validate.py results\dev_*.jsonl results\heldout_*.jsonl
  python scripts\validate.py --stage retrieval ...   # answers for the 20 may still be null
"""
import argparse
import glob
import json
import re
import sys


def expand(paths):
    """Let `results\\dev_*.jsonl` work even in shells that do not glob (PowerShell)."""
    out = []
    for p in paths:
        hits = sorted(glob.glob(p)) or [p]
        out.extend(hits)
    return out


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def validate_file(path, split, papers, answer_qids, stage="final"):
    errors = []
    expected = {q.qid: q for q in split.questions}
    seen = set()
    n_answers = 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            rec = json.loads(line)
            tag = f"{path}:{ln}"
            need = {"question_id", "target_paper_id", "retrieved", "answer", "abstained", "latency_ms", "llm_calls"}
            if set(rec) != need:
                errors.append(f"{tag} keys differ: {sorted(set(rec) ^ need)}")
                continue
            qid = rec["question_id"]
            if qid not in expected:
                errors.append(f"{tag} unknown question {qid}"); continue
            if qid in seen:
                errors.append(f"{tag} duplicate question {qid}")
            seen.add(qid)
            if rec["target_paper_id"] != expected[qid].paper_id:
                errors.append(f"{tag} wrong target_paper_id")
            ret = rec["retrieved"]
            if len(ret) != 10:
                errors.append(f"{tag} has {len(ret)} chunks, need 10"); continue
            if [c["rank"] for c in ret] != list(range(1, 11)):
                errors.append(f"{tag} ranks are not 1..10")
            if len({c["chunk_id"] for c in ret}) != 10:
                errors.append(f"{tag} chunk_ids not unique")
            if sum(len(c["text"]) for c in ret) > 12000:
                errors.append(f"{tag} total text > 12000 chars")
            for c in ret:
                if c["paper_id"] not in papers:
                    errors.append(f"{tag} {c['chunk_id']} paper not in split"); continue
                if not c["chunk_id"].startswith(c["paper_id"] + ":"):
                    errors.append(f"{tag} chunk_id/paper_id mismatch {c['chunk_id']}")
                if not c["source_spans"]:
                    errors.append(f"{tag} {c['chunk_id']} has no spans")
                byp = papers[c["paper_id"]].by_pid()
                text = norm_ws(c["text"])
                for s in c["source_spans"]:
                    p = byp.get(s["paragraph_id"])
                    if p is None:
                        errors.append(f"{tag} bad paragraph_id {s['paragraph_id']}"); continue
                    if not (0 <= s["start"] < s["end"] <= len(p.text)):
                        errors.append(f"{tag} span out of range {s}"); continue
                    if norm_ws(p.text[s["start"]:s["end"]]) not in text:
                        errors.append(f"{tag} span text not found in chunk text {s}")
            ans, ab = rec["answer"], rec["abstained"]
            if qid in answer_qids and not (stage == "retrieval" and ans is None and ab is None):
                n_answers += 1
                if not isinstance(ab, bool) or not isinstance(ans, str):
                    errors.append(f"{tag} answer question needs str answer + bool abstained")
                elif ab and ans != "":
                    errors.append(f"{tag} abstained=true needs answer=''")
                elif not ab and not ans.strip():
                    errors.append(f"{tag} empty answer needs abstained=true")
            elif qid not in answer_qids and (ans is not None or ab is not None):
                errors.append(f"{tag} answer/abstained must be null outside the 20")
    missing = set(expected) - seen
    if missing:
        errors.append(f"{path}: {len(missing)} questions missing")
    return errors, len(seen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stage", default="final", choices=["final", "retrieval"])
    a = ap.parse_args()
    from ragopt.data import load_all
    dev, held, aq = load_all()
    bad = 0
    for p in expand(a.files):
        sp = dev if "dev_" in p.replace("\\", "/").split("/")[-1] else held
        errs, n = validate_file(p, sp, sp.papers, aq, a.stage)
        print(("FAIL " if errs else "OK   ") + f"{p} ({n} records)")
        for e in errs[:10]:
            print("   ", e)
        bad += bool(errs)
    sys.exit(1 if bad else 0)
