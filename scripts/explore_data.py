import random
from datasets import load_dataset

# --- the task's exact split code ---
url = ("hf://datasets/allenai/qasper@0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46/"
       "qasper/validation-00000-of-00001.parquet")
ds = load_dataset("parquet", data_files={"validation": url}, split="validation")
paper_ids = sorted(set(ds["id"]))
random.Random(42).shuffle(paper_ids)
build_ids, held_out_ids = paper_ids[:80], paper_ids[80:130]
held = set(held_out_ids)
qids = sorted({str(qid) for paper in ds if paper["id"] in held
               for qid in (paper["qas"]["question_id"] if isinstance(paper["qas"], dict)
                           else [q["question_id"] for q in paper["qas"]])})
random.Random(42).shuffle(qids)
ANSWER_QIDS = set(qids[:20])
# -----------------------------------

def show(x, depth=0, maxdepth=5, key=""):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{key} dict keys={list(x.keys())}")
        if depth < maxdepth:
            for k, v in x.items():
                show(v, depth + 1, maxdepth, k)
    elif isinstance(x, list):
        print(f"{pad}{key} list len={len(x)}")
        if x and depth < maxdepth:
            show(x[0], depth + 1, maxdepth, key + "[0]")
    else:
        print(f"{pad}{key} {type(x).__name__}: {str(x)[:80]!r}")

print("columns:", ds.column_names)
print("total papers:", len(ds), "| dev:", len(build_ids), "| held-out:", len(held_out_ids))

n_dev_q = n_held_q = 0
for p in ds:
    nq = len(p["qas"]["question_id"]) if isinstance(p["qas"], dict) else len(p["qas"])
    if p["id"] in held:
        n_held_q += nq
    elif p["id"] in set(build_ids):
        n_dev_q += nq
print("dev questions:", n_dev_q, "| held-out questions:", n_held_q)
print("answer qids:", len(ANSWER_QIDS), sorted(ANSWER_QIDS)[:3], "...")

print("\n===== STRUCTURE OF ONE PAPER =====")
show(ds[0])