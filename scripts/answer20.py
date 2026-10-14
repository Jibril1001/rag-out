r"""Generate answers for the 20 fixed held-out questions (baseline and final).

  $env:GEMINI_API_KEY = "..."      (PowerShell)
  pip install google-genai
  python scripts\answer20.py
Reads results/heldout_{baseline,final}.jsonl (retrieval-only), writes them back with answers.
One identical grounded prompt for both systems; temperature 0; responses cached on disk.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from google import genai

from ragopt.data import load_all

MODEL = "gemini-3.5-flash-lite"
PROMPT = """Answer the question using ONLY the context passages below.
If the context does not contain the answer, abstain.
Reply with JSON only: {{"answer": "<short answer>", "abstained": false}}
or {{"answer": "", "abstained": true}}.

Context:
{ctx}

Question: {q}"""

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_, held, ANSWER_QIDS = load_all()
qtext = {q.qid: q.text for q in held.questions}
cache = Path(".cache/llm"); cache.mkdir(parents=True, exist_ok=True)
bk = Path("results/experiments"); bk.mkdir(parents=True, exist_ok=True)


def ask(prompt):
    f = cache / (hashlib.sha256((MODEL + prompt).encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    time.sleep(4.5)
    r = client.models.generate_content(
        model=MODEL, contents=prompt,
        config={"temperature": 0, "response_mime_type": "application/json"})
    out = {"text": r.text, "ms": (time.perf_counter() - t0) * 1000}
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


for name in ("baseline", "final"):
    path = Path(f"results/heldout_{name}.jsonl")
    backup = bk / f"heldout_{name}_retrieval_only.jsonl"
    if not backup.exists():
        shutil.copy(path, backup)          # keep the retrieval-only version
    recs = [json.loads(l) for l in backup.read_text(encoding="utf-8").splitlines()]
    n_abs = 0
    for rec in recs:
        if rec["question_id"] not in ANSWER_QIDS:
            continue
        ctx = "\n\n".join(f"[{c['rank']}] {c['text']}" for c in rec["retrieved"])
        resp = ask(PROMPT.format(ctx=ctx, q=qtext[rec["question_id"]]))
        try:
            d = json.loads(resp["text"])
            ans, ab = str(d.get("answer", "")).strip(), bool(d.get("abstained", False))
        except Exception:
            ans, ab = "", True
        if ab or not ans:
            ans, ab = "", True
        n_abs += ab
        rec["answer"], rec["abstained"] = ans, ab
        rec["llm_calls"] = 1
        rec["latency_ms"] = round(rec["latency_ms"] + resp["ms"])
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    print(name, "answered", len(ANSWER_QIDS), "| abstained", n_abs)
