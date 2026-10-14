"""Load QASPER, reproduce the task's fixed split, and expose papers/questions.

Paragraph IDs follow the task spec: "abstract" or "s{section}:p{paragraph}"
(zero-based, indices taken from the ORIGINAL lists so they stay stable even
when a paragraph is empty). Character offsets always refer to the original,
unmodified paragraph text.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

URL = ("hf://datasets/allenai/qasper@0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46/"
       "qasper/validation-00000-of-00001.parquet")
SEED = 42
FLOAT_PREFIX = "FLOAT SELECTED"


@dataclass(frozen=True)
class Paragraph:
    pid: str          # "abstract" | "s0:p4"
    section: str      # section heading ("Abstract" for the abstract)
    text: str         # original text, never modified


@dataclass
class Paper:
    paper_id: str
    title: str
    paragraphs: list[Paragraph]

    def by_pid(self) -> dict[str, Paragraph]:
        return {p.pid: p for p in self.paragraphs}


@dataclass
class Question:
    qid: str
    paper_id: str
    text: str
    gold_pids: set[str] = field(default_factory=set)   # paragraph-level gold
    n_float_evidence: int = 0       # figure/table evidence (no paragraph id)
    n_unmatched_evidence: int = 0   # text evidence we could not map
    unanswerable: bool = False      # majority of annotators


@dataclass
class Split:
    name: str
    papers: dict[str, Paper]
    questions: list[Question]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_paper(rec: dict) -> Paper:
    paras = []
    if rec["abstract"] and rec["abstract"].strip():
        paras.append(Paragraph("abstract", "Abstract", rec["abstract"]))
    ft = rec["full_text"]
    for si, (sname, plist) in enumerate(zip(ft["section_name"], ft["paragraphs"])):
        for pi, text in enumerate(plist):
            if text and text.strip():
                paras.append(Paragraph(f"s{si}:p{pi}", sname or "", text))
    return Paper(rec["id"], rec["title"], paras)


def map_evidence(paper: Paper) -> dict[str, str]:
    """normalized paragraph text -> pid (first occurrence wins)."""
    out: dict[str, str] = {}
    for p in paper.paragraphs:
        out.setdefault(norm_ws(p.text), p.pid)
    return out


def build_questions(rec: dict, paper: Paper) -> list[Question]:
    qas = rec["qas"]
    lookup = map_evidence(paper)
    questions = []
    for i, qid in enumerate(qas["question_id"]):
        q = Question(str(qid), paper.paper_id, qas["question"][i])
        annots = qas["answers"][i]["answer"]
        n_unans = 0
        for a in annots:
            n_unans += bool(a["unanswerable"])
            for ev in a["evidence"]:
                if ev.startswith(FLOAT_PREFIX):
                    q.n_float_evidence += 1
                    continue
                pid = lookup.get(norm_ws(ev))
                if pid is None:
                    q.n_unmatched_evidence += 1
                else:
                    q.gold_pids.add(pid)
        q.unanswerable = n_unans * 2 > len(annots)
        questions.append(q)
    return questions


def split_ids(paper_ids_all):
    """The task's exact reproducible split (80 dev / 50 held-out)."""
    ids = sorted(set(paper_ids_all))
    random.Random(SEED).shuffle(ids)
    return ids[:80], ids[80:130]


def answer_qids(held_questions_qids) -> set[str]:
    """The task's exact 20 answer question IDs (sampled from held-out)."""
    qids = sorted({str(q) for q in held_questions_qids})
    random.Random(SEED).shuffle(qids)
    return set(qids[:20])


def load_all():
    """Returns (dev_split, held_split, ANSWER_QIDS)."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"validation": URL}, split="validation")
    dev_ids, held_ids = split_ids(ds["id"])
    dev_set, held_set = set(dev_ids), set(held_ids)
    splits = {"dev": ([], {}), "heldout": ([], {})}
    for rec in ds:
        pid = rec["id"]
        key = "dev" if pid in dev_set else "heldout" if pid in held_set else None
        if key is None:
            continue
        paper = build_paper(rec)
        splits[key][1][pid] = paper
        splits[key][0].extend(build_questions(rec, paper))
    dev = Split("dev", splits["dev"][1], splits["dev"][0])
    held = Split("heldout", splits["heldout"][1], splits["heldout"][0])
    return dev, held, answer_qids(q.qid for q in held.questions)