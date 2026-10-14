"""Retrieval metrics: Recall@k and MRR@10 with the 80%-token coverage rule.

A gold paragraph counts as covered when the source spans of the retrieved
chunks cover >= 80% of its tokens. Only chunks from the TARGET paper count.

mode="union"  : spans of all chunks in the top-k are merged (default).
mode="single" : a single chunk alone must cover >= 80% of the paragraph.
(The brief does not say which one is meant, so both are implemented.)
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .chunking.base import Chunk
from .data import Paper, Question

TOKEN_RE = re.compile(r"\w+")
THRESHOLD = 0.8


class GoldIndex:
    """Token offsets of every paragraph, computed once."""

    def __init__(self, papers: dict[str, Paper]):
        self._tok: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
        for paper in papers.values():
            for p in paper.paragraphs:
                ms = list(TOKEN_RE.finditer(p.text))
                self._tok[(paper.paper_id, p.pid)] = (
                    [m.start() for m in ms], [m.end() for m in ms])

    def tokens(self, paper_id: str, pid: str):
        return self._tok[(paper_id, pid)]


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _covered_token_count(intervals, starts, ends) -> int:
    """Tokens lying fully inside the merged intervals."""
    n = 0
    for s, e in _merge(intervals):
        lo = bisect_left(starts, s)            # first token starting at/after s
        hi = bisect_right(ends, e)             # tokens ending at/before e
        n += max(0, hi - lo)
    return n


def covered_gold(chunks: list[Chunk], q: Question, gi: GoldIndex,
                 mode: str = "union", threshold: float = THRESHOLD) -> set[str]:
    chunks = [c for c in chunks if c.paper_id == q.paper_id]
    covered = set()
    for pid in q.gold_pids:
        starts, ends = gi.tokens(q.paper_id, pid)
        if not starts:
            continue
        if mode == "union":
            iv = [(s.start, s.end) for c in chunks for s in c.spans if s.paragraph_id == pid]
            frac = _covered_token_count(iv, starts, ends) / len(starts)
            ok = frac >= threshold
        elif mode == "single":
            ok = any(
                _covered_token_count([(s.start, s.end) for s in c.spans if s.paragraph_id == pid],
                                     starts, ends) / len(starts) >= threshold
                for c in chunks)
        else:
            raise ValueError(mode)
        if ok:
            covered.add(pid)
    return covered


def recall_at_k(ranked: list[Chunk], q: Question, gi: GoldIndex, k: int,
                mode: str = "union") -> float:
    return len(covered_gold(ranked[:k], q, gi, mode)) / len(q.gold_pids)


def reciprocal_rank(ranked: list[Chunk], q: Question, gi: GoldIndex, k: int = 10,
                    mode: str = "union") -> float:
    for r in range(1, k + 1):
        if covered_gold(ranked[:r], q, gi, mode):
            return 1.0 / r
    return 0.0


@dataclass
class QResult:
    qid: str
    r5: float
    r10: float
    rr: float


def evaluate_question(ranked, q, gi, mode="union") -> QResult:
    return QResult(q.qid, recall_at_k(ranked, q, gi, 5, mode),
                   recall_at_k(ranked, q, gi, 10, mode),
                   reciprocal_rank(ranked, q, gi, 10, mode))


def evaluate(ranked_by_qid: dict[str, list[Chunk]], questions: list[Question],
             gi: GoldIndex, mode: str = "union"):
    """Average over questions that have >= 1 gold paragraph."""
    res = [evaluate_question(ranked_by_qid[q.qid], q, gi, mode)
           for q in questions if q.gold_pids]
    n = len(res)
    return {"n": n,
            "R@5": sum(r.r5 for r in res) / n,
            "R@10": sum(r.r10 for r in res) / n,
            "MRR@10": sum(r.rr for r in res) / n}, res
