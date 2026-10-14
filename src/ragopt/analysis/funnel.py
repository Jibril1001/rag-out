"""Failure funnel: assign each dev question ONE primary outcome (priority order)."""
from __future__ import annotations

from ..chunking.base import Chunk
from ..data import Question
from ..headroom import _covered_tokens
from ..metrics import GoldIndex, covered_gold

CATEGORIES = ["success", "wrong_paper", "incomplete_multi_evidence",
              "boundary_split", "ranked_too_low", "not_found"]

DESCRIPTIONS = {
    "success": "all gold paragraphs covered in the top-10",
    "wrong_paper": "no top-10 chunk comes from the target paper",
    "incomplete_multi_evidence": "some, but not all, gold paragraphs covered",
    "boundary_split": "right place retrieved but chunk edges cut the paragraph (<80% covered)",
    "ranked_too_low": "gold covered within the top-50, but not the top-10",
    "not_found": "target paper present in top-10 but gold not even in the top-50",
}


def classify(q: Question, top10: list[Chunk], top50: list[Chunk], gi: GoldIndex,
             mode: str = "union") -> str:
    cov10 = covered_gold(top10, q, gi, mode)
    if len(cov10) == len(q.gold_pids):
        return "success"
    mine = [c for c in top10 if c.paper_id == q.paper_id]
    if not mine:
        return "wrong_paper"
    if cov10:
        return "incomplete_multi_evidence"
    if any(_covered_tokens(c, q.gold_pids, q, gi) for c in mine):
        return "boundary_split"
    if covered_gold(top50, q, gi, mode):
        return "ranked_too_low"
    return "not_found"


def chunks_overlap(a: Chunk, b: Chunk) -> bool:
    """True if two chunks share any characters of the same paragraph."""
    if a.paper_id != b.paper_id:
        return False
    for s in a.spans:
        for t in b.spans:
            if s.paragraph_id == t.paragraph_id and s.start < t.end and t.start < s.end:
                return True
    return False
