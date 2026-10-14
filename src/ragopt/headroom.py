"""Headroom analysis helpers (dev only; these use gold labels by design)."""
from __future__ import annotations

from .chunking.base import Chunk
from .data import Question
from .metrics import GoldIndex


def _covered_tokens(chunk: Chunk, gold_pids: set[str], q: Question, gi: GoldIndex) -> set[tuple[str, int]]:
    """(pid, token_index) pairs of gold paragraphs lying fully inside this chunk's spans."""
    out = set()
    for s in chunk.spans:
        if s.paragraph_id not in gold_pids:
            continue
        starts, ends = gi.tokens(q.paper_id, s.paragraph_id)
        for i, (ts, te) in enumerate(zip(starts, ends)):
            if s.start <= ts and te <= s.end:
                out.add((s.paragraph_id, i))
    return out


def oracle_ranking(chunks_of_paper: list[Chunk], q: Question, gi: GoldIndex, k: int = 10) -> list[Chunk]:
    """Greedy best-case top-k: repeatedly take the chunk adding most new gold tokens.

    Gives the CHUNKING CEILING: the recall a perfect ranker could reach with
    these chunks (greedy => a close lower bound of the true optimum).
    """
    cands = {c.chunk_id: (c, _covered_tokens(c, q.gold_pids, q, gi)) for c in chunks_of_paper}
    chosen, seen = [], set()
    for _ in range(k):
        best_id, best_gain = None, 0
        for cid, (c, toks) in cands.items():
            gain = len(toks - seen)
            if gain > best_gain:
                best_id, best_gain = cid, gain
        if best_id is None:
            break
        c, toks = cands.pop(best_id)
        chosen.append(c)
        seen |= toks
    # pad with remaining chunks so the list has k entries (they add nothing)
    for c, _ in list(cands.values())[: max(0, k - len(chosen))]:
        chosen.append(c)
    return chosen
