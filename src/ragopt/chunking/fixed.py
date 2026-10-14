"""Baseline: overlapping fixed-length CHARACTER chunks over the whole paper."""
from __future__ import annotations

from ..data import Paper
from .base import Chunk, Span

SEP = "\n\n"   # joins paragraphs in the working document; never part of a span


def chunk_paper_fixed(paper: Paper, size: int = 1000, overlap: int = 200) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    parts, segs, pos = [], [], 0          # segs: (doc_start, doc_end, pid)
    for p in paper.paragraphs:
        if parts:
            pos += len(SEP)
        segs.append((pos, pos + len(p.text), p.pid))
        parts.append(p.text)
        pos += len(p.text)
    doc = SEP.join(parts)
    n, stride = len(doc), size - overlap

    chunks, a = [], 0
    while a < n:
        b = min(a + size, n)
        spans = []
        for s0, s1, pid in segs:
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                spans.append(Span(pid, lo - s0, hi - s0))
        chunks.append(Chunk(f"{paper.paper_id}:c{len(chunks)}", paper.paper_id,
                            doc[a:b], spans))
        if b == n:
            break
        a += stride
    return chunks


def chunk_corpus_fixed(papers: dict[str, Paper], size: int = 1000, overlap: int = 200):
    return [c for paper in papers.values() for c in chunk_paper_fixed(paper, size, overlap)]
