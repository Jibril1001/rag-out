"""Structure-aware chunking: paragraph/section boundaries, sentence-level splitting.

* never crosses a section boundary (the abstract is its own section)
* paragraphs <= max_chars stay whole; longer ones are cut at sentence boundaries
* tiny units (< merge_below chars) are merged forward into the next unit of the
  same section, so we do not index one-line fragments
* `text` is the original text (spans reproduce it exactly: pieces joined by "\\n\\n");
  `embed_text` optionally prepends "title \\n section heading" (context=True)
"""
from __future__ import annotations

import re

from ..data import Paper
from .base import Chunk, Span

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SEP = "\n\n"


def split_long(text: str, max_chars: int) -> list[tuple[int, int]]:
    """(start, end) slices of `text`, each <= max_chars, cut at sentence ends where possible."""
    if len(text) <= max_chars:
        cuts = [(0, len(text))]
    else:
        bounds = [0] + [m.end() for m in _SENT_SPLIT.finditer(text)] + [len(text)]
        cuts, cur = [], 0
        for i in range(1, len(bounds)):
            end = bounds[i]
            if end - cur > max_chars and bounds[i - 1] > cur:
                cuts.append((cur, bounds[i - 1]))
                cur = bounds[i - 1]
            while end - cur > max_chars:                    # one sentence longer than max_chars
                cut = text.rfind(" ", cur + 1, cur + max_chars)
                cut = cut if cut > cur else cur + max_chars
                cuts.append((cur, cut))
                cur = cut
        cuts.append((cur, len(text)))
    out = []
    for s, e in cuts:                                       # trim whitespace, keep exact offsets
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def chunk_paper_structure(paper: Paper, max_chars: int = 1000, merge_below: int = 300,
                          context: bool = False) -> list[Chunk]:
    groups: dict[str, tuple[str, list]] = {}
    for p in paper.paragraphs:
        key = p.pid.split(":")[0]                           # "abstract" | "s3"
        groups.setdefault(key, (p.section, []))[1].append(p)

    chunks: list[Chunk] = []

    def flush(cur, section):
        if not cur:
            return
        text = SEP.join(t for _, t in cur)
        head = f"{paper.title}\n{section}\n" if context else ""
        chunks.append(Chunk(f"{paper.paper_id}:c{len(chunks)}", paper.paper_id, text,
                            [s for s, _ in cur], head + text))

    for section, paras in groups.values():
        units = [(Span(p.pid, s, e), p.text[s:e]) for p in paras for s, e in split_long(p.text, max_chars)]
        cur, cur_len = [], 0
        for span, txt in units:
            if cur and cur_len < merge_below and cur_len + len(SEP) + len(txt) <= max_chars:
                cur.append((span, txt))
                cur_len += len(SEP) + len(txt)
            else:
                flush(cur, section)
                cur, cur_len = [(span, txt)], len(txt)
        flush(cur, section)
    return chunks


def chunk_corpus_structure(papers, max_chars=1000, merge_below=300, context=False):
    return [c for p in papers.values()
            for c in chunk_paper_structure(p, max_chars, merge_below, context)]
