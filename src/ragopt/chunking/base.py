from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """A character range inside ONE original paragraph (end exclusive)."""
    paragraph_id: str   # "abstract" | "s0:p4"
    start: int
    end: int


@dataclass
class Chunk:
    chunk_id: str                 # "<paper_id>:c<n>"
    paper_id: str
    text: str                     # text RETURNED to the user / generator
    spans: list[Span]             # where `text` came from in the paper
    embed_text: str | None = None # text used for embedding/BM25 (defaults to text)

    def __post_init__(self):
        if self.embed_text is None:
            self.embed_text = self.text

    def to_record(self, rank: int, score: float | None = None) -> dict:
        rec = {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "text": self.text,
            "source_spans": [
                {"paragraph_id": s.paragraph_id, "start": s.start, "end": s.end}
                for s in self.spans
            ],
        }
        if score is not None:
            rec["score"] = float(score)
        return rec
