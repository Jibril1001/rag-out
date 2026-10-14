"""Config-driven retrieval pipeline: every experiment is a Config, not a code fork."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .bm25 import BM25, tokenize
from .chunking.fixed import chunk_corpus_fixed
from .chunking.structure import chunk_corpus_structure
from .dense import top_k
from .fusion import rrf
from .rerank import RERANKER, rerank_lists


@dataclass
class Config:
    name: str
    chunker: str = "fixed"        # fixed | structure
    size: int = 1000              # fixed: chunk chars
    overlap: int = 200            # fixed: overlap chars
    max_chars: int = 1000         # structure: max chars per chunk
    merge_below: int = 300        # structure: merge units shorter than this
    context: bool = False         # structure: prepend title + section heading to embed text
    retriever: str = "dense"      # dense | bm25 | hybrid
    pool: int = 50                # hybrid: candidates per retriever
    rrf_k: int = 60
    topn: int = 50                # candidates returned (first 10 are the answer)
    rerank_pool: int = 0          # >0: cross-encoder reorders the first N candidates
    reranker: str = RERANKER

    def to_dict(self):
        return asdict(self)


def build_chunks(cfg: Config, papers):
    if cfg.chunker == "fixed":
        return chunk_corpus_fixed(papers, cfg.size, cfg.overlap)
    if cfg.chunker == "structure":
        return chunk_corpus_structure(papers, cfg.max_chars, cfg.merge_below, cfg.context)
    raise ValueError(cfg.chunker)


def retrieve(cfg: Config, chunks, queries: list[str], emb=None, scorer=None) -> list[list[int]]:
    """Ranked chunk indices (length cfg.topn) for each query. Searches the WHOLE corpus."""
    texts = [c.embed_text for c in chunks]
    pool = max(cfg.pool, cfg.topn) if cfg.retriever == "hybrid" else cfg.topn
    dense_idx = bm_idx = None
    if cfg.retriever in ("dense", "hybrid"):
        D, Q = emb.encode_docs(texts), emb.encode_queries(queries)
        dense_idx, _ = top_k(Q @ D.T, pool)
    if cfg.retriever in ("bm25", "hybrid"):
        bm = BM25().fit([tokenize(t) for t in texts])
        bm_idx, _ = top_k(bm.score_matrix([tokenize(q) for q in queries]), pool)
    if cfg.retriever == "dense":
        ranked = [list(map(int, r[:cfg.topn])) for r in dense_idx]
    elif cfg.retriever == "bm25":
        ranked = [list(map(int, r[:cfg.topn])) for r in bm_idx]
    elif cfg.retriever == "hybrid":
        ranked = [[d for d, _ in rrf([list(map(int, a)), list(map(int, b))], cfg.rrf_k)][:cfg.topn]
                  for a, b in zip(dense_idx, bm_idx)]
    else:
        raise ValueError(cfg.retriever)
    if cfg.rerank_pool > 0:
        assert cfg.rerank_pool <= cfg.topn and scorer is not None
        ranked = rerank_lists(queries, chunks, ranked, cfg.rerank_pool, scorer)
    return ranked
