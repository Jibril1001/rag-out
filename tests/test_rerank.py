import numpy as np

from ragopt.chunking import Chunk, Span
from ragopt.pipeline import Config, retrieve
from ragopt.rerank import CrossEncoderScorer, rerank_lists


class FakeScorer(CrossEncoderScorer):
    def _predict_raw(self, pairs):
        return np.array([5.0 if "GOLD" in t else float(len(t) % 3) for _, t in pairs], dtype=np.float32)


def mk(n, gold_at):
    return [Chunk(f"c{i}", "P", "x" * (i + 1) + (" GOLD" if i == gold_at else ""), [Span("s0:p0", 0, 1)])
            for i in range(n)]


def test_rerank_moves_gold_to_top_inside_pool_only(tmp_path):
    chunks = mk(8, gold_at=4)
    ranked = [[0, 1, 2, 3, 4, 5, 6, 7]]
    out = rerank_lists(["q"], chunks, ranked, pool=5, scorer=FakeScorer(cache_dir=str(tmp_path)))
    assert out[0][0] == 4 and sorted(out[0][:5]) == [0, 1, 2, 3, 4]
    assert out[0][5:] == [5, 6, 7]                       # tail untouched
    # gold outside the pool cannot be rescued
    chunks2 = mk(8, gold_at=6)
    out2 = rerank_lists(["q"], chunks2, ranked, pool=5, scorer=FakeScorer(cache_dir=str(tmp_path / "b")))
    assert out2[0][5:] == [5, 6, 7]


def test_cache_reuse_and_persistence(tmp_path):
    chunks = mk(6, 2)
    s = FakeScorer(cache_dir=str(tmp_path))
    rerank_lists(["q1", "q2"], chunks, [[0, 1, 2], [3, 4, 5]], 3, s)
    assert s.stats["scored"] == 6
    s2 = FakeScorer(cache_dir=str(tmp_path))             # fresh object reads the disk cache
    rerank_lists(["q1", "q2"], chunks, [[0, 1, 2], [3, 4, 5]], 3, s2)
    assert s2.stats["scored"] == 0 and s2.stats["cache_hits"] == 6


def test_pipeline_applies_rerank(tmp_path):
    chunks = mk(8, gold_at=3)
    cfg = Config("t", retriever="bm25", topn=8, rerank_pool=8)
    for c in chunks:
        c.embed_text = c.text
    out = retrieve(cfg, chunks, ["xxxx GOLD"], None, FakeScorer(cache_dir=str(tmp_path)))
    assert out[0][0] == 3 and len(set(out[0])) == 8
