import hashlib

import numpy as np

from ragopt.dense import FlatIPIndex, top_k
from ragopt.embed import Embedder, QUERY_PREFIX


class FakeEmbedder(Embedder):
    """Deterministic fake: vector derived from the text hash, no model needed."""
    def _encode_raw(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=16).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def test_cache_hits_and_persistence(tmp_path):
    e = FakeEmbedder(cache_dir=str(tmp_path))
    a = e.encode_docs(["x", "y", "x"])
    assert e.stats == {"encoded": 2, "cache_hits": 1} or e.stats["encoded"] == 2
    e2 = FakeEmbedder(cache_dir=str(tmp_path))            # new object: loads from disk
    b = e2.encode_docs(["x", "y"])
    assert e2.stats["encoded"] == 0 and e2.stats["cache_hits"] == 2
    assert np.allclose(a[:2], b)


def test_query_and_doc_do_not_collide(tmp_path):
    e = FakeEmbedder(cache_dir=str(tmp_path))
    d, q = e.encode_docs(["same text"]), e.encode_queries(["same text"])
    assert not np.allclose(d, q)                          # prefix changes the vector
    assert QUERY_PREFIX.startswith("Represent")


def test_no_cache_always_encodes(tmp_path):
    e = FakeEmbedder(cache_dir=str(tmp_path))
    e.encode_queries(["q"], use_cache=False)
    e.encode_queries(["q"], use_cache=False)
    assert e.stats["encoded"] == 2


def test_top_k_matches_bruteforce():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(5, 50))
    idx, sc = top_k(S, 10)
    for i in range(5):
        expect = np.argsort(-S[i])[:10]
        assert list(idx[i]) == list(expect)
        assert np.allclose(sc[i], S[i][expect])


def test_flat_index_finds_itself_and_handles_small_corpus():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(20, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    idx, _ = FlatIPIndex(V).search(V[:3], 5)
    assert list(idx[:, 0]) == [0, 1, 2]
    idx, _ = FlatIPIndex(V[:4]).search(V[:1], 10)          # fewer docs than k
    assert idx.shape == (1, 4)
