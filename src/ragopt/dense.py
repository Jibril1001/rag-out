"""Exact dense search (inner product on L2-normalised vectors = cosine)."""
from __future__ import annotations

import numpy as np


def top_k(scores: np.ndarray, k: int):
    """Row-wise top-k of a (n_queries, n_docs) score matrix -> (indices, scores), sorted desc."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)


class FlatIPIndex:
    """Exact search: compares each query against every stored vector."""

    def __init__(self, vecs: np.ndarray):
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)

    def search(self, q: np.ndarray, k: int):
        return top_k(q.astype(np.float32) @ self.vecs.T, k)
