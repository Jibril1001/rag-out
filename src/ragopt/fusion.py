"""Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Fuse several ranked lists of doc ids. score(d) = sum_i w_i / (k + rank_i(d)), rank from 1."""
    weights = weights or [1.0] * len(rankings)
    score = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for r, d in enumerate(ranking, start=1):
            score[d] += w / (k + r)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))
