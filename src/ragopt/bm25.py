"""BM25 implemented from scratch (inverted index + Okapi scoring)."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have he her his how
i if in into is it its may might more most no not of on or our she should so such than that the their them then there
these they this those to was we were what when where which while who whom why will with would you your also use used
using based paper propose proposed""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs_tokens: list[list[str]]) -> "BM25":
        self.n = len(docs_tokens)
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        post = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                post[term].append((i, tf))
        self.idf, self.post = {}, {}
        for term, plist in post.items():
            df = len(plist)
            self.idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            ids = np.array([i for i, _ in plist], dtype=np.int64)
            tfs = np.array([t for _, t in plist], dtype=np.float32)
            self.post[term] = (ids, tfs)
        return self

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(query_tokens):
            if term not in self.post:
                continue
            ids, tfs = self.post[term]
            norm = tfs + self.k1 * (1 - self.b + self.b * self.dl[ids] / self.avgdl)
            out[ids] += self.idf[term] * tfs * (self.k1 + 1) / norm
        return out

    def score_matrix(self, queries_tokens: list[list[str]]) -> np.ndarray:
        return np.stack([self.scores(q) for q in queries_tokens])
