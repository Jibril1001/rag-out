"""Cross-encoder reranking with a persistent (query, passage) -> score cache."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _key(query: str, text: str) -> str:
    return hashlib.sha1((query + "\x1f" + text).encode("utf-8")).hexdigest()


class CrossEncoderScorer:
    def __init__(self, model_name: str = RERANKER, cache_dir: str = ".cache", batch_size: int = 32):
        self.model_name, self.batch_size, self._model = model_name, batch_size, None
        self.cache_path = Path(cache_dir) / f"ce_{model_name.replace('/', '__')}.npz"
        self._store: dict[str, float] = {}
        self.stats = {"scored": 0, "cache_hits": 0}
        if self.cache_path.exists():
            z = np.load(self.cache_path, allow_pickle=False)
            self._store = dict(zip(z["keys"].tolist(), z["vals"].tolist()))

    def _predict_raw(self, pairs: list[tuple[str, str]]) -> np.ndarray:   # overridden in tests
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
        return np.asarray(self._model.predict([list(p) for p in pairs], batch_size=self.batch_size,
                                              show_progress_bar=True), dtype=np.float32).reshape(-1)

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache_path, keys=np.array(list(self._store.keys())),
                 vals=np.array(list(self._store.values()), dtype=np.float32))

    def score(self, pairs: list[tuple[str, str]], use_cache: bool = True) -> np.ndarray:
        if not use_cache:
            self.stats["scored"] += len(pairs)
            return self._predict_raw(pairs)
        keys = [_key(q, t) for q, t in pairs]
        missing = {k: p for k, p in zip(keys, pairs) if k not in self._store}
        self.stats["cache_hits"] += len(keys) - len(missing)
        if missing:
            vals = self._predict_raw(list(missing.values()))
            self.stats["scored"] += len(missing)
            for k, v in zip(missing.keys(), vals):
                self._store[k] = float(v)
            self._save()
        return np.array([self._store[k] for k in keys], dtype=np.float32)


def rerank_lists(queries: list[str], chunks, ranked: list[list[int]], pool: int,
                 scorer: CrossEncoderScorer) -> list[list[int]]:
    """Re-order the first `pool` candidates of each ranking by cross-encoder score;
    everything after the pool keeps its original order."""
    pairs = [(q, chunks[j].embed_text) for q, ids in zip(queries, ranked) for j in ids[:pool]]
    scores = scorer.score(pairs)
    out, pos = [], 0
    for ids in ranked:
        head = ids[:pool]
        s = scores[pos:pos + len(head)]
        pos += len(head)
        order = sorted(range(len(head)), key=lambda t: (-s[t], t))     # ties keep original order
        out.append([head[t] for t in order] + list(ids[pool:]))
    return out
