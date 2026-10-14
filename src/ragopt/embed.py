"""Embedding wrapper with a persistent per-text cache.

Cache key = sha1(prefix + text), so unchanged chunks/queries are never
re-embedded across experiments, and doc vs query formatting never collide.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class Embedder:
    def __init__(self, model_name: str = MODEL, cache_dir: str = ".cache", batch_size: int = 64):
        self.model_name, self.batch_size = model_name, batch_size
        self._model = None
        self.cache_path = Path(cache_dir) / f"emb_{model_name.replace('/', '__')}.npz"
        self._store: dict[str, np.ndarray] = {}
        self.stats = {"encoded": 0, "cache_hits": 0}
        if self.cache_path.exists():
            z = np.load(self.cache_path, allow_pickle=False)
            self._store = dict(zip(z["keys"].tolist(), z["vecs"]))

    # -- model call (overridden in tests) -------------------------------------
    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                  show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache_path, keys=np.array(list(self._store.keys())),
                 vecs=np.stack(list(self._store.values())))

    # -- public API -----------------------------------------------------------
    def encode(self, texts: list[str], prefix: str = "", use_cache: bool = True) -> np.ndarray:
        full = [prefix + t for t in texts]
        keys = [_h(t) for t in full]
        if not use_cache:                       # timing runs: always hit the model
            self.stats["encoded"] += len(full)
            return self._encode_raw(full)
        missing = {}
        for k, t in zip(keys, full):
            if k not in self._store:
                missing[k] = t
        self.stats["cache_hits"] += len(keys) - len(missing)
        if missing:
            vecs = self._encode_raw(list(missing.values()))
            self.stats["encoded"] += len(missing)
            for k, v in zip(missing.keys(), vecs):
                self._store[k] = v
            self._save()
        return np.stack([self._store[k] for k in keys])

    def encode_docs(self, texts, use_cache=True):
        return self.encode(texts, "", use_cache)

    def encode_queries(self, texts, use_cache=True):
        return self.encode(texts, QUERY_PREFIX, use_cache)
