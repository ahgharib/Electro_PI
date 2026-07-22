from __future__ import annotations

import hashlib
import math

from src.ports.embedder import Embedder


class FakeEmbedder(Embedder):
    """Deterministic, dependency-free embedder for tests.

    Uses a simple hashing-trick bag-of-words vector so that texts sharing
    words end up with higher cosine similarity than unrelated texts -- good
    enough to exercise gate/threshold logic and retrieval ordering in tests
    without any network call or API key.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @property
    def model_name(self) -> str:
        return "fake-hashing-embedder"
