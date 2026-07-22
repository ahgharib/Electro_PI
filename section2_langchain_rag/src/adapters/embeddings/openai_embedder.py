from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from src.ports.embedder import Embedder


class OpenAIEmbedder(Embedder):
    """Embedder adapter for OpenAI. Used as the configured fallback provider,
    and as proof that swapping embedding providers is a config change, not
    a code change -- see FALLBACK_EMBEDDING_PROVIDER in .env."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")
        self._model = model
        self._client = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    @property
    def model_name(self) -> str:
        return self._model
