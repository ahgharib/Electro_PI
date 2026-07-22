from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.ports.embedder import Embedder


class GeminiEmbedder(Embedder):
    """Embedder adapter for Google's Gemini API (free tier by default).

    Requires GOOGLE_API_KEY. Model name is configurable (see .env /
    src/application/config.py) since Google periodically renames/deprecates
    embedding models -- e.g. text-embedding-004 was retired in favor of
    gemini-embedding-001.
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for GeminiEmbedder")
        self._model = model
        self._client = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    @property
    def model_name(self) -> str:
        return self._model
