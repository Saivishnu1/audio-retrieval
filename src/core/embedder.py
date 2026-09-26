"""Swappable text-embedding backends: OpenAIEmbedder (primary) and LocalEmbedder (offline)."""

from __future__ import annotations

from typing import Protocol

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dimension: int
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI `text-embedding-3-small` (1536-dim). Requires OPENAI_API_KEY."""

    dimension = 1536

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings

        self.model_name = model
        # api_key=None (vs. omitting the kwarg) trips OpenAIEmbeddings into
        # an async-callable path that fails with "Sync client is not available".
        kwargs = {"model": model}
        if api_key is not None:
            kwargs["api_key"] = api_key
        self._client = OpenAIEmbeddings(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class LocalEmbedder:
    """Offline sentence-transformers BAAI/bge-small-en-v1.5 (384-dim)."""

    dimension = 384

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, convert_to_numpy=False, show_progress_bar=False)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            BGE_QUERY_PREFIX + text, convert_to_numpy=False, show_progress_bar=False
        )
