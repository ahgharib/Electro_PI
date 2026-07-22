"""Component factory.

This is the ONLY place in the codebase that imports concrete adapter
classes (GeminiEmbedder, OpenAIGenerator, ChromaVectorStore, ...) and knows
about the Config object. Everything else -- the graph, the pipeline, the
scripts -- depends only on the ports. Swapping the default provider, adding
a new one, or changing the vector store means editing this file (and adding
one new adapter class), nothing more.
"""

from __future__ import annotations

from src.adapters.chunking.structure_aware_chunker import StructureAwareChunker
from src.adapters.embeddings.gemini_embedder import GeminiEmbedder
from src.adapters.embeddings.openai_embedder import OpenAIEmbedder
from src.adapters.generators.gemini_generator import GeminiGenerator
from src.adapters.generators.openai_generator import OpenAIGenerator
from src.adapters.loaders.registry import LoaderRegistry
from src.adapters.vectorstores.chroma_store import ChromaVectorStore
from src.application.config import Config
from src.application.resilient_adapters import ResilientEmbedder, ResilientGenerator
from src.ports.chunker import Chunker
from src.ports.embedder import Embedder
from src.ports.generator import Generator
from src.ports.vector_store import VectorStore


def _raw_embedder(provider: str, config: Config) -> Embedder:
    if provider == "gemini":
        return GeminiEmbedder(api_key=config.google_api_key, model=config.gemini_embedding_model)
    if provider == "openai":
        return OpenAIEmbedder(api_key=config.openai_api_key, model=config.openai_embedding_model)
    raise ValueError(f"Unknown embedding provider: {provider}")


def _raw_generator(provider: str, config: Config) -> Generator:
    if provider == "gemini":
        return GeminiGenerator(api_key=config.google_api_key, model=config.gemini_chat_model)
    if provider == "openai":
        return OpenAIGenerator(api_key=config.openai_api_key, model=config.openai_chat_model)
    raise ValueError(f"Unknown generation provider: {provider}")


def build_embedder(config: Config) -> Embedder:
    primary = _raw_embedder(config.embedding_provider, config)
    fallback = (
        _raw_embedder(config.fallback_embedding_provider, config)
        if config.fallback_embedding_provider
        else None
    )
    return ResilientEmbedder(
        primary=primary,
        primary_provider_name=config.embedding_provider,
        fallback=fallback,
        fallback_provider_name=config.fallback_embedding_provider or None,
        config=config,
    )


def build_generator(config: Config) -> Generator:
    primary = _raw_generator(config.generation_provider, config)
    fallback = (
        _raw_generator(config.fallback_generation_provider, config)
        if config.fallback_generation_provider
        else None
    )
    return ResilientGenerator(
        primary=primary,
        primary_provider_name=config.generation_provider,
        fallback=fallback,
        fallback_provider_name=config.fallback_generation_provider or None,
        config=config,
    )


def build_vector_store(config: Config) -> VectorStore:
    return ChromaVectorStore(
        persist_directory=config.persist_directory, collection_name=config.collection_name
    )


def build_chunker(config: Config) -> Chunker:
    return StructureAwareChunker(
        chunk_size_tokens=config.chunk_size_tokens, chunk_overlap_tokens=config.chunk_overlap_tokens
    )


def build_loader_registry() -> LoaderRegistry:
    return LoaderRegistry()
