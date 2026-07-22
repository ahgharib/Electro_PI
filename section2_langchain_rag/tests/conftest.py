from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

DOCS_DIR = str(Path(__file__).resolve().parent.parent / "docs")

from src.adapters.chunking.structure_aware_chunker import StructureAwareChunker
from src.adapters.embeddings.fake_embedder import FakeEmbedder
from src.adapters.generators.fake_generator import FakeGenerator
from src.adapters.loaders.registry import LoaderRegistry
from src.adapters.vectorstores.chroma_store import ChromaVectorStore
from src.application.config import Config


@pytest.fixture
def tmp_chroma_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def test_config(tmp_chroma_dir):
    config = Config()
    config.persist_directory = tmp_chroma_dir
    config.collection_name = "test_collection"
    config.relevance_threshold = 0.05  # fake embedder's cosine scores are low but consistent
    config.retrieval_k = 4
    config.verbose_console = False
    return config


@pytest.fixture
def fake_embedder():
    return FakeEmbedder(dim=64)


@pytest.fixture
def fake_generator():
    return FakeGenerator()


@pytest.fixture
def vector_store(test_config):
    return ChromaVectorStore(test_config.persist_directory, test_config.collection_name)


@pytest.fixture
def chunker():
    return StructureAwareChunker(chunk_size_tokens=200, chunk_overlap_tokens=20)


@pytest.fixture
def loader_registry():
    return LoaderRegistry()


@pytest.fixture
def indexed_vector_store(vector_store, chunker, fake_embedder, loader_registry):
    """A vector store pre-loaded with the real sample docs/ directory,
    using the fake embedder (no network / API key required)."""
    documents = loader_registry.load_directory(DOCS_DIR)
    for document in documents:
        chunks = chunker.chunk(document)
        embeddings = fake_embedder.embed_documents([c.text for c in chunks])
        vector_store.upsert(chunks, embeddings)
    return vector_store
