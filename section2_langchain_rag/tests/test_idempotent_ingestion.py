from __future__ import annotations

from pathlib import Path

from src.application.pipeline import RAGPipeline

DOCS_DIR = str(Path(__file__).resolve().parent.parent / "docs")


def _build_pipeline(test_config, embedder, vector_store, generator, chunker, loader_registry):
    return RAGPipeline(
        config=test_config,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        chunker=chunker,
        loader_registry=loader_registry,
    )


class _CountingEmbedder:
    """Wraps a real embedder and counts embed_documents() calls / texts
    embedded, so tests can assert how much "quota" a run would have used."""

    def __init__(self, inner):
        self._inner = inner
        self.call_count = 0
        self.texts_embedded = 0

    def embed_documents(self, texts):
        self.call_count += 1
        self.texts_embedded += len(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)

    @property
    def model_name(self):
        return self._inner.model_name


class TestIdempotentIngestion:
    def test_second_ingest_run_skips_unchanged_chunks(
        self, test_config, vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        counting_embedder = _CountingEmbedder(fake_embedder)
        pipeline = _build_pipeline(
            test_config, counting_embedder, vector_store, fake_generator, chunker, loader_registry
        )

        first_run_embedded = pipeline.ingest(docs_dir=DOCS_DIR)
        assert first_run_embedded > 0
        texts_after_first_run = counting_embedder.texts_embedded

        second_run_embedded = pipeline.ingest(docs_dir=DOCS_DIR)

        assert second_run_embedded == 0  # nothing changed -- nothing re-embedded
        assert counting_embedder.texts_embedded == texts_after_first_run  # no new embed calls

    def test_force_reembeds_everything_regardless_of_unchanged_content(
        self, test_config, vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        counting_embedder = _CountingEmbedder(fake_embedder)
        pipeline = _build_pipeline(
            test_config, counting_embedder, vector_store, fake_generator, chunker, loader_registry
        )

        first_run_embedded = pipeline.ingest(docs_dir=DOCS_DIR)
        second_run_embedded = pipeline.ingest(docs_dir=DOCS_DIR, force=True)

        assert second_run_embedded == first_run_embedded  # force re-embeds all of it again

    def test_skip_behavior_can_be_disabled_via_config(
        self, test_config, vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        test_config.skip_unchanged_chunks_on_ingest = False
        counting_embedder = _CountingEmbedder(fake_embedder)
        pipeline = _build_pipeline(
            test_config, counting_embedder, vector_store, fake_generator, chunker, loader_registry
        )

        first_run_embedded = pipeline.ingest(docs_dir=DOCS_DIR)
        second_run_embedded = pipeline.ingest(docs_dir=DOCS_DIR)

        assert second_run_embedded == first_run_embedded  # skip logic disabled -- re-embeds every time

    def test_vector_store_chunk_count_unaffected_by_repeat_ingestion(
        self, test_config, vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        pipeline = _build_pipeline(
            test_config, fake_embedder, vector_store, fake_generator, chunker, loader_registry
        )
        pipeline.ingest(docs_dir=DOCS_DIR)
        count_after_first = pipeline.vector_store.count()

        pipeline.ingest(docs_dir=DOCS_DIR)
        count_after_second = pipeline.vector_store.count()

        assert count_after_first == count_after_second  # no duplicate entries
