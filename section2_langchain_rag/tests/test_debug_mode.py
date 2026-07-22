from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout

from src.application.monitoring import Monitor
from src.application.pipeline import RAGPipeline
from src.core.models import Chunk, GateStatus, QueryMetrics, RetrievedChunk


def _rc(chunk_id, source, text, score, page=None, neighbor=False):
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id, doc_id="d", source_path=source, text=text,
            chunk_index=0, page_number=page,
        ),
        score=score,
        is_neighbor_expansion=neighbor,
    )


class TestDebugRetrievalPrinter:
    def test_prints_chunk_preview_score_and_pass_fail(self):
        monitor = Monitor(tempfile.mkdtemp(), verbose_console=False)
        chunks = [
            _rc("a::0", "docs/a.md", "the quick brown fox jumps over the lazy dog", 0.74),
            _rc("b::0", "docs/b.md", "some other unrelated sentence here", 0.31),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            monitor.print_debug_retrieval("test question", chunks, GateStatus.PASS, 0.55, QueryMetrics())
        output = buf.getvalue()

        assert "test question" in output
        assert "0.740" in output and "0.310" in output
        assert "quick brown fox" in output
        assert "PASS" in output
        assert "below-thr" in output

    def test_handles_empty_chunk_list(self):
        monitor = Monitor(tempfile.mkdtemp(), verbose_console=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            monitor.print_debug_retrieval("q", [], GateStatus.EMPTY_INDEX, 0.55, QueryMetrics())
        assert "empty index" in buf.getvalue().lower() or "No chunks retrieved" in buf.getvalue()

    def test_marks_neighbor_chunks_distinctly(self):
        monitor = Monitor(tempfile.mkdtemp(), verbose_console=False)
        chunks = [
            _rc("a::0", "docs/a.md", "real hit", 0.7),
            _rc("a::1", "docs/a.md", "pulled in for context", 0.0, neighbor=True),
        ]
        metrics = QueryMetrics(neighbor_chunks_added=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            monitor.print_debug_retrieval("q", chunks, GateStatus.PASS, 0.55, metrics)
        output = buf.getvalue()
        assert "neighbor" in output
        assert "Neighbor expansion: +1" in output

    def test_reports_context_trimming_when_present(self):
        monitor = Monitor(tempfile.mkdtemp(), verbose_console=False)
        metrics = QueryMetrics(context_chunks_dropped_for_budget=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            monitor.print_debug_retrieval("q", [_rc("a::0", "d.md", "x", 0.9)], GateStatus.PASS, 0.55, metrics)
        assert "2" in buf.getvalue() and "dropped" in buf.getvalue()


class TestStatsPassFailCounts:
    def test_get_stats_reports_explicit_pass_and_no_context_counts(self):
        monitor = Monitor(tempfile.mkdtemp(), verbose_console=False)
        monitor.log_query("q1", GateStatus.PASS, QueryMetrics(), "answer")
        monitor.log_query("q2", GateStatus.PASS, QueryMetrics(), "answer")
        monitor.log_query("q3", GateStatus.BELOW_THRESHOLD, QueryMetrics(), "no info")

        stats = monitor.get_stats()

        assert stats["gate_passed"] == "2/3"
        assert stats["gate_no_context"] == "1/3"
        assert stats["query_count"] == 3


class TestDebugModeThroughPipeline:
    def test_debug_true_prints_per_chunk_detail(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        pipeline = RAGPipeline(
            config=test_config, embedder=fake_embedder, vector_store=indexed_vector_store,
            generator=fake_generator, chunker=chunker, loader_registry=loader_registry,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            pipeline.answer("What is the return window?", debug=True)
        assert "[debug]" in buf.getvalue()

    def test_debug_false_does_not_print_per_chunk_detail(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        test_config.verbose_console = False
        pipeline = RAGPipeline(
            config=test_config, embedder=fake_embedder, vector_store=indexed_vector_store,
            generator=fake_generator, chunker=chunker, loader_registry=loader_registry,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            pipeline.answer("What is the return window?", debug=False)
        assert "[debug]" not in buf.getvalue()

    def test_config_debug_retrieval_flag_used_when_debug_not_passed_explicitly(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        test_config.debug_retrieval = True
        pipeline = RAGPipeline(
            config=test_config, embedder=fake_embedder, vector_store=indexed_vector_store,
            generator=fake_generator, chunker=chunker, loader_registry=loader_registry,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            pipeline.answer("What is the return window?")  # debug omitted -> falls back to config
        assert "[debug]" in buf.getvalue()
