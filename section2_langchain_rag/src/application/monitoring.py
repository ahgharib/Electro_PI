"""Monitoring.

Two outputs from the same QueryMetrics object, deliberately different in
detail level:

  1. A structured JSONL file (logs/rag_metrics.jsonl) -- ONE line per query,
     with every metric (LLM + RAG). This is the durable record: it doubles
     as the eval dataset for tuning the relevance threshold and as the
     input to scripts/eval.py's aggregate metrics.

  2. A verbose console printout -- human-readable, but deliberately partial.
     It shows enough to sanity-check a query while developing (gate outcome,
     top similarity score, token/cost/latency summary) without dumping the
     full per-chunk score list or the full prompt on every call. Set
     VERBOSE_CONSOLE=false in .env to silence it entirely.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from src.core.models import GateStatus, QueryMetrics, RetrievedChunk

logger = logging.getLogger("rag.monitoring")

_PREVIEW_WORDS = 15


class Monitor:
    def __init__(self, log_dir: str, verbose_console: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "rag_metrics.jsonl"
        self.verbose_console = verbose_console
        self._session_records: list[dict] = []

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def log_query(self, question: str, gate_status: GateStatus, metrics: QueryMetrics, answer_preview: str) -> None:
        record = {
            "timestamp": time.time(),
            "question": question,
            "gate_status": gate_status.value,
            "answer_preview": answer_preview[:200],
            **asdict(metrics),
        }
        self._session_records.append(record)
        self._write_jsonl(record)
        if self.verbose_console:
            self._print_verbose(record)

    def _write_jsonl(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _print_verbose(self, record: dict) -> None:
        # Deliberately partial: enough to sanity check, not a full data dump.
        # For the full per-chunk breakdown, use debug mode (see
        # print_debug_retrieval) instead of making this default view noisier.
        top_score = max(record["similarity_scores"], default=None)
        top_score_str = f"{top_score:.3f}" if top_score is not None else "n/a"
        print(
            "[rag] "
            f"gate={record['gate_status']} "
            f"chunks={record['num_chunks_retrieved']} "
            f"top_score={top_score_str} "
            f"model={record['model_used']} "
            f"tokens={record['total_tokens']} "
            f"cost=${record['estimated_cost_usd']:.5f} "
            f"latency={record['total_latency_ms']:.0f}ms"
        )

    # ------------------------------------------------------------------ #
    # Debug mode -- opt-in, full per-chunk detail. NOT part of the default
    # verbose printout on purpose (see module docstring: the standard
    # console output is deliberately partial). Turn this on per-query via
    # RAGPipeline.answer(question, debug=True), or globally with
    # DEBUG_RETRIEVAL=true in .env, when you actually need to see why a
    # particular query landed where it did -- e.g. while tuning
    # RELEVANCE_THRESHOLD or investigating a wrong-document retrieval.
    # ------------------------------------------------------------------ #

    def print_debug_retrieval(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        gate_status: GateStatus,
        threshold: float,
        metrics: QueryMetrics,
    ) -> None:
        print(f"\n[debug] Question: {question!r}")

        if not retrieved_chunks:
            print("[debug] No chunks retrieved (empty index or retrieval failure).")
        else:
            print(
                f"[debug] Retrieved {len(retrieved_chunks)} chunk(s), "
                f"relevance threshold={threshold:.3f}:"
            )
            # Sort for display by score (neighbors -- score 0.0 by
            # construction -- naturally sort last), so the strongest match
            # is always shown first regardless of the order retrieval
            # returned them in.
            ordered = sorted(retrieved_chunks, key=lambda rc: (-rc.score, rc.is_neighbor_expansion))
            for i, rc in enumerate(ordered, start=1):
                c = rc.chunk
                if rc.is_neighbor_expansion:
                    tag = "neighbor "
                else:
                    tag = "PASS     " if rc.score >= threshold else "below-thr"
                preview = " ".join(c.text.split()[:_PREVIEW_WORDS])
                location = f"page={c.page_number}" if c.page_number is not None else (c.section_path or "")
                print(
                    f"  #{i} {tag}  score={rc.score:.3f}  {c.chunk_id}  "
                    f"({c.source_path}{', ' + location if location else ''})"
                )
                print(f'       "{preview}..."')
            print(
                "[debug] Note: only the TOP score decides the gate (below-thr on a "
                "lower-ranked chunk here doesn't reject the query by itself)."
            )

        gate_explanation = {
            GateStatus.PASS: "PASS -- top score cleared the threshold, generation ran.",
            GateStatus.BELOW_THRESHOLD: "BELOW_THRESHOLD -- no chunk cleared the threshold, generation skipped.",
            GateStatus.EMPTY_INDEX: "EMPTY_INDEX -- nothing indexed, generation skipped.",
            GateStatus.RETRIEVAL_FAILED: "RETRIEVAL_FAILED -- embedder/vector store call failed.",
            GateStatus.QUESTION_TOO_LONG: "QUESTION_TOO_LONG -- rejected before retrieval even ran.",
        }.get(gate_status, gate_status.value)
        print(f"[debug] Gate decision: {gate_explanation}")

        if metrics.neighbor_chunks_added:
            print(f"[debug] Neighbor expansion: +{metrics.neighbor_chunks_added} chunk(s) pulled in for context.")
        if metrics.context_chunks_dropped_for_budget:
            print(
                f"[debug] Context trimming: {metrics.context_chunks_dropped_for_budget} "
                f"chunk(s) dropped to fit the context token budget."
            )
        if metrics.retrieval_failed:
            print(f"[debug] Retrieval failed this query (quota_exceeded={metrics.quota_exceeded}).")
        if metrics.structured_output_failed:
            print(
                "[debug] Structured output failed -- fell back to parsing raw text for the answer. "
                "Citations may be missing even though retrieval succeeded (check retrieved_sources)."
            )

    # ------------------------------------------------------------------ #
    # Aggregate stats (this-session; scripts/eval.py can aggregate the
    # full on-disk log across sessions the same way)
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        if not self._session_records:
            return {"query_count": 0}

        n = len(self._session_records)
        passed = sum(1 for r in self._session_records if r["gate_status"] == "pass")
        no_context = n - passed
        empty_index = sum(1 for r in self._session_records if r["gate_status"] == "empty_index")
        below_threshold = sum(1 for r in self._session_records if r["gate_status"] == "below_threshold")
        retrieval_failures = sum(1 for r in self._session_records if r["retrieval_failed"])
        structured_output_failures = sum(1 for r in self._session_records if r["structured_output_failed"])
        avg_latency = sum(r["total_latency_ms"] for r in self._session_records) / n
        avg_retriever_latency = sum(r["retriever_latency_ms"] for r in self._session_records) / n
        all_scores = [s for r in self._session_records for s in r["similarity_scores"]]
        avg_top_score = (
            sum(max(r["similarity_scores"], default=0.0) for r in self._session_records) / n
        )
        total_cost = sum(r["estimated_cost_usd"] for r in self._session_records)
        total_tokens = sum(r["total_tokens"] for r in self._session_records)
        total_dropped_for_budget = sum(r["context_chunks_dropped_for_budget"] for r in self._session_records)
        total_neighbors_added = sum(r["neighbor_chunks_added"] for r in self._session_records)

        return {
            "query_count": n,
            "gate_passed": f"{passed}/{n}",
            "gate_no_context": f"{no_context}/{n}",
            "no_context_rate": round(no_context / n, 3),
            "empty_index_events": empty_index,
            "below_threshold_events": below_threshold,
            "retrieval_failures": retrieval_failures,
            "structured_output_failures": structured_output_failures,
            "avg_total_latency_ms": round(avg_latency, 1),
            "avg_retriever_latency_ms": round(avg_retriever_latency, 1),
            "avg_top_similarity_score": round(avg_top_score, 3),
            "avg_similarity_score_all_chunks": round(sum(all_scores) / len(all_scores), 3) if all_scores else None,
            "total_estimated_cost_usd": round(total_cost, 5),
            "total_tokens": total_tokens,
            "total_chunks_dropped_for_budget": total_dropped_for_budget,
            "total_neighbor_chunks_added": total_neighbors_added,
        }

    def print_stats_summary(self) -> None:
        stats = self.get_stats()
        print("\n--- RAG session summary ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
