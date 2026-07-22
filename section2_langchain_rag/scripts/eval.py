"""Runs a small, fixed evaluation question set end-to-end and prints
results plus aggregate RAG/LLM metrics. This IS the "example questions and
actual answers" deliverable required by the task -- run this after
`ingest.py` and copy the output into NOTES.md.

Deliberately kept to 5 questions, not a large sweep: this project runs on
Gemini's free tier, and a bigger question set risks burning the daily
quota before finishing a run (see README, "Managing free-tier quota").
5 questions is enough to cover one of each required case:
  - EASY   (Q1): a single direct fact from one document.
  - MEDIUM (Q2): a specific fact + its named reporting process, one document.
  - HARD   (Q3): a grounding test -- does the answer stay faithful to what
    THIS document says, rather than substituting the model's own current
    knowledge? (the document's data is from 2013 and is now outdated in
    the real world; a correct answer says so rather than "correcting" it)
  - EDGE, below threshold (Q4): genuinely unrelated to all 3 documents --
    should never reach the LLM at all (gate_status=below_threshold).
  - EDGE, passes but should self-refuse (Q5): a compound question that
    splices real vocabulary from two different documents together
    ("moons" + "the cloud"). This one DOES clear the relevance threshold
    (verified empirically -- see CHANGELOG.md), so it's the model's own
    grounding instruction, not the retrieval gate, that has to catch it.
    Kept in deliberately to demonstrate that second layer of defense.

Usage:
    python scripts/eval.py
    python scripts/eval.py --out logs/eval_results.md
    python scripts/eval.py --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.pipeline import RAGPipeline  # noqa: E402

EVAL_QUESTIONS = [
    # EASY -- single direct fact (NIST SP 800-145)
    "How many essential characteristics, service models, and deployment "
    "models does the NIST definition of cloud computing have?",

    # MEDIUM -- a specific fact plus its reporting process (CDC Flu VIS)
    "What is VAERS, and how do you report a reaction to it?",

    # HARD -- grounding test (NASA "Our Solar System" lithograph)
    "Is the moon and distance data in this document current, and how "
    "should an answer handle that?",

    # EDGE CASE -- genuinely unrelated to all 3 docs; should be rejected
    # by the relevance gate before the LLM is ever called
    "What is the maximum towing capacity of a diesel pickup truck?",

    # EDGE CASE -- compound question blending real vocabulary from two
    # unrelated documents ("moons" from NASA doc + "the cloud" from NIST
    # doc); passes the relevance gate, so the model's own grounding
    # instruction is what has to catch it, not retrieval
    "How many moons does the cloud have, according to these documents?",
]


def render_markdown(results, stats) -> str:
    lines = ["# Evaluation Results\n"]
    for i, r in enumerate(results, start=1):
        lines.append(f"## Q{i}: {r.question}\n")
        lines.append(f"**Answer:** {r.answer}\n")
        if r.citations:
            lines.append("**Sources (as cited by the model):**")
            for c in r.citations:
                sections = f" ({', '.join(c.section_paths)})" if c.section_paths else ""
                lines.append(f"- `{c.source_path}`{sections} — chunks: {', '.join(c.chunk_ids)}")
        elif r.retrieved_sources:
            lines.append("**Sources:** model did not confirm citations, but these were retrieved and used as context:")
            for c in r.retrieved_sources:
                sections = f" ({', '.join(c.section_paths)})" if c.section_paths else ""
                lines.append(f"- `{c.source_path}`{sections} — chunks: {', '.join(c.chunk_ids)}")
        else:
            lines.append("**Sources:** none")
        structured_flag = " | structured_output_failed=True" if r.metrics.structured_output_failed else ""
        lines.append(
            f"\n_gate={r.gate_status.value} | chunks_retrieved={r.metrics.num_chunks_retrieved} | "
            f"top_score={max(r.metrics.similarity_scores, default=0):.3f} | "
            f"tokens={r.metrics.total_tokens} | cost=${r.metrics.estimated_cost_usd:.5f} | "
            f"latency={r.metrics.total_latency_ms:.0f}ms{structured_flag}_\n"
        )
    lines.append("## Session summary\n")
    for k, v in stats.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed evaluation question set.")
    parser.add_argument("--out", default=None, help="Optional path to write a markdown report to")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full per-chunk retrieval detail for every question (scores, previews, "
        "trimming/neighbor expansion) instead of the normal partial summary line.",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    if pipeline.vector_store.count() == 0:
        print("Vector store is empty. Run `python scripts/ingest.py` first.\n")
        return

    results = []
    for question in EVAL_QUESTIONS:
        print(f"\nQ: {question}")
        result = pipeline.answer(question, debug=args.debug)
        print(f"A: {result.answer}")
        results.append(result)

    stats = pipeline.stats()
    pipeline.monitor.print_stats_summary()
    print(f"\n{stats['gate_passed']} questions passed the relevance gate, "
          f"{stats['gate_no_context']} returned a no-context response.")

    if args.out:
        report = render_markdown(results, stats)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
