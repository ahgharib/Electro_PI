"""Ask a question against the RAG pipeline.

Usage:
    python scripts/ask.py "What is the return window?"
    python scripts/ask.py --debug "What is the return window?"
    python scripts/ask.py            # interactive mode, Ctrl+C / "exit" to quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.pipeline import RAGPipeline  # noqa: E402


def print_result(result) -> None:
    print(f"\nAnswer: {result.answer}")
    if result.citations:
        print("\nSources (as cited by the model):")
        for c in result.citations:
            sections = f" ({', '.join(c.section_paths)})" if c.section_paths else ""
            print(f"  - {c.source_path}{sections} [{', '.join(c.chunk_ids)}]")
    elif result.retrieved_sources:
        # The model didn't confirm citations (e.g. structured output
        # failed and the fallback couldn't recover any), but retrieval DID
        # find and use real context -- surface that explicitly rather than
        # silently showing "Sources: none" for an answer that may well be
        # grounded. See CHANGELOG.md.
        print("\nSources: model did not confirm citations, but these were retrieved and used as context:")
        for c in result.retrieved_sources:
            sections = f" ({', '.join(c.section_paths)})" if c.section_paths else ""
            print(f"  - {c.source_path}{sections} [{', '.join(c.chunk_ids)}]")
    else:
        print("\nSources: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG pipeline.")
    parser.add_argument("question", nargs="*", help="The question to ask (omit for interactive mode)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full per-chunk retrieval detail (scores, previews, trimming/neighbor "
        "expansion) instead of the normal partial summary line.",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if pipeline.vector_store.count() == 0:
        print("Vector store is empty. Run `python scripts/ingest.py` first.")

    if args.question:
        question = " ".join(args.question)
        result = pipeline.answer(question, debug=args.debug)
        print_result(result)
        return

    print("Interactive mode. Type a question, or 'exit' to quit.\n")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            result = pipeline.answer(question, debug=args.debug)
            print_result(result)
            print()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        pipeline.monitor.print_stats_summary()


if __name__ == "__main__":
    main()
