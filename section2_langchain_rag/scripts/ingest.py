"""Ingest documents from docs/ (or a given directory) into the vector store.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --docs-dir ./my_docs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--docs-dir", default=None, help="Directory of .md/.pdf files (default: from .env)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every chunk even if it's already indexed and unchanged "
        "(normally unchanged chunks are skipped -- see README, 'Idempotent ingestion')",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    print(f"Embedding provider: {pipeline.config.embedding_provider}")
    print(f"Vector store: Chroma @ {pipeline.config.persist_directory}")
    print("Ingesting...\n")

    total_embedded = pipeline.ingest(docs_dir=args.docs_dir, force=args.force)

    print(f"\nDone. Embedded {total_embedded} chunks this run. Vector store now contains {pipeline.vector_store.count()} chunks.")


if __name__ == "__main__":
    main()
