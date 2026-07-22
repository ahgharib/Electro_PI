"""Deterministic content hashing, used to make ingestion idempotent.

A chunk's ID is deterministic (doc_id::chunk::index), but its *content*
can change between runs (edited source doc, different chunking config).
Storing a content hash alongside each chunk lets ingest() tell "this exact
chunk is already indexed, skip re-embedding it" apart from "this ID exists
but the content changed, needs re-embedding" -- without ever needing to
re-embed unchanged content just because ingest.py was run again.
"""

from __future__ import annotations

import hashlib


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
