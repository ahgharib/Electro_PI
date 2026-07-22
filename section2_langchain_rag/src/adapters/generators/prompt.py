"""Shared prompt construction, used by every Generator adapter so the
grounding rules and output schema stay identical regardless of provider."""

from __future__ import annotations

from src.core.models import RetrievedChunk

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question using ONLY the \
information in the provided context chunks. Do not use outside knowledge, and do not guess.

Rules:
- If the context does not contain enough information to answer, say so plainly instead of \
guessing or using outside knowledge.
- Every factual claim in your answer must be supported by the provided context.
- List the chunk_id(s) that support your answer.
- Respond with ONLY a single JSON object, no other text, in exactly this shape:
{"answer": "<your answer text>", "cited_chunk_ids": ["<chunk_id>", "..."]}
"""


def build_context_block(context_chunks: list[RetrievedChunk]) -> str:
    parts = []
    for rc in context_chunks:
        c = rc.chunk
        tag = f"[chunk_id={c.chunk_id} source={c.source_path}"
        if c.section_path:
            tag += f" section=\"{c.section_path}\""
        if c.page_number is not None:
            tag += f" page={c.page_number}"
        tag += "]"
        parts.append(f"{tag}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def build_history_block(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    # Keep only the last few turns -- this is a lightweight pass-through,
    # not managed conversation memory (see NOTES.md, "Memory").
    lines = []
    for q, a in history[-3:]:
        lines.append(f"Previous Q: {q}\nPrevious A: {a}")
    return "\n\n".join(lines)


def build_user_message(
    question: str,
    context_chunks: list[RetrievedChunk],
    history: list[tuple[str, str]] | None = None,
) -> str:
    history_block = build_history_block(history)
    context_block = build_context_block(context_chunks)
    msg = ""
    if history_block:
        msg += f"Conversation so far:\n{history_block}\n\n"
    msg += f"Context:\n{context_block}\n\nQuestion: {question}"
    return msg
