"""Normalizes LangChain's AIMessage.content into a plain string.

This is the fix for the bug where newer Gemini models (and any provider
using the "content blocks" format) return `.content` as something other
than a plain string:

  - a plain string                                    (older / simple case)
  - a single content-block dict:  {"type": "text", "text": "...", "extras": {...}}
  - a list of content blocks:     [{"type": "text", "text": "..."}, ...]
    where blocks can also carry non-text metadata (e.g. Gemini's "extras"/
    "signature" field, used for reasoning continuity across turns -- not
    meant to ever reach the user).

Every Generator adapter MUST pass its raw provider response through this
function before constructing a GenerationResult, so `GenerationResult.text`
is *always* guaranteed to be a plain string. That keeps the rest of the
system (JSON parsing, citation extraction) free of provider-specific
response-shape handling -- normalizing provider quirks is exactly the
adapter's job in this architecture, not the application layer's.
"""

from __future__ import annotations


def extract_response_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # A single content block. Only the "text" part is ever surfaced --
        # any other key (e.g. "extras"/"signature") is internal provider
        # metadata and must never leak into the answer.
        return content.get("text", "") or ""

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in (None, "text"):
                    parts.append(block.get("text", "") or "")
                # other block types (e.g. tool_use, thinking/reasoning
                # blocks) are intentionally skipped -- not part of the
                # user-facing answer.
        return "".join(parts)

    if content is None:
        return ""

    # Unknown shape -- fall back to a string conversion rather than
    # raising, but this should not happen for a well-behaved adapter.
    return str(content)
