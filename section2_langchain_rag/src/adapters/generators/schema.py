"""Shared answer schema for provider-native structured output.

Used with each provider's with_structured_output(...) so the model's
output is constrained server-side to this shape, rather than only
requested via prompt instructions and hoped for. See
src/adapters/generators/gemini_generator.py and openai_generator.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StructuredAnswer(BaseModel):
    """The only shape a generation call is allowed to return."""

    answer: str = Field(description="The answer to the user's question, grounded only in the provided context.")
    cited_chunk_ids: list[str] = Field(
        default_factory=list,
        description="chunk_id values (exactly as given in the context) that support the answer. "
        "Empty if the context does not answer the question.",
    )
