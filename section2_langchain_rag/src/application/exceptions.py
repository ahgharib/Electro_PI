"""Custom exceptions.

Adapters raise these instead of leaking provider-specific exceptions
(google.api_core errors, openai.APIError, etc.) up through the ports. This
keeps the application layer able to catch one small, known set of failure
types regardless of which provider is behind a port.
"""


class RagError(Exception):
    """Base class for all application-level errors."""


class ComponentError(RagError):
    """A component (embedder, generator, vector store) failed after retries."""

    def __init__(self, component: str, provider: str, original: Exception | None = None):
        self.component = component
        self.provider = provider
        self.original = original
        msg = f"{component} ({provider}) failed"
        if original is not None:
            msg += f": {original}"
        super().__init__(msg)


class QuotaExceededError(ComponentError):
    """The provider rejected the call specifically due to a rate limit or
    daily quota being exceeded (HTTP 429 / RESOURCE_EXHAUSTED), as opposed
    to some other failure. Distinguished from a generic ComponentError
    because the correct response is different: retrying sooner won't help
    for a daily quota, and the user-facing message should say so rather
    than implying a transient blip."""


class NoProviderAvailableError(RagError):
    """Both the primary and fallback provider for a component failed."""


class QuestionTooLongError(RagError):
    """The user's question exceeds the configured token limit."""

    def __init__(self, token_count: int, limit: int):
        self.token_count = token_count
        self.limit = limit
        super().__init__(
            f"Question is {token_count} tokens, which exceeds the {limit} token limit."
        )


class EmptyIndexError(RagError):
    """The vector store has nothing indexed."""
