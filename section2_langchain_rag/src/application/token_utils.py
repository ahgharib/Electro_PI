"""Approximate, dependency-free token counting.

Deliberately NOT using tiktoken here: tiktoken's encoders download their
BPE merge tables from an external blob store the first time they're used,
which means chunk-splitting and the question-length guardrail -- two
fairly foundational, frequently-called pieces of this system -- would
silently depend on a third-party CDN being reachable. That's a bad
property for infrastructure code, and it genuinely fails in network-
restricted environments (firewalled corporate networks, air-gapped
deployments, this test environment itself).

Neither use case here needs an exact token count: chunk sizing just needs
a consistent, roughly-proportional length metric, and the question-length
guardrail just needs a reasonable ceiling. Real token counts for cost/
billing purposes always come from the provider's own API response
(usage_metadata), never from this heuristic.

~1.3 tokens per whitespace-delimited word is a standard rule-of-thumb
approximation for English text and is within the right order of magnitude
for both use cases.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")
_TOKENS_PER_WORD = 1.3


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    words = _WORD_RE.findall(text)
    return max(1, round(len(words) * _TOKENS_PER_WORD))
