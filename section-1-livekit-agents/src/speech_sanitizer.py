"""Sanitizes text right before it's spoken.

Two layers, deliberately not just one:

1. `SYSTEM_INSTRUCTIONS` (persona.py) asks the LLM not to produce emojis,
   markdown, or unspeakable symbols in the first place. This is the *first*
   line of defense, and it's the one that keeps ordinary text natural
   (it's a prompt, so it can't reliably enforce anything).
2. This module is the *second* line of defense: a hard guarantee, applied
   in `SupportAgent.tts_node` right before text reaches the TTS engine. A
   prompt instruction is probabilistic -- models occasionally ignore it
   (we've already seen a couple of instruction-following gaps in this
   project) -- so nothing downstream of this function should ever have to
   deal with an emoji or a stray markdown character reaching a speaker.

What this deliberately does NOT do: convert digits to spelled-out words
(e.g. "25" -> "twenty-five"). Real TTS engines already pronounce numbers
correctly; doing this ourselves would mean pulling in a numbers-to-words
library and handling currency/ordinals/decimals correctly, which is a
real chunk of complexity for a problem that isn't actually broken in
practice with real providers -- it would only cosmetically affect Mock TTS's
text log, not anything a customer would ever hear. Kept in scope: things
that are unambiguously wrong for *any* voice pipeline (emoji, markdown).
"""

from __future__ import annotations

import re

# Common emoji blocks + variation selector + zero-width joiner (covers the
# overwhelming majority of emoji without needing an external dependency).
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicator (flag) letters
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U0000fe0f"  # variation selector-16 (emoji presentation)
    "\U0000200d"  # zero-width joiner (used in compound emoji)
    "]+",
    flags=re.UNICODE,
)

# Markdown formatting characters that have no natural spoken form. This is a
# support-bot voice agent, so there's no legitimate reason for any of these
# to appear in what the model says out loud.
_MARKDOWN_CHARS = re.compile(r"[*_#`~]")

# List-bullet markers at the start of a line ("- item" / "* item"). Matched
# only at line-start (with required trailing whitespace) so this never
# touches legitimate hyphenated words like "out-for-delivery" mid-sentence.
_BULLET_PREFIX = re.compile(r"^[*-]\s+", flags=re.MULTILINE)

_EXTRA_WHITESPACE = re.compile(r" {2,}")


def sanitize_for_speech(text: str) -> str:
    """Strip emoji and markdown formatting from text bound for TTS."""
    text = _EMOJI_PATTERN.sub("", text)
    text = _BULLET_PREFIX.sub("", text)
    text = _MARKDOWN_CHARS.sub("", text)
    text = _EXTRA_WHITESPACE.sub(" ", text)
    return text.strip()
