"""Unit tests for the speech sanitizer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speech_sanitizer import sanitize_for_speech  # noqa: E402


class TestSanitizeForSpeech:
    def test_strips_emoji(self) -> None:
        assert sanitize_for_speech("Your order is on its way! 🎉🚀") == "Your order is on its way!"

    def test_strips_markdown_bold(self) -> None:
        assert sanitize_for_speech("Your order **A100** is ready") == "Your order A100 is ready"

    def test_strips_markdown_header_and_bullets(self) -> None:
        assert sanitize_for_speech("# Status\n- preparing\n- 25 min") == "Status\npreparing\n25 min"

    def test_strips_backticks_and_underscores(self) -> None:
        assert sanitize_for_speech("order_id `A100` confirmed") == "orderid A100 confirmed"

    def test_leaves_normal_text_and_numbers_untouched(self) -> None:
        text = "Order A100 is currently preparing, ETA 25 minutes."
        assert sanitize_for_speech(text) == text

    def test_collapses_double_spaces_left_by_stripped_characters(self) -> None:
        assert sanitize_for_speech("ready **now**") == "ready now"

    def test_handles_emoji_only_string(self) -> None:
        assert sanitize_for_speech("🎉🚀✅") == ""
