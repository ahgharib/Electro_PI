"""Tests for build_stt() / build_tts() -- the Task 1.2 swap mechanism.

These prove the factories build the *right kind of object with the right
arguments* for each provider selection. They do NOT and cannot prove a
live conversation works against real Deepgram/Cartesia audio -- that needs
a real API key and network access neither this test environment nor the
dev sandbox this repo was built in has. See providers.py's module
docstring and NOTES.md for the honest scope of what's verified here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ConfigurationError  # noqa: E402
from mock_providers import MockSTT, MockTTS  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a clean provider-selection environment."""
    for var in ("STT_PROVIDER", "TTS_PROVIDER", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestBuildSTT:
    def test_defaults_to_mock_with_no_env_set(self) -> None:
        from providers import build_stt

        assert isinstance(build_stt(), MockSTT)

    def test_explicit_mock_selection(self, monkeypatch) -> None:
        from providers import build_stt

        monkeypatch.setenv("STT_PROVIDER", "mock")
        assert isinstance(build_stt(), MockSTT)

    def test_deepgram_without_api_key_raises_configuration_error(self, monkeypatch) -> None:
        from providers import build_stt

        monkeypatch.setenv("STT_PROVIDER", "deepgram")
        with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY"):
            build_stt()

    def test_deepgram_with_api_key_builds_real_deepgram_stt(self, monkeypatch) -> None:
        pytest.importorskip(
            "livekit.plugins.deepgram",
            reason="optional provider package -- pip install -r requirements-optional-providers.txt",
        )
        from providers import build_stt

        monkeypatch.setenv("STT_PROVIDER", "deepgram")
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-test-key")
        result = build_stt()

        from livekit.plugins import deepgram

        assert isinstance(result, deepgram.STT)

    def test_unknown_provider_raises_configuration_error(self, monkeypatch) -> None:
        from providers import build_stt

        monkeypatch.setenv("STT_PROVIDER", "not-a-real-provider")
        with pytest.raises(ConfigurationError, match="Unknown STT_PROVIDER"):
            build_stt()


class TestBuildTTS:
    def test_defaults_to_mock_with_no_env_set(self) -> None:
        from providers import build_tts

        assert isinstance(build_tts(), MockTTS)

    def test_cartesia_without_api_key_raises_configuration_error(self, monkeypatch) -> None:
        from providers import build_tts

        monkeypatch.setenv("TTS_PROVIDER", "cartesia")
        with pytest.raises(ConfigurationError, match="CARTESIA_API_KEY"):
            build_tts()

    def test_cartesia_with_api_key_builds_real_cartesia_tts(self, monkeypatch) -> None:
        pytest.importorskip(
            "livekit.plugins.cartesia",
            reason="optional provider package -- pip install -r requirements-optional-providers.txt",
        )
        from providers import build_tts

        monkeypatch.setenv("TTS_PROVIDER", "cartesia")
        monkeypatch.setenv("CARTESIA_API_KEY", "fake-test-key")
        result = build_tts()

        from livekit.plugins import cartesia

        assert isinstance(result, cartesia.TTS)

    def test_unknown_provider_raises_configuration_error(self, monkeypatch) -> None:
        from providers import build_tts

        monkeypatch.setenv("TTS_PROVIDER", "not-a-real-provider")
        with pytest.raises(ConfigurationError, match="Unknown TTS_PROVIDER"):
            build_tts()
