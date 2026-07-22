"""Unit tests for the mock STT/TTS providers.

These exercise MockSTT/MockTTS through the *real* LiveKit `stt.STT` /
`tts.TTS` public methods (`recognize`, `synthesize`) -- not our own
internals -- so a passing test is evidence the mocks satisfy the actual
plugin interface real providers implement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mock_providers import MockSTT, MockTTS, TextAudioCodec  # noqa: E402


class TestTextAudioCodec:
    def test_round_trip_preserves_text(self) -> None:
        original = "What is the status of order A100?"
        frame = TextAudioCodec.encode(original)
        assert frame.sample_rate == 16_000
        assert frame.num_channels == 1
        assert TextAudioCodec.decode(frame) == original

    def test_handles_odd_byte_length_text(self) -> None:
        # UTF-8 for this string is an odd number of bytes; encode() must
        # pad to a whole number of int16 samples without corrupting the text.
        original = "cancel A101"
        frame = TextAudioCodec.encode(original)
        assert TextAudioCodec.decode(frame) == original


class TestMockSTT:
    @pytest.mark.asyncio
    async def test_recognize_returns_final_transcript_matching_input(self) -> None:
        stt_engine = MockSTT()
        frame = TextAudioCodec.encode("cancel order A100 please")

        event = await stt_engine.recognize(frame)

        assert event.type.value == "final_transcript"
        assert len(event.alternatives) == 1
        assert event.alternatives[0].text == "cancel order A100 please"
        assert event.alternatives[0].confidence == 1.0

    def test_capabilities_are_offline_only(self) -> None:
        stt_engine = MockSTT()
        assert stt_engine.capabilities.streaming is False
        assert stt_engine.capabilities.offline_recognize is True


class TestMockTTS:
    @pytest.mark.asyncio
    async def test_synthesize_produces_nonempty_audio(self) -> None:
        tts_engine = MockTTS()
        stream = tts_engine.synthesize("Your order is on its way.")

        frames = [audio.frame async for audio in stream]

        assert len(frames) >= 1
        total_bytes = sum(len(f.data) for f in frames)
        assert total_bytes > 0

    @pytest.mark.asyncio
    async def test_longer_text_produces_longer_audio(self) -> None:
        tts_engine = MockTTS()

        short_stream = tts_engine.synthesize("Okay.")
        short_frames = [a async for a in short_stream]
        short_bytes = sum(len(a.frame.data) for a in short_frames)

        long_stream = tts_engine.synthesize(
            "Your order has been received, is currently being prepared in the "
            "kitchen, and should be out for delivery within the next thirty "
            "minutes depending on traffic conditions in your area."
        )
        long_frames = [a async for a in long_stream]
        long_bytes = sum(len(a.frame.data) for a in long_frames)

        assert long_bytes > short_bytes

    def test_capabilities_report_no_streaming(self) -> None:
        tts_engine = MockTTS()
        assert tts_engine.capabilities.streaming is False
