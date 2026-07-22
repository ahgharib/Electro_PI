"""Mocked STT and TTS providers.

Why these exist and what they actually are
--------------------------------------------
The task explicitly allows mocking STT/TTS with text I/O as long as the LLM
and tool-calling logic is real. Crucially, "mocking" here means mocking the
*providers*, not the *framework*: both classes below are real subclasses of
LiveKit's `stt.STT` / `tts.TTS` abstract base classes, implementing the
actual methods the framework calls (`_recognize_impl`, `synthesize`). This
is what makes the pipeline still genuinely "STT -> LLM -> TTS" shaped, and
what makes Task 1.2 (swapping a provider) a one-line change instead of a
rewrite: any code that talks to `session.stt` / `session.tts` doesn't know
or care that these are stubs.

MockTTS is a "real" mock in the fullest sense: given text, it produces
actual valid PCM16 audio frames (silence, with duration proportional to a
typical speaking rate) through the real `AudioEmitter` protocol, and logs
the text it "spoke". It would work if wired into a real audio output today.

MockSTT is more clearly a stand-in. There is no real audio in this MVP (no
microphone, no room), so there is nothing genuine for it to "recognize". We
still implement it as a real STT so it's independently testable and
demonstrates the interface -- but the round-trip only works with audio
frames produced by `TextAudioCodec.encode` in this same module (i.e. audio
we ourselves fabricated to carry text). See NOTES.md for the write-up on
why the live demo instead drives conversation turns through
`AgentSession.run(..., input_modality="text")`, the SDK's own first-party
text-input path, rather than forcing fabricated audio through VAD.
"""

from __future__ import annotations

import logging

from livekit import rtc
from livekit.agents import stt, tts
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import shortuuid

logger = logging.getLogger("voice_agent.mock_providers")

SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
WORDS_PER_MINUTE = 150  # rough average speaking rate, used to size mock audio


class TextAudioCodec:
    """Packs a UTF-8 string into a valid (silent) `rtc.AudioFrame` and back.

    This is not audio compression -- it's a documented convention so
    MockSTT/MockTTS can be exercised end-to-end (encode -> real AudioFrame
    object -> decode) in isolation, without a real microphone or speech
    model. Real speech never round-trips through this; it exists purely to
    let the mock STT half of the pair be testable as a real STT.
    """

    @staticmethod
    def encode(text: str) -> rtc.AudioFrame:
        data = text.encode("utf-8")
        if len(data) % 2 != 0:  # int16 samples must be an even number of bytes
            data += b"\x00"
        samples_per_channel = len(data) // 2
        return rtc.AudioFrame(
            data=data,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=samples_per_channel,
        )

    @staticmethod
    def decode(frame: rtc.AudioFrame) -> str:
        return bytes(frame.data).rstrip(b"\x00").decode("utf-8", errors="replace")


class MockSTT(stt.STT):
    """Text-in-a-box STT: decodes text that was encoded via `TextAudioCodec`."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )

    @property
    def model(self) -> str:
        return "text-io-stub"

    @property
    def provider(self) -> str:
        return "mock"

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        text = TextAudioCodec.decode(frame)

        logger.info("mock_stt_recognized", extra={"text": text})

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=shortuuid("mock_stt_"),
            alternatives=[
                stt.SpeechData(language="en-US", text=text, confidence=1.0)
            ],
        )


class _MockChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: "MockTTS", input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        text = self._input_text
        logger.info("mock_tts_speaking", extra={"text": text})

        # Silence, sized to roughly how long a real TTS would take to say
        # this out loud, so the pipeline's timing/duration bookkeeping
        # (metrics, transcript alignment) behaves like it would with a real
        # provider.
        word_count = max(len(text.split()), 1)
        duration_s = max((word_count / WORDS_PER_MINUTE) * 60.0, 0.3)
        num_samples = int(SAMPLE_RATE * duration_s)
        silence = b"\x00\x00" * num_samples  # 16-bit PCM silence

        output_emitter.initialize(
            request_id=shortuuid("mock_tts_"),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        output_emitter.push(silence)


class MockTTS(tts.TTS):
    """Text-in-a-box TTS: emits real (silent) audio frames and logs the text."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

    @property
    def model(self) -> str:
        return "text-io-stub"

    @property
    def provider(self) -> str:
        return "mock"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _MockChunkedStream(tts_instance=self, input_text=text, conn_options=conn_options)
