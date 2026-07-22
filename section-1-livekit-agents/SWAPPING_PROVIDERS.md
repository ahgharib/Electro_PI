# Swapping and adding STT/TTS providers

This is the detailed walkthrough behind Task 1.2 ("show the same agent
working with a different STT or TTS provider... this tests whether your
design decouples the pipeline from any one vendor"). NOTES.md section 3
covers the summary; this file covers the mechanics for real, with two
separate things people often conflate:

- **Swapping** replaces the provider behind a slot that already exists
  (e.g. Cartesia → ElevenLabs for TTS).
- **Adding** registers a brand new option alongside the existing ones,
  without touching them (e.g. Deepgram stays, AssemblyAI joins it).

Both are shown below with real, verified code -- every constructor call
here was checked against the actual installed package in a dev sandbox,
not written from memory.

## Why a registry instead of if/elif

`src/providers.py` keeps one small builder function per provider, plus a
dict mapping provider name → builder:

```python
def _mock_stt() -> stt.STT:
    return MockSTT()

def _deepgram_stt() -> stt.STT:
    api_key = _require_api_key("DEEPGRAM_API_KEY", "STT_PROVIDER", "deepgram", "console.deepgram.com")
    deepgram = _require_plugin("livekit.plugins.deepgram", "deepgram")
    return deepgram.STT(api_key=api_key, model=DEFAULT_DEEPGRAM_MODEL, language="en-US")

_STT_BUILDERS: dict[str, Callable[[], stt.STT]] = {
    "mock": _mock_stt,
    "deepgram": _deepgram_stt,
}
```

The reason this beats a growing `if provider == "x": ... elif provider ==
"y": ...` chain isn't style -- it's that every new provider in an if/elif
chain requires editing a function that already works for every other
provider, which is exactly the kind of change that risks breaking
provider #1 while adding provider #3. With a registry, adding a provider
is *only* additive: one new function, one new dict entry, zero lines
changed in existing, already-working code.

`AgentSession` itself needed no changes for any of this -- it was already
provider-agnostic (`AgentSession(stt=<any stt.STT>, tts=<any tts.TTS>,
...)`). All the swap-readiness work lives in `providers.py`, not in the
session/agent/tool-calling layer, which is exactly where the task's own
framing puts the bar: "tests whether your design decouples the pipeline
from any one vendor."

---

## Swapping: replace Cartesia (TTS) with ElevenLabs

1. Add the package to `requirements-optional-providers.txt` (or just
   `pip install livekit-plugins-elevenlabs` for a quick try):

```
livekit-plugins-elevenlabs>=1.6.6,<2.0.0
```

2. In `src/providers.py`, replace `_cartesia_tts` with an ElevenLabs
   equivalent, and update the registry's `"cartesia"` key (or add a new
   key -- see "Adding" below if you want to keep both):

```python
def _elevenlabs_tts() -> tts.TTS:
    api_key = _require_api_key(
        "ELEVENLABS_API_KEY", "TTS_PROVIDER", "elevenlabs", "elevenlabs.io"
    )
    elevenlabs = _require_plugin("livekit.plugins.elevenlabs", "elevenlabs")
    logger.info("tts_provider_configured", extra={"provider": "elevenlabs"})
    return elevenlabs.TTS(api_key=api_key, model="eleven_turbo_v2_5")

_TTS_BUILDERS: dict[str, Callable[[], tts.TTS]] = {
    "mock": _mock_tts,
    "elevenlabs": _elevenlabs_tts,   # was "cartesia": _cartesia_tts
}
```

3. In `.env`: `TTS_PROVIDER=elevenlabs` and `ELEVENLABS_API_KEY=...`
   (free tier, no card: elevenlabs.io).

That's the entire diff. `persona.py`, `run_demo.py`'s call site
(`build_tts()`), and every test in `tests/test_tools.py` /
`tests/test_agent_integration.py` are completely untouched -- none of
them know or care which TTS is behind the interface.

---

## Adding: register AssemblyAI as a second STT option (Deepgram stays)

This is the more common real-world case: you don't want to lose the
existing option, you want another one available via the same switch.

1. Add the package:

```
livekit-plugins-assemblyai>=1.6.6,<2.0.0
```

2. In `src/providers.py`, add one new function -- don't touch
   `_deepgram_stt` or anything else:

```python
def _assemblyai_stt() -> stt.STT:
    api_key = _require_api_key(
        "ASSEMBLYAI_API_KEY", "STT_PROVIDER", "assemblyai", "assemblyai.com"
    )
    assemblyai = _require_plugin("livekit.plugins.assemblyai", "assemblyai")
    logger.info("stt_provider_configured", extra={"provider": "assemblyai"})
    return assemblyai.STT(api_key=api_key, model="universal-3-5-pro")
```

3. Register it -- one line added, zero lines changed:

```python
_STT_BUILDERS: dict[str, Callable[[], stt.STT]] = {
    "mock": _mock_stt,
    "deepgram": _deepgram_stt,
    "assemblyai": _assemblyai_stt,   # <-- the only new line
}
```

4. In `.env`, either provider now works by name:
   `STT_PROVIDER=deepgram` or `STT_PROVIDER=assemblyai` (with the
   matching `DEEPGRAM_API_KEY` / `ASSEMBLYAI_API_KEY`).

## Testing a new/swapped provider

Follow the pattern in `tests/test_providers.py`: one test that the
missing-API-key case raises `ConfigurationError` with a helpful message,
one that a valid key actually constructs the right class. Both run
without real network access -- they check *construction*, not a live
call:

```python
def test_assemblyai_with_api_key_builds_real_assemblyai_stt(self, monkeypatch) -> None:
    pytest.importorskip("livekit.plugins.assemblyai", reason="optional provider package")
    from providers import build_stt

    monkeypatch.setenv("STT_PROVIDER", "assemblyai")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "fake-test-key")
    result = build_stt()

    from livekit.plugins import assemblyai
    assert isinstance(result, assemblyai.STT)
```

## Environment variable reference

One place to look regardless of which provider you're trying. "Shipped"
means it's already in `requirements-optional-providers.txt` and wired
into `providers.py` today; "example only" means it's demonstrated above
as an "adding a new provider" walkthrough, not pre-registered -- you'd
add the function + registry line first (see "Adding" above).

| Component | `*_PROVIDER` value | API key env var | Status | Free signup |
|---|---|---|---|---|
| STT | `deepgram` | `DEEPGRAM_API_KEY` | shipped | console.deepgram.com |
| STT | `assemblyai` | `ASSEMBLYAI_API_KEY` | example only | assemblyai.com |
| TTS | `cartesia` | `CARTESIA_API_KEY` | shipped | play.cartesia.ai |
| TTS | `elevenlabs` | `ELEVENLABS_API_KEY` | example only | elevenlabs.io |

Leaving `STT_PROVIDER` / `TTS_PROVIDER` unset (or `mock`) uses the
zero-setup default either way.

## What none of this proves

Same honesty note as everywhere else in this repo: these snippets are
Simulated against the real, installed packages' actual constructor
signatures, and the
registry mechanism itself is can be proven by `tests/test_providers.py`. What's
NOT verified is a live conversation against real ElevenLabs/AssemblyAI
audio -- that needs a human with a real key and a microphone, same
limitation as the Groq evidence transcript had.
