# Section 1 — Write-up

## Architecture summary

`AgentSession(stt=MockSTT(), tts=MockTTS(), llm=<Groq, falling back to Cerebras>)`
running a `SupportAgent(Agent)` persona with two real `@function_tool`
methods (`get_order_status`, `cancel_order`). STT/TTS are mocked (real
subclasses of the SDK's `stt.STT`/`tts.TTS` base classes, not a bypass of
the framework); the LLM is real. See `src/mock_providers.py` and
`src/config.py` for why, and `transcripts/README.md` for how to generate
the live evidence transcript against the real LLM.

---

## 1. Extending this to support barge-in / interruption handling

This agent doesn't need to *add* barge-in support so much as *configure*
it: `AgentSession` handles interruption by default (`InterruptionOptions`
inside `TurnHandlingOptions`), and the reason it's invisible in this demo
is that MockTTS produces silence with no real playback device, and the
scripted/interactive demo drives turns via `input_modality="text"` rather
than a live audio stream — there's no ongoing agent speech to interrupt.

To make barge-in real, this needs two things this MVP deliberately doesn't
have, per the "mock STT/TTS" allowance in the task:

1. **A real audio path.** `session.start(agent=agent, room=room)` against
   an actual LiveKit room, with a real STT that streams `interim_transcript`
   events while the agent is talking, and a real TTS whose audio is
   actually being played back (so there's something to interrupt).
2. **Tuning `InterruptionOptions`** for the persona. The framework exposes
   this directly, no custom logic required:
   - `mode`: `"vad"` (interrupt on any detected speech energy) vs
     `"adaptive"` (an ML classifier that tries to tell a real interruption
     from a backchannel like "mm-hmm" or "right" and *doesn't* interrupt on
     those). For a support bot reading back sensitive info (e.g. "the
     refund will be...") adaptive mode avoids being cut off by acknowledgic
     noises.
   - `min_duration` / `min_words`: how much of the user talking over the
     agent counts as a real interruption vs. noise. The defaults (0.5s / 0
     words) are reasonable for a support bot; I'd raise `min_duration`
     slightly (~0.7s) for a persona that reads order details, since a
     half-second blip is more likely to be noise than a real interruption
     mid-sentence.
   - `resume_false_interruption` + `false_interruption_timeout`: if the
     agent stops because it *thought* it was interrupted but the caller
     didn't actually say anything meaningful, it resumes the same
     utterance rather than restarting the turn. Worth keeping on for a
     support bot — restarting "your order is..." from scratch every time
     someone coughs is a bad experience.
   - `backchannel_boundary`: suppresses false interruptions right at the
     start/end of the agent's turn, which is exactly when "okay" / "right"
     overlap tends to happen.

None of this requires touching `SupportAgent` or the tools at all — it's
purely `AgentSession` construction, which is the payoff of keeping STT/TTS
behind the standard interface from the start.

## 2. Adding a second tool safely

I didn't just design this — `cancel_order` in `src/persona.py` **is** the
second tool, built specifically to exercise the unhappy path. The pattern,
generalized:

- **Schema**: `@function_tool` derives the JSON schema the LLM sees from
  the method's type hints and docstring (`Args:` block per-parameter). Keep
  parameters primitive and unambiguous (`order_id: str`, not a dict the
  model has to guess the shape of), and put constraints the model should
  respect in the docstring, since that's what it actually reads.
- **Error handling**: never let a tool raise a bare exception. Catch the
  failure mode and raise `livekit.agents.llm.ToolError(message)` instead —
  the framework catches this, marks the `FunctionCallOutputEvent` as
  `is_error=True`, and feeds the message back to the LLM as the tool
  result rather than crashing the turn. The system prompt tells the
  persona explicitly to explain tool errors in plain language rather than
  repeat them verbatim (see `SYSTEM_INSTRUCTIONS`) — this is what turns
  "ToolError: order not cancellable" into "sorry, that one's already out
  for delivery so I can't cancel it."
- **What to validate before the "real" logic runs**: existence (does the
  referenced entity exist at all) before state (is it in a state this
  operation is valid for) — `cancel_order` checks `order is None` before
  `not order.cancellable` for exactly this reason; the error messages are
  meaningfully different and the LLM should be able to tell "I don't know
  that order" from "I know it, but it's too late."
- **Side-effect safety**: `cancel_order` mutates state, so it's built to be
  safe to call twice — cancelling an already-cancelled order raises the
  same "no longer cancellable" `ToolError` rather than silently succeeding
  or double-processing (`tests/test_tools.py::test_cancellable_order_succeeds`
  checks this explicitly).

## 3. Task 1.2 (bonus) — swapping a pipeline component

*Full walkthrough with verified code for both swapping an existing
provider and adding a brand new one: [`SWAPPING_PROVIDERS.md`](./SWAPPING_PROVIDERS.md).*

This is no longer just a documented diff -- it's a real, working switch.
`AgentSession` was already provider-agnostic from day one (it just takes
any `stt.STT` / `tts.TTS` instance); what was missing was making that
switch something you flip, not something you'd have to hand-edit
`run_demo.py` to try. `src/providers.py` now adds `build_stt()` /
`build_tts()`, mirroring `config.py`'s `build_llm()` pattern exactly:

```bash
# Default -- mock, zero setup, what the README's quickstart uses:
python src/run_demo.py

# Swap to real providers -- no code changes, just env vars:
pip install -r requirements-optional-providers.txt
echo "STT_PROVIDER=deepgram" >> .env
echo "DEEPGRAM_API_KEY=..." >> .env
echo "TTS_PROVIDER=cartesia" >> .env
echo "CARTESIA_API_KEY=..." >> .env
python src/run_demo.py
```

`tests/test_providers.py` proves this mechanically: each provider
selection is asserted to construct the *right class* with the *right
arguments* (`deepgram.STT`, `cartesia.TTS`), including the missing-key and
missing-package error paths. Nothing in `persona.py` or
`tests/test_tools.py` changes at all, because neither ever imports
`mock_providers` or `providers` — they only see the `Agent`/tool-calling
surface, which was provider-independent by construction from the start.

What this still doesn't prove, for the same reason `build_llm()` couldn't
be proven against a real LLM from inside this dev sandbox: a live
conversation against real Deepgram/Cartesia audio needs a human with a
(free trial) key and a microphone, which I don't have here. What's
verified is that the switch is real and wired correctly; what's not
verified is the live audio round-trip itself. If you run the swap
yourself, that's also the point where barge-in (section 1 above) becomes
observable for the first time, since it needs a live audio stream to
actually interrupt.

The `livekit-plugins-deepgram`/`livekit-plugins-cartesia` packages live in
`requirements-optional-providers.txt`, not the base `requirements.txt` --
deliberately, so the README's "under 10 minutes, zero extra signups"
quickstart path stays true for anyone who just wants to see the core
tool-calling demo.

---

## Honest limitations / assumptions

- **Turn detection / VAD is not exercised** in the text-modality demo,
  since there's no real audio. See section 1 above for what changes once
  wired to a real audio path.
- **Memory is intentionally session-only** (the SDK's own `ChatContext`,
  accumulated automatically across `session.run()` calls — verified in
  dev that this persists correctly across turns). No persistent/long-term
  memory store, by design — out of scope for an MVP support-bot demo.
- **The mock order "database" is per-`SupportAgent`-instance, not a real
  service call.** Discovered and fixed a real bug during testing where it
  was originally a module-level dict shared across all agent instances,
  which made tests (and would make concurrent conversations) interfere
  with each other's order state.
- **Known, not yet fixed: occasional malformed `<function=...>` text in a
  reply** (see `CONVERSATIONS.md` #3). Rare, and never resulted in an
  actual duplicate tool call — but it's exactly the kind of thing that
  would sound broken through a real TTS, and the current sanitizer
  (`speech_sanitizer.py`) doesn't catch this pattern yet since it targets
  emoji/markdown, not leaked tool-call syntax.

## Highlights


- **Crash on transient LLM failure** — a partial response + failed tool
  call used to crash the script outright. Now degrades to a spoken apology
  instead (`retry_on_chunk_sent=True` in `config.py` + a try/except in
  `run_demo.py` that deliberately doesn't resubmit the turn, to avoid
  duplicating it in chat history).
- **Repeated tool calls on an already-known-missing order** — 
  an explicit instruction in `SYSTEM_INSTRUCTIONS` to reuse a result
  already established this conversation instead of re-querying.
- **Cerebras fallback model was invalid** (`404`) — swapped to the
  current, verified, tool-calling-capable free-tier model (`gpt-oss-120b`).
- **Console was a wall of interleaved JSON** — quiet by default now
  (conversation + one compact `tool_call: (...)` line per call);
  `--verbose` for the full stream for debugging.

Reasoning and code-level detail for each lives inline as comments in
`config.py`, `run_demo.py`, and `persona.py`, right next to the fix.
