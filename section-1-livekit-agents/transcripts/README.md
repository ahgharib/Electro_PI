# Transcripts

## `offline_smoke_test_transcript.txt`
A transcript from `run_demo.py`'s conversation loop with the real LLM
temporarily swapped for the offline `ScriptedLLM` test double. This proves
the session/agent/tool-calling/logging plumbing works end-to-end. It is
**not** evidence of a real LLM invoking a tool -- see the disclaimer at the
top of that file.

## Generating the real evidence transcript

This needs a free API key, which isn't something that can be baked into
the repo. It takes about two minutes:

1. Get a free Groq key (no credit card): https://console.groq.com/keys
2. `cp .env.example .env` and paste the key in as `GROQ_API_KEY`
3. `pip install -r requirements.txt`
4. `python src/run_demo.py`

This runs the scripted 5-turn conversation in `src/run_demo.py` against the
real model and writes:
- a live console transcript (also what you'd screen-record for the "short
  screen recording" option in the task)
- `transcripts/<timestamp>_session.jsonl`, a structured JSON-lines log of
  every turn, tool call, tool result, and assistant message

Rename or copy that file to something like `live_demo_transcript.jsonl`
before submitting, so it's not confused with the offline one above.

Want to see your own message trigger a tool call instead of the script?
Run `python src/run_demo.py --interactive` and type your own turns.
