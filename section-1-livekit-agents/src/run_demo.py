"""Run the support agent against a real LLM and produce a transcript.

This is the script that generates the actual required evidence for Task
1.1: a real `AgentSession` (Agent persona + two real tools + mocked
STT/TTS), driven by a real LLM (Groq, with Cerebras fallback -- see
config.py), deciding on its own whether to call a tool based on what the
simulated caller says.

Usage
-----
    cp .env.example .env        # then paste in a free GROQ_API_KEY
    pip install -r requirements.txt
    python src/run_demo.py                 # runs the scripted conversation
    python src/run_demo.py --interactive    # type your own turns instead
    python src/run_demo.py --verbose        # also show raw JSON logs on console

By default the console only shows the clean conversation (Customer / Sam /
tool_call lines) -- pass --verbose to also see the raw structured JSON log
stream (HTTP requests, fallback/retry events, etc.), which is what's
useful when debugging rather than reading a transcript.

Every turn is always logged as structured JSON to
`transcripts/<timestamp>_session.jsonl` regardless of --verbose, which is
what gets submitted as the "transcript/log" evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ConfigurationError, build_llm  # noqa: E402
from livekit.agents import (  # noqa: E402
    AgentSession,
    APIError,
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)
from logging_config import configure_logging  # noqa: E402
from persona import SupportAgent  # noqa: E402
from providers import build_stt, build_tts  # noqa: E402

# A short but complete scripted conversation: greeting, a happy-path status
# lookup, a cancel attempt that the tool correctly rejects, and a cancel
# attempt that succeeds. This is short enough to read in under a minute
# (per the task's own evidence bar) while still exercising both tools and
# both the happy and error paths -- see the plan discussion in NOTES.md for
# why this length/shape was chosen.
SCRIPTED_TURNS = [
    "Hi, I'd like to check on an order.",
    "It's order A100 -- what's the status?",
    "Actually can you cancel order A101 for me?",
    "Okay, then please cancel order A100 instead.",
    "That's all, thanks!",
]


def _print_and_log(logger, label: str, text: str) -> None:
    print(f"{label}: {text}")
    logger.info("transcript_line", extra={"speaker": label, "text": text})


def _compact(text: str, limit: int = 70) -> str:
    """Collapse whitespace and truncate long output for a one-line summary."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _format_tool_line(name: str, arguments_json: str, is_error: bool, output: str) -> str:
    """Render one tool call + its result as a single readable line:
    tool_call: (get_order_status, A100, OK: Order A100 is currently 'preparing'...)
    """
    try:
        args = json.loads(arguments_json)
        arg_str = ", ".join(str(v) for v in args.values()) if args else "-"
    except (json.JSONDecodeError, AttributeError, TypeError):
        arg_str = arguments_json
    tag = "ERROR" if is_error else "OK"
    return f"tool_call: ({name}, {arg_str}, {tag}: {_compact(output)})"


async def run_conversation(*, interactive: bool, logger) -> None:
    llm_instance = build_llm()
    session = AgentSession(stt=build_stt(), tts=build_tts(), llm=llm_instance)
    agent = SupportAgent()

    await session.start(agent=agent)
    logger.info("session_started", extra={"interactive": interactive})

    try:
        turn_source = _interactive_turns() if interactive else iter(SCRIPTED_TURNS)
        for user_text in turn_source:
            _print_and_log(logger, "Customer", user_text)

            try:
                result = await session.run(user_input=user_text, input_modality="text")
            except APIError as e:
                # The FallbackAdapter (config.py) already retries across
                # providers internally -- if we still end up here, every
                # candidate LLM genuinely failed. We deliberately do NOT
                # re-call session.run() with the same user_input to "retry":
                # verified in dev that doing so appends a second, duplicate
                # copy of the user's turn to the session's chat history
                # (session.history), silently corrupting the conversation
                # the LLM sees on every later turn. Instead we degrade
                # gracefully once and let the *next* real user turn proceed
                # normally -- exactly the "canned degraded response" fallback
                # described in NOTES.md's predicted-failure-modes section.
                logger.error("llm_call_failed", extra={"error": str(e)})
                _print_and_log(
                    logger,
                    "Sam (agent)",
                    "Sorry, I'm having trouble reaching our systems right now. "
                    "Could you say that again in a moment?",
                )
                continue

            pending_calls: dict[str, FunctionCallEvent] = {}
            for event in result.events:
                if isinstance(event, FunctionCallEvent):
                    pending_calls[event.item.call_id] = event
                    logger.info(
                        "tool_call_invoked",
                        extra={"tool": event.item.name, "arguments": event.item.arguments},
                    )
                elif isinstance(event, FunctionCallOutputEvent):
                    call_event = pending_calls.pop(event.item.call_id, None)
                    name = call_event.item.name if call_event else event.item.name
                    arguments = call_event.item.arguments if call_event else "{}"
                    logger.info(
                        "tool_call_result",
                        extra={
                            "tool": name,
                            "is_error": event.item.is_error,
                            "output": event.item.output,
                        },
                    )
                    line = _format_tool_line(name, arguments, event.item.is_error, event.item.output)
                    print(line)
                elif isinstance(event, ChatMessageEvent) and event.item.role == "assistant":
                    _print_and_log(logger, "Sam (agent)", event.item.text_content or "")
    finally:
        await session.aclose()
        logger.info("session_ended")


def _interactive_turns():
    print("Interactive mode -- type a message, or 'quit' to end.\n")
    while True:
        try:
            text = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError: stdin closed (e.g. terminal/pipe closed) instead of
            # typing "quit". KeyboardInterrupt: Ctrl+C while waiting on
            # input(). Both used to propagate all the way up through the
            # async generator and out of asyncio.run(), producing a raw
            # traceback -- end the conversation cleanly instead.
            print("\n(input closed -- ending conversation)")
            return
        if text.lower() in {"quit", "exit"}:
            return
        if text:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive", action="store_true", help="Type your own turns instead of the script."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Also show raw JSON logs on the console."
    )
    args = parser.parse_args()

    load_dotenv()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = Path(__file__).resolve().parents[1] / "transcripts" / f"{timestamp}_session.jsonl"
    logger = configure_logging(log_file=log_file, console_json=args.verbose)

    try:
        asyncio.run(run_conversation(interactive=args.interactive, logger=logger))
    except ConfigurationError as e:
        print(f"\nConfiguration error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        # Ctrl+C while awaiting the network (not while sitting at the
        # input() prompt -- that case is handled inside _interactive_turns)
        # used to produce a raw traceback plus a dangling-coroutine
        # RuntimeWarning from the session shutting down mid-await. Exit
        # quietly instead; the transcript up to this point is still on disk.
        print("\n(interrupted -- ending conversation)")
        sys.exit(130)

    print(f"\nTranscript written to: {log_file}")


if __name__ == "__main__":
    main()
