"""
Simple interactive client: type a prompt, watch tokens print live as they
stream in from the running API. Run multiple copies of this script in
separate terminals to manually exercise concurrency against the same
server -- a human-driven complement to scripts/load_test.py, and a good way
to visually confirm streaming actually streams (not just buffers and dumps).

Usage:
    python scripts/interactive_client.py --url http://localhost:8000
"""
import argparse
import sys

import httpx


def stream_once(url: str, prompt: str, max_new_tokens: int, timeout: float) -> None:
    with httpx.Client() as client:
        try:
            with client.stream(
                "POST", f"{url}/generate/stream",
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
                timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode(errors="replace")
                    print(f"[HTTP {resp.status_code}] {body}")
                    return
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        print()  # trailing newline
                        return
                    if payload.startswith("[ERROR]"):
                        print(f"\n[ERROR] {payload}")
                        return
                    print(payload, end="", flush=True)
        except httpx.TimeoutException:
            print(f"\n[client timeout after {timeout}s -- server may be queued behind other requests]")
        except httpx.ConnectError:
            print(f"\n[could not connect to {url} -- is the server running?]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    print(f"Connecting to {args.url} ...")
    try:
        with httpx.Client() as client:
            health = client.get(f"{args.url}/health", timeout=10).json()
        print(f"Health: {health}")
        if not health.get("ready"):
            print("Warning: model is not ready yet -- requests may 503 until it finishes loading.")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach {args.url}/health ({exc}). Is the server running?")
        sys.exit(1)

    print("\nType a prompt and press Enter. Ctrl+C to quit.")
    print("Tip: run this script in multiple terminals at once to manually test concurrency.\n")

    while True:
        try:
            prompt = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not prompt:
            continue
        stream_once(args.url, prompt, args.max_new_tokens, args.timeout)
        print()


if __name__ == "__main__":
    main()
