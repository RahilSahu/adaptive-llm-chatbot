#!/usr/bin/env python3
"""Adaptive LLM Chatbot: interactive CLI.

Usage:
    python chat.py                 # provider auto-detected from API keys
    python chat.py --demo          # no key needed, watch the pipeline work
    python chat.py --provider openai --model gpt-4o

In-chat commands:
    /temp          show the current temperature mode
    /temp 0.3      pin the temperature (manual override)
    /temp auto     return to self-adjusting temperature
    /extract <txt> run structured extraction on a text
    /events        show the last turn's pipeline events
    /quit          exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import config, extraction, graph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive LLM Chatbot")
    parser.add_argument("--demo", action="store_true", help="force the keyless demo model")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google", "demo"])
    parser.add_argument("--model", help="model name override (or set CHATBOT_MODEL)")
    parser.add_argument("--thread", default="cli", help="conversation thread id (memory key)")
    args = parser.parse_args()

    if args.model:
        os.environ["CHATBOT_MODEL"] = args.model
    provider = "demo" if args.demo else (args.provider or config.detect_provider())

    tracing = config.setup_tracing()
    print("Adaptive LLM Chatbot")
    print(f"  provider: {provider}  model: {config.model_name(provider)}")
    print(f"  langsmith: {'ON, project ' + str(tracing.project) if tracing.enabled else 'off (' + tracing.reason + ')'}")
    print("  commands: /temp, /temp 0.3, /temp auto, /extract <text>, /events, /quit")
    print()

    override: float | None = None
    app = graph.build_app(provider, temp_override=override)
    last_events: list[str] = []

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/events":
            for event in last_events:
                print(f"  {event}")
            continue
        if user_input.startswith("/temp"):
            parts = user_input.split()
            if len(parts) == 1:
                print(f"  temperature mode: {'manual ' + str(override) if override is not None else 'auto (self-adjusting)'}")
            elif parts[1] == "auto":
                override = None
                app = graph.build_app(provider, temp_override=None)
                print("  temperature: auto (self-adjusting)")
            else:
                try:
                    override = float(parts[1])
                    app = graph.build_app(provider, temp_override=override)
                    print(f"  temperature pinned to {override}")
                except ValueError:
                    print("  usage: /temp 0.3  or  /temp auto")
            continue
        if user_input.startswith("/extract "):
            text = user_input[len("/extract "):]
            result = extraction.extract_info(text)
            print(f"  engine: {result['engine']}")
            print(json.dumps(result["extractions"], indent=2, ensure_ascii=False))
            continue

        state = graph.run_turn(app, user_input, thread_id=args.thread)
        last_events = state.get("events", [])
        reply = state["messages"][-1].content
        print(f"bot> {reply}")
        if state.get("decision_mode"):
            print(
                f"     [mode: {state['decision_mode']} | temperature: "
                f"{state.get('decision_temperature')} | thread: {args.thread}]"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
