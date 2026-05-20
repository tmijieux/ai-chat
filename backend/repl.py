#!/usr/bin/env python3
"""CLI REPL for the agent — mirrors the web client but runs in a terminal."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.agent import AgentSession, run_agent
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list

GREY   = "\033[90m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
RESET  = "\033[0m"


def _print_event(event: dict) -> None:
    t = event["type"]
    if t == "thinking":
        print(f"{GREY}{event.get('content', '')}{RESET}", end="", flush=True)
    elif t == "content":
        print(event.get("content", ""), end="", flush=True)
    elif t == "tool_result":
        content = event.get("content", "")
        preview = content[:300] + ("…" if len(content) > 300 else "")
        print(f"\n{CYAN}[{event.get('tool_name', 'tool')}] {preview}{RESET}")
    elif t == "iteration_end":
        print(f"\n{GREY}[{event.get('prompt_tokens', 0)} prompt tokens]{RESET}")
    elif t == "error":
        print(f"\n{RED}[error] {event.get('message')}{RESET}")
    elif t == "done":
        print()


async def _run_turn(
    messages: list[dict],
    tools: list[dict],
    working_directory: str | None,
) -> None:
    session = AgentSession()
    agent_task = asyncio.create_task(run_agent(session, messages, tools, working_directory))

    while True:
        event = await session.outbound.get()

        if event["type"] == "tool_confirm":
            print(f"\n{YELLOW}[confirm] {event.get('tool_name', '')}\n{event.get('preview', '')}{RESET}")
            answer = input("Approve? [y/n]: ").strip().lower()
            reason = None
            if answer != "y":
                reason = input("Reason (optional): ").strip() or None
            session.resolve_confirm(event["tool_id"], answer == "y", reason)
        else:
            _print_event(event)

        if event["type"] in ("done", "error"):
            break

    await agent_task


async def main() -> None:
    import os
    print("Agent REPL  —  type 'exit' to quit\n")
    # raw = input("Workspace directory (empty to disable file tools): ").strip()
    # working_directory = raw or None
    working_directory = os.path.realpath(os.getcwd())
    
    tools = get_ollama_tool_list(list(TOOL_REGISTRY.keys()))
    messages: list[dict] = []

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})
        try:
            await _run_turn(messages, tools, working_directory)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[aborted]{RESET}")
            messages.pop()


if __name__ == "__main__":
    asyncio.run(main())
