#!/usr/bin/env python3
"""CLI REPL for the agent — mirrors the web client but runs in a terminal."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _drain_stdin() -> None:
    """Discard any extra characters sitting in stdin (e.g. from a multi-line paste)."""
    if sys.platform == "win32":
        import msvcrt
        import time
        time.sleep(0.05)  # let paste finish arriving
        while msvcrt.kbhit():
            msvcrt.getwch()
    else:
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()


def _read_line(prompt: str) -> str:
    """Read one line and drain any paste overflow from stdin."""
    line = input(prompt).strip()
    _drain_stdin()
    return line

from agent.agent import AgentSession, run_agent
from agent.pipeline import PipelineOrchestrator
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from llm import backend

GREY   = "\033[90m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
RED    = "\033[31m"
RESET  = "\033[0m"

_active_stage: list[str | None] = [None]  # mutable cell to track current stage across calls


def _print_event(event: dict) -> None:
    """Print a single agent event. Prints a stage header when the pipeline stage changes."""
    stage = event.get("_pipeline_stage")

    if stage != _active_stage[0]:
        _active_stage[0] = stage
        if stage is not None:
            print(f"\n{GREEN}━━━ [{stage}] ━━━{RESET}")
        else:
            print(f"\n{GREEN}━━━ [synthesize] ━━━{RESET}")

    t = event["type"]
    if t == "thinking":
        print(f"{GREY}{event.get('content', '')}{RESET}", end="", flush=True)
    elif t == "content":
        print(event.get("content", ""), end="", flush=True)
    elif t == "tool_call_start":
        print(f"\n{YELLOW}[{event.get('tool_name', '')}] ", end="", flush=True)
    elif t == "tool_call_chunk":
        print(f"{YELLOW}{event.get('chunk', '')}{RESET}", end="", flush=True)
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
    """Classic agent turn — mutates messages in-place with the assistant response."""
    session = AgentSession()
    agent_task = asyncio.create_task(run_agent(session, messages, tools, working_directory))

    while True:
        event = await session.outbound.get()

        if event["type"] == "tool_confirm":
            print(f"\n{YELLOW}[confirm] {event.get('tool_name', '')}\n{event.get('preview', '')}{RESET}")
            answer = _read_line("Approve? [y/n]: ").lower()
            reason = None
            if answer != "y":
                reason = _read_line("Reason (optional): ") or None
            session.resolve_confirm(event["tool_id"], answer == "y", reason)
        else:
            _print_event(event)

        if event["type"] in ("done", "error"):
            break

    await agent_task


async def _run_pipeline_turn(
    messages: list[dict],
    tools: list[dict],
    working_directory: str | None,
) -> None:
    """Pipeline turn — stage headers print when the stage changes. History is not carried forward."""
    _active_stage[0] = "INIT"  # force header to print on the first event
    orchestrator = PipelineOrchestrator(
        system_messages=[m for m in messages if m.get("role") == "system"],
        working_directory=working_directory,
        regular_tools=tools,
    )
    user_message = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )

    session = AgentSession()
    agent_task = asyncio.create_task(orchestrator.run(session, user_message, messages))

    while True:
        event = await session.outbound.get()

        if event["type"] == "tool_confirm":
            stage = event.get("_pipeline_stage")
            tag = f"[{stage}] " if stage is not None else ""
            print(f"\n{YELLOW}{tag}[confirm] {event.get('tool_name', '')}\n{event.get('preview', '')}{RESET}")
            answer = _read_line("Approve? [y/n]: ").lower()
            reason = None
            if answer != "y":
                reason = _read_line("Reason (optional): ") or None
            session.resolve_confirm(event["tool_id"], answer == "y", reason)
        else:
            _print_event(event)

        if event["type"] in ("done", "error"):
            break

    await agent_task


async def main() -> None:
    import os
    print("Agent REPL  —  type 'exit' to quit")
    await backend.ensure_running()
    working_directory = os.path.realpath(os.getcwd())

    raw_mode = input("Mode? [c]lassic / [p]ipeline (default: classic): ").strip().lower()
    use_pipeline = raw_mode.startswith("p")
    mode_label = "pipeline" if use_pipeline else "classic"
    print(f"Running in {mode_label} mode. Workspace: {working_directory}\n")

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
            if use_pipeline:
                await _run_pipeline_turn(messages, tools, working_directory)
            else:
                await _run_turn(messages, tools, working_directory)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[aborted]{RESET}")
            messages.pop()


if __name__ == "__main__":
    asyncio.run(main())
