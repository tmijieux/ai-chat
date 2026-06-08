import json
import asyncio
import logging
import aiohttp
from typing import Any, Callable, Awaitable

from .tools import TOOL_REGISTRY, get_ollama_tool_list
from llm import backend
from llm.base import ToolCallStartEvent, ToolCallArgEvent

CTX_LIMIT = 2**15

logger = logging.getLogger(__name__)


class AgentSession:
    """Manages bidirectional communication between the agent loop and the WebSocket client."""

    def __init__(self):
        self.outbound: asyncio.Queue[dict] = asyncio.Queue()
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._compression_event: asyncio.Event = asyncio.Event()
        self._compression_conv_id: str | None = None
        self.refresh_messages_callback: Callable[[str], Awaitable[list[dict]]] | None = None
        self.finish_result: dict | None = None

    async def emit(self, event: dict) -> None:
        await self.outbound.put(event)

    async def request_confirm(
        self, tool_id: str, tool_name: str, arguments: dict, preview: str,
        diff_lines: list | None = None,
    ) -> tuple[bool, str | None]:
        """Emit a confirmation request and suspend until the client responds."""
        event: dict = {
            "type": "tool_confirm",
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "preview": preview,
        }
        if diff_lines is not None:
            event["diff_lines"] = diff_lines
        await self.emit(event)
        future: asyncio.Future[tuple[bool, str | None]] = asyncio.get_running_loop().create_future()
        self._pending_confirms[tool_id] = future
        return await future

    def resolve_confirm(self, tool_id: str, approved: bool, reason: str | None = None) -> None:
        future = self._pending_confirms.pop(tool_id, None)
        if future and not future.done():
            future.set_result((approved, reason))

    async def await_compression(self) -> str | None:
        """Suspend the agent loop until the frontend sends compression_done."""
        self._compression_event.clear()
        await self._compression_event.wait()
        return self._compression_conv_id

    def resume_after_compression(self, conv_id: str) -> None:
        self._compression_conv_id = conv_id
        self._compression_event.set()


def _find_superseded_read_file_indices(pairs: list[tuple[str, str]]) -> list[int]:
    """Given (role, content_json) pairs, return indices of superseded read_file results."""
    path_indices: dict[str, list[int]] = {}
    for i, (role, content_str) in enumerate(pairs):
        if role != "tool":
            continue
        try:
            content = json.loads(content_str or "")
        except (json.JSONDecodeError, ValueError):
            continue
        if content.get("tool") != "read_file" or content.get("status") != "success":
            continue
        path_indices.setdefault(content.get("path", ""), []).append(i)
    return [i for indices in path_indices.values() for i in indices[:-1]]


def _deduplicate_file_reads(messages: list[dict[str, Any]]) -> None:
    pairs = [(m.get("role", ""), m.get("content", "")) for m in messages]
    for i in _find_superseded_read_file_indices(pairs):
        try:
            path = json.loads(messages[i].get("content", "")).get("path", "")
        except (json.JSONDecodeError, ValueError):
            path = ""
        try:
            tool_call_id = json.loads(messages[i].get("content", "")).get("tool_call_id")
        except (json.JSONDecodeError, ValueError):
            tool_call_id = None
        messages[i]["content"] = json.dumps({
            "tool": "read_file",
            "status": "evicted",
            "path": path,
            "tool_call_id": tool_call_id,
            "reason": "file content removed — analysis was expressed in conversation above, superseded by later read",
        })




def _log_context(messages: list[dict[str, Any]]) -> None:
    print(f"\n=== CONTEXT ({len(messages)} messages) ===")
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if content is None:
            content = ""
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            img_count = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
            content = " ".join(text_parts) + (f" [+{img_count} image(s)]" if img_count else "")
        if role == "system":
            print(f"  [system] {content[:120].replace('\n', ' ')}")
        elif role == "user":
            print(f"  [user] {content[:200].replace('\n', ' ')}")
        elif role == "tool":
            try:
                j = json.loads(content)
                tool = j.get("tool", "?")
                status = j.get("status", "?")
                path = j.get("path", "")
                if tool == "read_file":
                    suffix = " [evicted]" if status == "evicted" else ""
                    print(f"  [tool] FILE {path}{suffix}")
                elif tool == "list_directory":
                    print(f"  [tool] DIRECTORY {path}")
                elif tool == "glob_files":
                    pattern = j.get("pattern", "")
                    print(f"  [tool] GLOB {pattern} in {path}")
                elif tool == "grep_files":
                    pattern = j.get("pattern", "")
                    glob_pat = j.get("glob_pattern", "")
                    suffix = f" [{glob_pat}]" if glob_pat else ""
                    print(f"  [tool] GREP '{pattern}' in {path}{suffix}")
                else:
                    print(f"  [tool] {tool}: {status}")
            except (json.JSONDecodeError, ValueError):
                print(f"  [tool] {content[:80].replace('\n', ' ')}")
        elif role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls is not None and len(tool_calls) > 0:
                print(f"  [thinking] {content.replace('\n', ' ')[:120]}")
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
                print(f"  [assistant] {len(tool_calls)} tool call(s): {names}")
            elif content.strip() != "":
                print(f"  [assistant] {content.replace('\n', ' ')[:120]}")
        else:
            print(f"  [{role}] {content[:80].replace('\n', ' ')}")
    print("=" * 40)


async def chat_with_tools(
    messages: list[dict[str, Any]],
    session: AgentSession,
    tools: list[dict],
    working_directory: str | None,
    extra_tools: dict | None = None,
) -> bool:
    """
    One iteration of the LLM call + tool execution loop.
    Returns True when the agent is done (no tool calls were made).
    """
    prepared = backend.prepare_messages(messages)
    _log_context(prepared)

    ctx_before_generation = await backend.count_tokens(prepared, tools)
    max_tokens = CTX_LIMIT - ctx_before_generation
    print(f"[tokens] context before generation: {ctx_before_generation}/{CTX_LIMIT}, max_tokens={max_tokens}")
    await session.emit({"type": "ctx_update", "ctx_tokens": ctx_before_generation})

    message: dict[str, Any] = {"role": "assistant", "content": "", "thinking": ""}
    tool_calls_acc: dict[int, dict] = {}  # index → {id, name, arguments_str}
    prompt_eval_count: int = ctx_before_generation
    eval_count: int = 0
    done_reason: str = ""

    async for event in backend.stream_completion(prepared, tools, temperature=0.3, max_tokens=max_tokens):
        etype = event["type"]

        if etype == "thinking":
            message["thinking"] += event["content"]
            await session.emit({"type": "thinking", "content": event["content"]})

        elif etype == "content":
            message["content"] += event["content"]
            await session.emit({"type": "content", "content": event["content"]})

        elif etype == "tool_call_start":
            idx = event["index"]
            tool_calls_acc[idx] = {"id": event["id"], "name": event["name"], "arguments_str": ""}
            await session.emit({"type": "tool_call_start", "tool_id": event["id"], "tool_name": event["name"]})

        elif etype == "tool_call_arg":
            idx = event["index"]
            if idx in tool_calls_acc:
                tool_calls_acc[idx]["arguments_str"] += event["fragment"]
            await session.emit({"type": "tool_call_chunk", "tool_id": tool_calls_acc.get(idx, {}).get("id", ""), "chunk": event["fragment"]})

        elif etype == "done":
            prompt_eval_count = ctx_before_generation
            eval_count = event["completion_tokens"]
            done_reason = event["finish_reason"]
            print(f"[tokens] prompt_tokens={prompt_eval_count} completion_tokens={eval_count} finish_reason={done_reason}")

    # Build tool_calls list from accumulated fragments
    tool_calls: list[dict] = [
        {
            "id": acc["id"],
            "function": {
                "name": acc["name"],
                "arguments": json.loads(acc["arguments_str"]) if acc["arguments_str"] else {},
            },
        }
        for acc in (tool_calls_acc[i] for i in sorted(tool_calls_acc))
    ]

    if done_reason == "length":
        await session.emit({"type": "error", "message": f"Context limit reached during generation: {prompt_eval_count + eval_count}/{CTX_LIMIT} tokens. The response was cut off."})
        return True, False

    finished_without_response = len(message["content"]) == 0 and len(tool_calls) == 0
    if finished_without_response:
        logger.warning("Agent finished without response: no content, no tool calls")

    if len(message["content"]) > 0:
        messages[:] = [m for m in messages if not m.get("_transient")]
        if len(tool_calls) > 0:
            message["tool_calls"] = tool_calls
        messages.append(message)
    elif len(message["thinking"]) > 0 or len(tool_calls) > 0:
        messages.append({
            "role": "assistant", 
            "content": f"<think>{message['thinking']}</think>", 
            "tool_calls": tool_calls
        })
        # old code (transient hack):
        # messages.append({"role": "assistant", , "_transient": True})


    if len(tool_calls) > 0:
        ctx_before = await backend.count_tokens(backend.prepare_messages(messages), tools)
        print(f"[tokens] context before tool execution: {ctx_before}/{CTX_LIMIT}")
        await session.emit({"type": "ctx_update", "ctx_tokens": ctx_before})
        for tool_call in tool_calls:
            tool_name: str = tool_call.get("function", {}).get("name", "")
            tool_args: dict = tool_call.get("function", {}).get("arguments", {})
            call_id: str = tool_call.get("id", f"tc-{id(tool_call)}")

            await session.emit({"type": "tool_call", "tool_id": call_id, "tool_name": tool_name, "arguments": tool_args})

            effective_registry = {**TOOL_REGISTRY, **(extra_tools or {})}
            if tool_name not in effective_registry:
                result_dict = {"tool": tool_name, "status": "error", "error": {"message": f"Unknown tool: {tool_name}"}}
                log_msg = None
            else:
                tool_instance = effective_registry[tool_name]
                result_dict = await tool_instance.execute(tool_args, session, working_directory)
                log_msg = tool_instance.label(tool_args)

            result_dict["tool_call_id"] = call_id
            tool_output = json.dumps(result_dict)
            messages.append({"role": "tool", "name": tool_name, "content": tool_output})

            ctx_after = await backend.count_tokens(backend.prepare_messages(messages), tools)
            print(f"[tokens] context after tool result '{tool_name}': {ctx_after}/{CTX_LIMIT}")
            await session.emit({"type": "ctx_update", "ctx_tokens": ctx_after})
            if ctx_after > CTX_LIMIT:
                # Emit tool_result first so frontend saves it to DB before compressing
                await session.emit({"type": "tool_result", "tool_id": call_id, "tool_name": tool_name, "content": tool_output, "log_message": log_msg})
                await session.emit({"type": "compressing", "ctx_tokens": ctx_after, "ctx_limit": CTX_LIMIT})
                conv_id = await session.await_compression()
                if conv_id is not None and session.refresh_messages_callback is not None:
                    refreshed = await session.refresh_messages_callback(conv_id)
                    messages[:] = refreshed
                ctx_after_compress = await backend.count_tokens(backend.prepare_messages(messages), tools)
                print(f"[tokens] context after compression: {ctx_after_compress}/{CTX_LIMIT}")
                await session.emit({"type": "ctx_update", "ctx_tokens": ctx_after_compress})
                if ctx_after_compress > CTX_LIMIT:
                    await session.emit({"type": "error", "message": f"Context still exceeds limit after compression: {ctx_after_compress}/{CTX_LIMIT} tokens"})
                    return True, False
                continue

            await session.emit({"type": "tool_result", "tool_id": call_id, "tool_name": tool_name, "content": tool_output, "log_message": log_msg})

    # Emit iteration_end after tool results so the frontend receives tool_result events
    # before iteration_end. The frontend rotation logic patches tool results from iteration N
    # with prompt_tokens from iteration N+1 — this ordering makes that work correctly.
    await session.emit({
        "type": "iteration_end",
        "prompt_tokens": prompt_eval_count,
        "response_tokens": eval_count,
    })

    _deduplicate_file_reads(messages)
    return len(tool_calls) == 0, finished_without_response


async def run_agent(
    session: AgentSession,
    messages: list[dict[str, Any]],
    tools: list[dict],
    working_directory: str | None,
) -> None:
    """Run the full agent loop until done, emitting events via session."""
    try:
        finished = False
        finished_without_response = False
        while not finished:
            finished, finished_without_response = await chat_with_tools(messages, session, tools, working_directory)
        await session.emit({"type": "done", "finished_without_response": finished_without_response})
    except asyncio.CancelledError:
        await session.emit({"type": "error", "message": "Agent was aborted"})
    except aiohttp.ClientConnectorError as e:
        logger.error("LLM backend connection error: %s", e)
        await session.emit({"type": "error", "message": "LLM backend is not running"})
    except Exception as e:
        logger.exception("Unexpected error in agent loop")
        await session.emit({"type": "error", "message": str(e)})
