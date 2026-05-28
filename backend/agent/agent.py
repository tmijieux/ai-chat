import json
import asyncio
import logging
import aiohttp
from typing import Any

from .tools import TOOL_REGISTRY, get_ollama_tool_list
from tokenizer import count_tokens

#CTX_LIMIT = 16384
CTX_LIMIT = 2**15

OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # kept local; main.py owns the base URL constant
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"  # kept local; main.py owns the base URL constant
MODEL_NAME = "qwen3.5:9b"

logger = logging.getLogger(__name__)


class AgentSession:
    """Manages bidirectional communication between the agent loop and the WebSocket client."""

    def __init__(self):
        self.outbound: asyncio.Queue[dict] = asyncio.Queue()
        self._pending_confirms: dict[str, asyncio.Future] = {}

    async def emit(self, event: dict) -> None:
        await self.outbound.put(event)

    async def request_confirm(
        self, tool_id: str, tool_name: str, arguments: dict, preview: str
    ) -> tuple[bool, str | None]:
        """Emit a confirmation request and suspend until the client responds."""
        await self.emit({
            "type": "tool_confirm",
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "preview": preview,
        })
        future: asyncio.Future[tuple[bool, str | None]] = asyncio.get_running_loop().create_future()
        self._pending_confirms[tool_id] = future
        return await future

    def resolve_confirm(self, tool_id: str, approved: bool, reason: str | None = None) -> None:
        future = self._pending_confirms.pop(tool_id, None)
        if future and not future.done():
            future.set_result((approved, reason))


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
) -> bool:
    """
    One iteration of the Ollama call + tool execution loop.
    Returns True when the agent is done (no tool calls were made).
    """
    def _to_ollama_msg(m: dict) -> dict:
        msg: dict = {"role": m["role"], "content": m["content"]}
        if "tool_calls" in m:
            msg["tool_calls"] = m["tool_calls"]
        return msg
    ollama_messages = [_to_ollama_msg(m) for m in messages]
    _log_context(ollama_messages)
    #print(json.dumps(ollama_messages, indent=2, ensure_ascii=False))
    ctx_before_generation = count_tokens(messages, tools)
    num_predict = CTX_LIMIT - ctx_before_generation
    print(f"[tokens] context before generation: {ctx_before_generation}/{CTX_LIMIT}, num_predict={num_predict}")

    async with aiohttp.ClientSession() as http:
        async with http.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL_NAME,
                "messages": ollama_messages,
                "tools": tools,
                "stream": True,
                "options": {"temperature": 0.3, "num_ctx": CTX_LIMIT, "num_predict": num_predict},
            },
        ) as response:
            message: dict[str, Any] = {"role": "assistant", "content": "", "thinking": ""}
            tool_calls: list[dict] = []
            prompt_eval_count: int = 0
            eval_count: int = 0
            done_reason: str = ""

            async for raw_chunk in response.content.iter_chunks():
                text = (raw_chunk[0] if isinstance(raw_chunk, tuple) else raw_chunk).decode().strip()
                if text.startswith("data:"):
                    text = text[5:].strip()
                if not text:
                    continue
                try:
                    chunk_json = json.loads(text)
                except json.JSONDecodeError:
                    continue
                try:
                    if "message" in chunk_json:
                        resp_msg = chunk_json["message"]
                        thinking = resp_msg.get("thinking", "")
                        content = resp_msg.get("content", "")
                        if thinking:
                            message["thinking"] += thinking
                            await session.emit({"type": "thinking", "content": thinking})
                        if content:
                            message["content"] += content
                            await session.emit({"type": "content", "content": content})
                        if "tool_calls" in resp_msg:
                            tool_calls.extend(resp_msg["tool_calls"])
                    if chunk_json.get("done"):
                        prompt_eval_count = chunk_json.get("prompt_eval_count", 0)
                        eval_count = chunk_json.get("eval_count", 0)
                        done_reason = chunk_json.get("done_reason", "")
                        print(f"[tokens] prompt_eval_count={prompt_eval_count} eval_count={eval_count} done_reason={done_reason}")
                except Exception as e:
                    await session.emit({"type": "error", "message": str(e)})

    if done_reason == "length":
        await session.emit({"type": "error", "message": f"Context limit reached during generation: {prompt_eval_count + eval_count}/{CTX_LIMIT} tokens. The response was cut off."})
        return True

    if len(message["content"]) == 0 and len(tool_calls) == 0:
        logger.warning("Degenerate response: thinking only, no content, no tool calls")

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
        ctx_before = count_tokens(messages, tools)
        print(f"[tokens] context before tool execution: {ctx_before}/{CTX_LIMIT}")
        for tool_call in tool_calls:
            tool_name: str = tool_call.get("function", {}).get("name", "")
            tool_args: dict = tool_call.get("function", {}).get("arguments", {})
            call_id: str = tool_call.get("id", f"tc-{id(tool_call)}")

            await session.emit({"type": "tool_call", "tool_id": call_id, "tool_name": tool_name, "arguments": tool_args})

            if tool_name not in TOOL_REGISTRY:
                result_dict = {"tool": tool_name, "status": "error", "error": {"message": f"Unknown tool: {tool_name}"}}
                log_msg = None
            else:
                tool_instance = TOOL_REGISTRY[tool_name]
                result_dict = await tool_instance.execute(tool_args, session, working_directory)
                log_msg = tool_instance.label(tool_args)

            result_dict["tool_call_id"] = call_id
            tool_output = json.dumps(result_dict)
            messages.append({"role": "tool", "name": tool_name, "content": tool_output})

            ctx_after = count_tokens(messages, tools)
            print(f"[tokens] context after tool result '{tool_name}': {ctx_after}/{CTX_LIMIT}")
            if ctx_after > CTX_LIMIT:
                await session.emit({"type": "error", "message": f"Context limit exceeded after tool result from '{tool_name}': {ctx_after}/{CTX_LIMIT} tokens"})
                return True

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
    return len(tool_calls) == 0


async def run_agent(
    session: AgentSession,
    messages: list[dict[str, Any]],
    tools: list[dict],
    working_directory: str | None,
) -> None:
    """Run the full agent loop until done, emitting events via session."""
    try:
        finished = False
        while not finished:
            finished = await chat_with_tools(messages, session, tools, working_directory)
        await session.emit({"type": "done"})
    except asyncio.CancelledError:
        await session.emit({"type": "error", "message": "Agent was aborted"})
    except aiohttp.ClientConnectorError as e:
        logger.error("Ollama connection error: %s", e)
        await session.emit({"type": "error", "message": "Ollama is not running — start Ollama and try again"})
    except Exception as e:
        logger.exception("Unexpected error in agent loop")
        await session.emit({"type": "error", "message": str(e)})
