import json
import asyncio
import logging
import aiohttp
from typing import Any

from .tools import TOOL_REGISTRY, get_ollama_tool_list

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"  # kept local; main.py owns the base URL constant
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
        messages[i]["content"] = json.dumps({
            "tool": "read_file",
            "status": "evicted",
            "path": path,
            "reason": "file content removed — analysis was expressed in conversation above, superseded by later read",
        })




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
    async with aiohttp.ClientSession() as http:
        async with http.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{k: v for k, v in m.items() if k not in ("_transient", "thinking")} for m in messages],
                "tools": tools,
                "stream": True,
                "options": {"temperature": 0.3},
            },
        ) as response:
            message: dict[str, Any] = {"role": "assistant", "content": "", "thinking": ""}
            tool_calls: list[dict] = []

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
                        await session.emit({
                            "type": "iteration_end",
                            "prompt_tokens": chunk_json.get("prompt_eval_count", 0),
                            "response_tokens": chunk_json.get("eval_count", 0),
                        })
                except Exception as e:
                    await session.emit({"type": "error", "message": str(e)})

    if len(message["content"]) == 0 and len(tool_calls) == 0:
        logger.warning("Degenerate response: thinking only, no content, no tool calls")

    if len(message["content"]) > 0:
        messages[:] = [m for m in messages if not m.get("_transient")]
        messages.append(message)
    elif len(message["thinking"]) > 0 and len(tool_calls) > 0:
        #{{\"toolcall\":\"redacted\",\"status\":\"redacted\"}}
        messages.append({"role": "assistant", "content": f"<think>{message['thinking']}</think>", "_transient": True})


    if len(tool_calls) > 0:
        for tool_call in tool_calls:
            tool_name: str = tool_call.get("function", {}).get("name", "")
            tool_args: dict = tool_call.get("function", {}).get("arguments", {})
            call_id: str = tool_call.get("id", f"tc-{id(tool_call)}")

            await session.emit({"type": "tool_call", "tool_id": call_id, "tool_name": tool_name, "arguments": tool_args})

            if tool_name not in TOOL_REGISTRY:
                result_dict = {"tool": tool_name, "status": "error", "error": {"message": f"Unknown tool: {tool_name}"}}
            else:
                result_dict = await TOOL_REGISTRY[tool_name].execute(tool_args, session, working_directory)

            tool_output = json.dumps(result_dict)
            await session.emit({"type": "tool_result", "tool_id": call_id, "tool_name": tool_name, "content": tool_output})
            messages.append({"role": "tool", "name": tool_name, "content": tool_output})

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
    except aiohttp.ClientConnectorError:
        await session.emit({"type": "error", "message": "Ollama is not running — start Ollama and try again"})
    except Exception as e:
        await session.emit({"type": "error", "message": str(e)})
