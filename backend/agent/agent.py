import json
import asyncio
import aiohttp
from typing import Any

from .tools_definition import TOOLS
from .system_prompt import SYSTEM_PROMPT
from .tool_implementation import execute_tool

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"


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


async def chat_with_tools(messages: list[dict[str, Any]], session: AgentSession) -> bool:
    """
    One iteration of the Ollama call + tool execution loop.
    Returns True when the agent is done (no tool calls were made).
    """
    async with aiohttp.ClientSession() as http:
        async with http.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": [{"type": "function", "function": t} for t in TOOLS],
                "stream": True,
                "options": {"temperature": 0.3},
            },
        ) as response:
            message: dict[str, Any] = {"role": "assistant", "content": "", "thinking": ""}
            has_tool_calls = False

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
                            message["thinking"] = message.get("thinking", "") + thinking
                            await session.emit({"type": "thinking", "content": thinking})
                        if content:
                            message["content"] += content
                            await session.emit({"type": "content", "content": content})

                        if "tool_calls" in resp_msg:
                            if message["content"]:
                                messages.append(message)
                                message = {"role": "assistant", "content": "", "thinking": ""}

                            for tool_call in resp_msg["tool_calls"]:
                                tool_name: str = tool_call.get("function", {}).get("name", "")
                                tool_args: dict = tool_call.get("function", {}).get("arguments", {})
                                call_id: str = tool_call.get("id", f"tc-{id(tool_call)}")

                                await session.emit({
                                    "type": "tool_call",
                                    "tool_id": call_id,
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                })

                                tool_output = await execute_tool(tool_name, call_id, tool_args, session)

                                await session.emit({
                                    "type": "tool_result",
                                    "tool_id": call_id,
                                    "tool_name": tool_name,
                                    "content": tool_output,
                                })

                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": tool_output,
                                })
                            has_tool_calls = True

                    if chunk_json.get("done"):
                        await session.emit({
                            "type": "iteration_end",
                            "prompt_tokens": chunk_json.get("prompt_eval_count", 0),
                            "response_tokens": chunk_json.get("eval_count", 0),
                        })

                except Exception as e:
                    await session.emit({"type": "error", "message": str(e)})

            if message["content"] and (not messages or messages[-1]["role"] != "assistant"):
                messages.append(message)

    return not has_tool_calls


async def run_agent(session: AgentSession, messages: list[dict[str, Any]]) -> None:
    """Run the full agent loop until done, emitting events via session."""
    try:
        finished = False
        while not finished:
            finished = await chat_with_tools(messages, session)
        await session.emit({"type": "done"})
    except asyncio.CancelledError:
        await session.emit({"type": "error", "message": "Agent was aborted"})
    except Exception as e:
        await session.emit({"type": "error", "message": str(e)})
