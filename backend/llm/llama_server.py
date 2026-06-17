import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import AsyncIterator, Sequence

import aiohttp
from fastapi import HTTPException

from tokenizer import render_messages
from message_types import LLMMessage
from .base import (
    LLMBackend, StreamEvent,
    ContentEvent, ThinkingEvent, ToolCallStartEvent, ToolCallArgEvent, DoneEvent,
)

MODEL_NAME = "local"
LLAMA_BASE_URL = "http://127.0.0.1:8080"
LLAMA_CHAT_URL = f"{LLAMA_BASE_URL}/v1/chat/completions"
LLAMA_TOKENIZE_URL = f"{LLAMA_BASE_URL}/tokenize"
LLAMA_HEALTH_URL = f"{LLAMA_BASE_URL}/health"
LLAMA_SERVER_EXE = str(Path.home() / "ai/llama.cpp/build/bin/Release/llama-server.exe")
GGUF_PATH = str(Path.home() / "ai/models/unsloth/Qwen3.5-9B-UD-Q3_K_XL.gguf")
MMPROJ_PATH = str(Path.home() / "ai/models/unsloth/mmproj-F16.gguf")
CTX_LIMIT = 2**15 # 14 -> 16K, 15 -> 32K, 16 -> 65k

logger = logging.getLogger(__name__)


class LlamaServerBackend(LLMBackend):

    async def ensure_running(self) -> None:
        async with aiohttp.ClientSession() as http:
            try:
                async with http.get(LLAMA_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        logger.info("llama-server already running.")
                        return
            except Exception:
                pass

        logger.info("llama-server not detected — launching ...")
        p = subprocess.Popen(
            [
                LLAMA_SERVER_EXE,
                "-m", GGUF_PATH,
                "--mmproj", MMPROJ_PATH,
                "-c", str(CTX_LIMIT),
                "-ngl", "99",
                "--port", "8080",
                "--host", "127.0.0.1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        

        async with aiohttp.ClientSession() as http:
            for _ in range(120):  # 60s — model loading takes time
                ret = p.poll() 
                if ret is not None:
                    output = p.stdout.read().decode(errors="replace") if p.stdout else ""
                    logger.error("llama-server exited with code %s:\n%s", ret, output)
                    break
                await asyncio.sleep(0.5)
                try:
                    async with http.get(LLAMA_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=1)) as r:
                        if r.status == 200:
                            logger.info("llama-server started successfully.")
                            return
                except Exception:
                    pass

        logger.warning("llama-server did not respond within 60s — continuing anyway.")

    async def check_or_raise(self) -> None:
        async with aiohttp.ClientSession() as http:
            try:
                async with http.get(LLAMA_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
        raise HTTPException(status_code=503, detail="llama-server is not running")

    async def count_tokens(self, messages: Sequence[LLMMessage], tools: list) -> int:
        rendered = render_messages(messages, tools)
        return await self.count_text_tokens(rendered)

    async def count_text_tokens(self, text: str) -> int:
        """Count tokens for raw text, bypassing the chat template."""
        async with aiohttp.ClientSession() as http:
            async with http.post(
                LLAMA_TOKENIZE_URL,
                json={"content": text},
            ) as r:
                data = await r.json()
                return len(data["tokens"])

    def prepare_messages(self, messages: Sequence[LLMMessage]) -> Sequence[LLMMessage]:
        """Convert internal format to OpenAI wire format for llama-server."""
        result = []
        for m in messages:
            msg: dict = {"role": m["role"], "content": m.get("content", "")}

            if "tool_calls" in m:
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"tc-{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            # arguments must be a JSON string in OpenAI format
                            "arguments": (
                                json.dumps(tc["function"]["arguments"])
                                if isinstance(tc["function"]["arguments"], dict)
                                else tc["function"]["arguments"]
                            ),
                        },
                    }
                    for i, tc in enumerate(m["tool_calls"])
                ]

            if m["role"] == "tool":
                # OpenAI requires tool_call_id as a top-level field on tool messages
                try:
                    raw = m.get("content")
                    if not isinstance(raw, str):
                        raise ValueError(f"tool message has non-string content: {type(raw)}")
                    content_data = json.loads(raw)
                    tool_call_id = content_data.get("tool_call_id")
                    if tool_call_id:
                        msg["tool_call_id"] = tool_call_id
                except (json.JSONDecodeError, ValueError):
                    pass

            result.append(msg)
        return result

    async def stream_completion(
        self,
        messages: Sequence[LLMMessage],
        tools: list,
        temperature: float,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        tool_choice: dict | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        body: dict = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        tool_calls_acc: dict[int, dict] = {}  # index → {id, name, arguments_str}
        finish_reason: str = "stop"
        prompt_tokens: int = 0
        completion_tokens: int = 0

        async with aiohttp.ClientSession() as http:
            async with http.post(LLAMA_CHAT_URL, json=body) as response:
                if response.status != 200:
                    body_text = await response.text()
                    logger.error("llama-server error %s: %s", response.status, body_text)
                    return

                async for line_bytes in response.content:
                    line = line_bytes.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                        timings = chunk.get("timings") or {}
                        prompt_tokens = timings.get("prompt_n", 0)
                        completion_tokens = timings.get("predicted_n", 0)

                    # Thinking — llama-server exposes it in reasoning_content, not <think> tags
                    thinking_frag = delta.get("reasoning_content") or ""
                    if thinking_frag:
                        yield ThinkingEvent(type="thinking", content=thinking_frag)

                    # Content
                    content_frag = delta.get("content") or ""
                    if content_frag:
                        yield ContentEvent(type="content", content=content_frag)

                    # Tool calls — fragmented across chunks, keyed by index
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_calls_acc:
                            tc_id = tc_delta.get("id", f"tc-{idx}")
                            name = (tc_delta.get("function") or {}).get("name", "")
                            tool_calls_acc[idx] = {"id": tc_id, "name": name, "arguments_str": ""}
                            yield ToolCallStartEvent(type="tool_call_start", index=idx, id=tc_id, name=name)

                        args_frag = (tc_delta.get("function") or {}).get("arguments") or ""
                        if args_frag:
                            tool_calls_acc[idx]["arguments_str"] += args_frag
                            yield ToolCallArgEvent(type="tool_call_arg", index=idx, fragment=args_frag)

        yield DoneEvent(
            type="done",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
