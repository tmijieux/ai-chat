import asyncio
import json
import logging
import os
import subprocess
from typing import AsyncIterator

import aiohttp
from fastapi import HTTPException

from tokenizer import count_tokens, warmup as warmup_tokenizer
from .base import (
    LLMBackend, StreamEvent,
    ContentEvent, ThinkingEvent, ToolCallStartEvent, ToolCallArgEvent, DoneEvent,
    ThinkingParser,
)

MODEL_NAME = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
CTX_LIMIT = 2**14

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):

    async def ensure_running(self) -> None:
        await asyncio.to_thread(warmup_tokenizer)

        async with aiohttp.ClientSession() as http:
            try:
                async with http.get(f"{OLLAMA_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        logger.info("Ollama already running.")
                        return
            except Exception as e:
                logger.exception("ollama detection", exc_info=e)

        logger.info("Ollama not detected — launching 'ollama serve' ...")
        env = os.environ.copy()
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**env, "OLLAMA_CONTEXT_LENGTH": "16384", "OLLAMA_KEEP_ALIVE": "60m"},
            start_new_session=True,
        )

        async with aiohttp.ClientSession() as http:
            for _ in range(60):
                await asyncio.sleep(0.5)
                try:
                    async with http.get(f"{OLLAMA_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=1)) as r:
                        if r.status == 200:
                            logger.info("Ollama started successfully.")
                            return
                except Exception:
                    pass

        logger.warning("Ollama did not respond within 30s — continuing without it.")

    async def check_or_raise(self) -> None:
        async with aiohttp.ClientSession() as http:
            try:
                async with http.get(f"{OLLAMA_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
        raise HTTPException(status_code=503, detail="Ollama is not running")

    async def count_tokens(self, messages: list, tools: list) -> int:
        return count_tokens(messages, tools)

    def prepare_messages(self, messages: list) -> list:
        result = []
        for m in messages:
            msg: dict = {"role": m["role"], "content": m.get("content", "")}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            result.append(msg)
        return result

    async def stream_completion(
        self,
        messages: list,
        tools: list,
        temperature: float,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        tool_choice: dict | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        options: dict = {"temperature": temperature, "num_ctx": CTX_LIMIT}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        parser = ThinkingParser()

        async with aiohttp.ClientSession() as http:
            async with http.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "tools": tools,
                    "stream": True,
                    "options": options,
                },
            ) as response:
                finish_reason = "stop"
                prompt_tokens = 0
                completion_tokens = 0

                async for raw_chunk in response.content.iter_chunks():
                    text = (raw_chunk[0] if isinstance(raw_chunk, tuple) else raw_chunk).decode().strip()
                    if text.startswith("data:"):
                        text = text[5:].strip()
                    if not text:
                        continue
                    try:
                        chunk = json.loads(text)
                    except json.JSONDecodeError:
                        continue

                    if "message" in chunk:
                        msg = chunk["message"]
                        content_frag = msg.get("content") or ""
                        if content_frag:
                            thinking, content = parser.feed(content_frag)
                            if thinking:
                                yield ThinkingEvent(type="thinking", content=thinking)
                            if content:
                                yield ContentEvent(type="content", content=content)

                        # Ollama delivers complete tool calls (not fragmented)
                        for i, tc in enumerate(msg.get("tool_calls") or []):
                            fn = tc.get("function", {})
                            tc_id = tc.get("id", f"tc-{i}")
                            name = fn.get("name", "")
                            args = fn.get("arguments", {})
                            yield ToolCallStartEvent(type="tool_call_start", index=i, id=tc_id, name=name)
                            yield ToolCallArgEvent(type="tool_call_arg", index=i, fragment=json.dumps(args))

                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)
                        finish_reason = chunk.get("done_reason", "stop")

                t, c = parser.flush()
                if t:
                    yield ThinkingEvent(type="thinking", content=t)
                if c:
                    yield ContentEvent(type="content", content=c)

                yield DoneEvent(
                    type="done",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finish_reason=finish_reason,
                )
