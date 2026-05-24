"""
Measure the actual token overhead breakdown for a simple message.

Run from the backend/ directory:
    python -m agent.count_context_overhead
"""

import asyncio
import aiohttp
from .agent import MODEL_NAME, OLLAMA_CHAT_URL
from .tools import TOOL_REGISTRY, get_ollama_tool_list

HELLO = [{"role": "user", "content": "hello"}]


async def _call(messages: list, tools: list = [], system: str | None = None) -> int:
    if system is not None:
        messages = [{"role": "system", "content": system}] + messages
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 1},
            },
        ) as resp:
            data = await resp.json()
            if "prompt_eval_count" not in data:
                raise RuntimeError(f"unexpected response: {data}")
            return data["prompt_eval_count"]


async def main():
    print(f"Model: {MODEL_NAME}\n")

    all_tools = get_ollama_tool_list(list(TOOL_REGISTRY.keys()))

    baseline          = await _call([{"role": "user", "content": "."}])
    hello_only        = await _call(HELLO)
    hello_tools       = await _call(HELLO, tools=all_tools)

    with open("../chat_db.sqlite", "rb"):
        pass  # just checking it exists; system prompt loaded separately below

    print(f"baseline (user='.', no tools, no system): {baseline}")
    print(f"hello only (no tools, no system):         {hello_only}  (+{hello_only - baseline} vs baseline)")
    print(f"hello + all tools (no system):            {hello_tools} (+{hello_tools - hello_only} for tools)")
    print()
    print("To include your system prompt, pass its content as --system or add it manually below.")


if __name__ == "__main__":
    asyncio.run(main())
