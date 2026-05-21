"""
One-time script to measure the token cost of each tool individually.

Run from the backend/ directory:
    python -m agent.count_tool_tokens

Methodology:
  - baseline: call Ollama with no tools, empty messages → prompt_eval_count
  - per tool: call with that single tool, empty messages → prompt_eval_count
  - delta = per_tool_count - baseline
"""

import asyncio
import aiohttp
from .agent import MODEL_NAME, OLLAMA_CHAT_URL
from .tools import TOOL_REGISTRY


DUMMY_MESSAGES = [{"role": "user", "content": "."}]


async def _call(tools: list[dict]) -> int:
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL_NAME,
                "messages": DUMMY_MESSAGES,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 1},
            },
        ) as resp:
            data = await resp.json()
            if "prompt_eval_count" not in data:
                raise RuntimeError(f"unexpected response: {data}")
            return data["prompt_eval_count"]


MINIMAL_TOOL = [{"type": "function", "function": {"name": "x", "description": "", "parameters": {}}}]


async def _call_with_system(system: str) -> int:
    messages = [{"role": "system", "content": system}] + DUMMY_MESSAGES
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": [],
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

    baseline = await _call([])
    print(f"baseline (no tools, no system): {baseline} tokens")

    minimal_tool = await _call(MINIMAL_TOOL)
    system_x = await _call_with_system("x")
    print(f"minimal tool (name='x', empty desc, no params): {minimal_tool} tokens (+{minimal_tool - baseline} from baseline)")
    print(f"system prompt 'x' (no tools): {system_x} tokens (+{system_x - baseline} from baseline)")
    print(f"tool framework overhead vs equivalent system text: {minimal_tool - system_x} tokens")

    minimal_tool_x2 = await _call(MINIMAL_TOOL * 2)
    minimal_tool_x3 = await _call(MINIMAL_TOOL * 3)
    print(f"2 minimal tools: {minimal_tool_x2} tokens (+{minimal_tool_x2 - baseline} from baseline)")
    print(f"3 minimal tools: {minimal_tool_x3} tokens (+{minimal_tool_x3 - baseline} from baseline)")
    print(f"delta 1→2 tools: {minimal_tool_x2 - minimal_tool} tokens")
    print(f"delta 2→3 tools: {minimal_tool_x3 - minimal_tool_x2} tokens\n")

    results = {}
    for name, tool in TOOL_REGISTRY.items():
        schema = {"type": "function", "function": tool.to_ollama_schema()}
        count = await _call([schema])
        delta = count - baseline
        results[name] = delta
        print(f"  {name}: {count} total, +{delta} delta")

    print("\n--- verification: measured_delta vs computed token_count ---")
    for name, tool in TOOL_REGISTRY.items():
        from agent.tools.base import TOOL_FRAMEWORK_OVERHEAD
        measured = results[name]
        computed = tool.token_count
        expected = measured - TOOL_FRAMEWORK_OVERHEAD
        match = "OK" if computed == expected else f"MISMATCH (got {computed}, expected {expected})"
        print(f"  {name}: measured_delta={tool.measured_delta}, token_count={computed} [{match}]")


if __name__ == "__main__":
    asyncio.run(main())
