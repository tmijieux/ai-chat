import asyncio
from typing import Any

import aiohttp

from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from tokenizer import count_tokens


async def count_token_with_ollama(messages: list[dict[str, Any]], tool_names: list[str] | None = None) -> int:
    """Generate 1 single token and read prompt_eval_count returned by Ollama."""
    if tool_names is None:
        tool_names = list(TOOL_REGISTRY.keys())
    tools = get_ollama_tool_list(tool_names)

    print("=== COUNTING TOKEN ===")
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
        ) as response:
            content = await response.json()
            print("content=", content)
            return content["prompt_eval_count"]


def count_token_local(messages: list[dict[str, Any]], tool_names: list[str] | None = None) -> int:
    """Count tokens locally using tiktoken + GGUF vocab — no Ollama call needed."""
    if tool_names is None:
        tool_names = list(TOOL_REGISTRY.keys())
    tools = get_ollama_tool_list(tool_names)
    return count_tokens(messages, tools)


async def count_token_in_file(file_path: str):
    with open(file_path, "r") as f:
        data = f.read()

        nb_token = 0
        chunk_size = 10000
        if len(data) > chunk_size:
            for i in range(0, len(data), chunk_size):
                data_part = data[i: i + chunk_size]
                nb_token_part = await count_token_with_ollama([{"role": "user", "content": data_part}], tool_names=[])
                nb_token += nb_token_part
        else:
            nb_token = await count_token_with_ollama([{"role": "user", "content": data}], tool_names=[])
        print("TOTAL TOKENS=", nb_token)


if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) <= 1:
        print("usage count_token file1 [file2 [file3]]")
    for arg in args[1:]:
        print(arg)
        asyncio.run(count_token_in_file(arg))
