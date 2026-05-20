import asyncio
from typing import Any

import aiohttp

from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL
from agent.system_prompt import SYSTEM_PROMPT
from agent.tools_definition import TOOLS


async def count_token(messages: list[dict[str,Any]], use_tools: bool = True) -> int:
    """ Generate 1 single token and look at prompt_eval_count returned by ollama """
    
    print("=== COUNTING TOKEN ===")
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            OLLAMA_CHAT_URL,
            json = {
                "model": MODEL_NAME,
                "messages": messages,
                "tools": [{"type":"function", "function":t} for t in TOOLS] if use_tools else [],
                "stream": False,
                "options": {"temperature": 0, "num_predict":1}
            },
        ) as response:
            content = await response.json()
            print("content=",content)

            return content["prompt_eval_count"]

async def count_token_in_system_prompt():
    nb_token = await count_token([{"role": "system", "content": SYSTEM_PROMPT}], use_tools=True)
    print("NB TOKEN=", nb_token)

async def count_token_in_file(file_path: str):
    with open(file_path, "r") as f:
        data = f.read()

        nb_token = 0
        chunk_size = 10000
        if len(data) > chunk_size:
            for i in range(0, len(data), chunk_size):
                data_part = data[i: i +chunk_size]
                nb_token_part = await count_token([{"role": "user", "content": data_part}], use_tools=False)

                nb_token += nb_token_part
        else:
            nb_token = await count_token([{"role": "user", "content": data}], use_tools=False)
        print("TOTAL TOKENS=", nb_token)

if __name__ == "__main__":
    #asyncio.run(count_token_in_system_prompt())
    
    import sys
    args = sys.argv
    if len(args) <= 1:
        print("usage count_token file1 [file2 [file3]]")
    for arg in args[1:]:
        print(arg)
        asyncio.run(count_token_in_file(arg))