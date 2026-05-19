import json
import asyncio
import aiohttp
from typing import Any


# Import your existing constants
from .tools_definition import TOOLS
from .system_prompt import SYSTEM_PROMPT
from .tool_implementation import execute_tool

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"

async def chat_with_tools(messages: list[dict[str, Any]], stream: bool = True) -> bool:
    """
    Core agent loop that handles tool calls and responses.
    """
    # Use aiohttp ClientSession for async requests
    from pprint  import pprint

    # pprint(TOOLS)

    # exit(1)
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            OLLAMA_CHAT_URL,
            json = {
                "model": MODEL_NAME,
                "messages": messages,
                "tools": [{"type":"function", "function":t} for t in TOOLS],
                "stream": stream,
                "options": {"temperature": 0.3}
            },
        ) as response:

            message = {"role":"assistant", "content":"", "thinking":""}

            thinking_reported = False
            response_reported = False
            has_tool_calls = False

            async for chunk in response.content.iter_chunks():
                # Decode the chunk
                #print("chunk=",chunk)
                if isinstance(chunk, tuple):
                    text, has_error = chunk
                    text = text.decode().strip()
                else:
                    text = chunk.decode().strip()
                # Remove SSE 'data:' prefix
                if text.startswith("data:"):
                    text = text[5:].strip()

                if not text:
                    continue

                try:
                    chunk_json = json.loads(text)

                    # Update message content if in stream
                    if "message" in chunk_json:
                        response_message = chunk_json["message"]

                        thinking = response_message.get("thinking", "")
                        content = response_message.get("content", "")
                        message["content"] += content
                        #message["thinking"] += thinking
                        if len(thinking) > 0:
                            if not thinking_reported:
                                thinking_reported  = True
                                print("\n === THINKING: ===")
                            print(thinking, end="", flush=True)
                        if len(content) > 0:
                            if not response_reported:
                                response_reported  = True
                                print("\n === RESPONSE: ===")
                            print(content, end="", flush=True)

                        # if len(thinking) > 0:
                        #     print(thinking, end="", flush=True)

                        # Handle tool calls
                        if "tool_calls" in response_message:
                            #if len(message["content"]) > 0 or len(message["thinking"]) > 0:
                            if len(message["content"]) > 0:
                                messages.append(message)
                                message = {"role":"assistant","content":"","thinking":""}

                            print("\n\n\n=== TOOL_CALLS=", response_message["tool_calls"]," ===\n\n", flush=True)
                            for tool_call in response_message["tool_calls"]:
                                tool_name: str = tool_call.get("function", {}).get("name", "")
                                tool_args: dict = tool_call.get("function", {}).get("arguments", {})
                                call_id: str = tool_call.get("id", "unknown")

                                # Execute tool
                                tool_output = execute_tool(tool_name, call_id, tool_args)
                                print("tool_output= ", len(tool_output), "characters")
                                print("tool_output= ", tool_output,"\n\n\n")

                                # Add tool output to messages
                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": tool_output
                                })
                            has_tool_calls = True

                    # Print in real-time
                    # print(text, end="", flush=True)
                    # print("\n")  # Newline after each chunk

                    if "done" in chunk_json and chunk_json["done"]:
                        print("\n=== GENERATING DONE ===")
                        #print(chunk_json)
                        prompt_token = chunk_json["prompt_eval_count"]
                        response_token = chunk_json["eval_count"]
                        context_usage_pct = int((100*(prompt_token / 16384))  * 100) / 100
                        print(f"\n === TOKEN IN PROMPT: {prompt_token} | TOKEN IN RESPONSE(think+response): {response_token} CONTEXT USAGE = {context_usage_pct}%")


                except json.JSONDecodeError:
                    # Ignore non-JSON chunks
                    continue
                except Exception as e:
                    print(f"\n[Error processing chunk]: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if len(message["content"]) > 0 and messages[-1]["role"] != "assistant":
                messages.append(message)

            # If streaming ended without tools being called, return
            # But if tool_call is found in the stream, we need to execute and loop.
            # This requires parsing tool_calls from the last response.
            print("")  # Newline after stream

    return not has_tool_calls




async def main():
    print("=== Agent Starting ===")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    while True:
        user_prompt = input(">>> ")
        user_prompt = user_prompt.strip()

        if len(user_prompt) == 0:
            continue
        user_message = {"role": "user", "content": user_prompt}
        messages.append(user_message)

        finished = False
        while not finished:
            finished = await chat_with_tools(messages)


async def count_token(messages: list[dict[str,Any]], use_tools: bool = True) -> int:
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
    asyncio.run(main())
    #asyncio.run(count_token_in_system_prompt())
    #asyncio.run(count_token_in_file("claude/scratch.txt"))