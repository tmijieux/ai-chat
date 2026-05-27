"""Demo script: render a simulated conversation and compare local vs Ollama token counts."""
import asyncio
from agent.tools import get_ollama_tool_list
from agent.count_token import count_token_with_ollama
from tokenizer import render_messages, count_tokens

tool_names = ["glob_files"]
tools = get_ollama_tool_list(tool_names)

messages = [
    {
        "role": "system",
        "content": "You are a coding assistant. You have access to tools to browse the filesystem.",
    },
    {
        "role": "user",
        "content": "Find all Python files in the backend directory.",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "glob_files",
                    "arguments": {"pattern": "**/*.py", "path": "backend"},
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"tool": "glob_files", "status": "success", "files": ["backend/main.py", "backend/database.py", "backend/tables.py", "backend/agent/agent.py"], "file_count": 4}',
    },
    {
        "role": "assistant",
        "content": "I found 4 Python files in the backend directory: main.py, database.py, tables.py, and agent/agent.py.",
    },
]

rendered = render_messages(messages, tools, add_generation_prompt=False)

print("=" * 60)
print("RENDERED TEMPLATE:")
print("=" * 60)
print(rendered)
print("=" * 60)

local = count_tokens(messages, tools)
ollama = asyncio.run(count_token_with_ollama(messages, tool_names=tool_names))

print(f"Local  (tiktoken): {local}")
print(f"Ollama (inference): {ollama}")
print(f"Delta: {ollama - local:+d}")
print("=" * 60)
