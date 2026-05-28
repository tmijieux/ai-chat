"""Demo script: render a simulated conversation and compare local vs Ollama token counts.

Also benchmarks local tokenizer speed vs Ollama 1-token inference.
Run from backend/:
  python tokenize_script.py
"""
import asyncio
import time
from agent.tools import get_ollama_tool_list, TOOL_REGISTRY
from agent.count_token import count_token_with_ollama
from tokenizer import render_messages, count_tokens, warmup

SAMPLE_MESSAGES = [
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


def compare_counts(messages=SAMPLE_MESSAGES, tool_names=None):
    """Render the conversation, then compare local vs Ollama token counts."""
    if tool_names is None:
        tool_names = ["glob_files"]
    tools = get_ollama_tool_list(tool_names)

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


def benchmark(messages=SAMPLE_MESSAGES, n_local=20, n_ollama=3):
    """Measure local tokenizer speed vs Ollama 1-token inference."""
    all_tool_names = list(TOOL_REGISTRY.keys())
    tools = get_ollama_tool_list(all_tool_names)

    warmup()

    times_local = []
    for _ in range(n_local):
        t = time.perf_counter()
        count_tokens(messages, tools)
        times_local.append(time.perf_counter() - t)

    async def _run_ollama():
        times = []
        for _ in range(n_ollama):
            t = time.perf_counter()
            await count_token_with_ollama(messages, all_tool_names)
            times.append(time.perf_counter() - t)
        return times

    times_ollama = asyncio.run(_run_ollama())

    avg_l = sum(times_local) / len(times_local) * 1000
    avg_o = sum(times_ollama) / len(times_ollama) * 1000

    print(f"Local  tokenizer : {avg_l:.2f} ms avg  (n={n_local})")
    print(f"Ollama inference : {avg_o:.0f} ms avg  (n={n_ollama})")
    print(f"Speedup          : {avg_o / avg_l:.0f}x")


def make_16k_messages(target=16384):
    """Duplicate SAMPLE_MESSAGES (excluding system prompt) until just under target tokens."""
    all_tool_names = list(TOOL_REGISTRY.keys())
    tools = get_ollama_tool_list(all_tool_names)
    system = SAMPLE_MESSAGES[:1]
    tail = SAMPLE_MESSAGES[1:]
    n = 1
    while count_tokens(system + tail * (n + 1), tools) < target:
        n += 1
    msgs = system + tail * n
    actual = count_tokens(msgs, tools)
    print(f"Built {n} copies -> {actual} tokens (target {target})")
    return msgs


if __name__ == "__main__":
    compare_counts()
    print()
    print("--- small message benchmark ---")
    benchmark()
    print()
    print("--- ~16k token benchmark ---")
    benchmark(messages=make_16k_messages())
