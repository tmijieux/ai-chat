"""
Find the specific byte sequence that causes tiktoken to undercount by 1 vs llama.cpp
for 7 out of 9 tools (edit_file and run_shell are immune due to compensating chars).

Strategy: bisect each tool JSON to find which half contains the discrepancy.
We do this by sending sub-schemas to Ollama and comparing local vs Ollama token counts.
"""
import asyncio, json
from tokenizer import _load, count_tokens
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from agent.count_token import count_token_with_ollama

BASE = [{"role": "user", "content": "."}]
enc, _ = _load()
def tok(s): return len(enc.encode(s, allowed_special="all"))


def _go_json(x) -> str:
    s = json.dumps(x, ensure_ascii=False)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


async def delta_for_tool(name: str) -> int:
    tools = get_ollama_tool_list([name])
    l = count_tokens(BASE, tools)
    o = await count_token_with_ollama(BASE, tool_names=[name])
    return o - l


async def compare_schemas(label: str, schemas: list[dict]) -> tuple[int, int]:
    """Send raw schemas to Ollama and compare local vs Ollama."""
    import aiohttp
    from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL
    l = count_tokens(BASE, schemas)
    async with aiohttp.ClientSession() as sess:
        async with sess.post(OLLAMA_CHAT_URL, json={
            "model": MODEL_NAME, "messages": BASE, "tools": schemas,
            "stream": False, "options": {"temperature": 0, "num_predict": 1},
        }) as r:
            data = await r.json()
            o = data["prompt_eval_count"]
    print(f"  {label:50s}  local={l}  ollama={o}  delta={o-l:+d}")
    return l, o


async def find_discriminating_sequence():
    """
    Identify what's different about edit_file vs glob_files (both simple, one +0 one +1).
    Start with glob_files schema, progressively mutate it toward edit_file schema.
    """
    print("=== DISCRIMINATING SEQUENCE SEARCH ===\n")

    glob_schema = {"type": "function", "function": TOOL_REGISTRY["glob_files"].to_ollama_schema()}
    edit_schema = {"type": "function", "function": TOOL_REGISTRY["edit_file"].to_ollama_schema()}

    print("Baseline:")
    await compare_schemas("glob_files (expect delta=+1)", [glob_schema])
    await compare_schemas("edit_file  (expect delta=+0)", [edit_schema])

    # edit_file has an em-dash '—' in description. Remove it and see if delta changes.
    edit_no_dash = json.loads(_go_json(edit_schema))
    edit_no_dash["function"]["description"] = edit_no_dash["function"]["description"].replace("—", "-")
    print("\nEdit edit_file description:")
    await compare_schemas("edit_file with '-' instead of '—'", [edit_no_dash])

    # Add em-dash to glob_files description
    glob_with_dash = json.loads(_go_json(glob_schema))
    glob_with_dash["function"]["description"] += " — example"
    print("\nAdd em-dash to glob_files:")
    await compare_schemas("glob_files with '— example' appended", [glob_with_dash])

    # What about run_shell? It has \\u0026\\u0026 (literal in JSON) causing +1 locally.
    # Add \\u0026 to glob_files description
    glob_with_amp = json.loads(_go_json(glob_schema))
    glob_with_amp["function"]["description"] = glob_with_amp["function"]["description"].replace(
        "glob pattern", "glob\\u0026pattern"
    )
    print("\nAdd \\u0026 to glob_files description:")
    await compare_schemas("glob_files with \\u0026 in description", [glob_with_amp])


async def main():
    await find_discriminating_sequence()


if __name__ == "__main__":
    asyncio.run(main())
