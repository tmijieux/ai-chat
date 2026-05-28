"""
Confirm the enum field is what causes run_shell's +20 token discrepancy.
Compare local vs Ollama for synthetic schemas with/without enum.
"""
import asyncio, json, copy
from agent.count_token import count_token_with_ollama
from tokenizer import count_tokens, warmup

BASE = [{"role": "user", "content": "."}]

# Minimal schema with and without enum
SCHEMA_NO_ENUM = {"type": "function", "function": {
    "name": "run_shell",
    "description": "Execute a shell command.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "Command to run."},
        "shell_mode": {"type": "string", "description": "Shell to use."},
    }, "required": ["command"]},
}}

SCHEMA_WITH_ENUM = copy.deepcopy(SCHEMA_NO_ENUM)
SCHEMA_WITH_ENUM["function"]["parameters"]["properties"]["shell_mode"]["enum"] = ["bash", "cmd"]


async def compare(label, schema):
    tools = [schema]
    l = count_tokens(BASE, tools)
    o = await count_token_with_ollama(BASE, tool_names=None)  # can't pass raw schema, use workaround below
    return l


async def compare_raw(label, tools):
    """Count locally and via Ollama for a list of raw tool dicts."""
    l = count_tokens(BASE, tools)
    # Ollama count_token_with_ollama only accepts tool_names; call directly
    import aiohttp
    from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL
    async with aiohttp.ClientSession() as sess:
        async with sess.post(OLLAMA_CHAT_URL, json={
            "model": MODEL_NAME,
            "messages": BASE,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1},
        }) as r:
            data = await r.json()
            o = data["prompt_eval_count"]
    print(f"  {label:40s}  local={l}  ollama={o}  delta={o-l:+d}")
    return l, o


async def main():
    warmup()
    print("=== ENUM VS NO-ENUM COMPARISON ===\n")

    l_no,   o_no   = await compare_raw("run_shell WITHOUT enum",  [SCHEMA_NO_ENUM])
    l_with, o_with = await compare_raw("run_shell WITH enum",     [SCHEMA_WITH_ENUM])

    print(f"\n  Adding enum adds to local : {l_with - l_no:+d} tokens")
    print(f"  Adding enum adds to ollama: {o_with - o_no:+d} tokens")
    print(f"  Delta difference caused by enum: {(o_with - l_with) - (o_no - l_no):+d} tokens")

    # Also check the actual run_shell tool from registry
    from agent.tools import TOOL_REGISTRY
    run_shell_schema = {"type": "function", "function": TOOL_REGISTRY["run_shell"].to_ollama_schema()}
    print()
    await compare_raw("actual run_shell from registry", [run_shell_schema])

    # Show what our tojson produces for the enum schema
    print("\n=== OUR RENDERED JSON (with enum) ===")
    print(json.dumps(SCHEMA_WITH_ENUM, ensure_ascii=False))

    print("\n=== OUR RENDERED JSON (actual run_shell) ===")
    print(json.dumps(run_shell_schema, ensure_ascii=False))

    # Check per-1-token discrepancy for simple tools: is it a separator difference?
    print("\n=== SEPARATOR HYPOTHESIS (glob_files, delta=+1) ===")
    from agent.tools import TOOL_REGISTRY
    glob_schema = {"type": "function", "function": TOOL_REGISTRY["glob_files"].to_ollama_schema()}
    # Try different JSON serializations
    from tokenizer import _load
    enc, _ = _load()
    def tok(s): return len(enc.encode(s, allowed_special="all"))

    default    = json.dumps(glob_schema, ensure_ascii=False)
    compact    = json.dumps(glob_schema, ensure_ascii=False, separators=(',', ':'))
    with_nl    = json.dumps(glob_schema, ensure_ascii=False) + "\n"

    print(f"  default (', ', ': '):  {tok(default)} tokens, {len(default)} chars")
    print(f"  compact (',', ':'):    {tok(compact)} tokens, {len(compact)} chars")
    print(f"  default + newline:     {tok(with_nl)} tokens")


if __name__ == "__main__":
    asyncio.run(main())
