"""
Test hypothesis: Go's json.Marshal escapes & → \\u0026, causing Ollama to tokenize
the run_shell description with more tokens than our Python json.dumps.
"""
import asyncio, json, re
from agent.count_token import count_token_with_ollama
from agent.tools import TOOL_REGISTRY
from tokenizer import count_tokens, warmup, _load

BASE = [{"role": "user", "content": "."}]


def html_escape_json(obj) -> str:
    """Mimic Go json.Marshal: escape &, <, > to \\uXXXX."""
    s = json.dumps(obj, ensure_ascii=False)
    s = s.replace("&", "\\u0026")
    s = s.replace("<", "\\u003c")
    s = s.replace(">", "\\u003e")
    return s


def count_with_custom_json(msgs, tool_dicts):
    """Count tokens using a custom serializer for tools."""
    from jinja2 import Environment
    from tokenizer import _load
    enc, chat_template_str = _load()

    def raise_exception(msg): raise ValueError(msg)
    env = Environment()
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = html_escape_json  # swap the serializer

    rendered = env.from_string(chat_template_str).render(
        messages=msgs,
        tools=tool_dicts,
        add_generation_prompt=True,
        add_vision_id=False,
        enable_thinking=True,
    )
    return len(enc.encode(rendered, allowed_special="all"))


async def main():
    warmup()
    enc, _ = _load()
    def tok(s): return len(enc.encode(s, allowed_special="all"))

    run_shell = TOOL_REGISTRY["run_shell"]
    schema = {"type": "function", "function": run_shell.to_ollama_schema()}

    # Show what each serializer produces for run_shell
    default_json = json.dumps(schema, ensure_ascii=False)
    go_like_json  = html_escape_json(schema)

    print("=== SERIALIZER COMPARISON ===")
    print(f"  default json.dumps : {tok(default_json):>5} tokens, {len(default_json):>5} chars")
    print(f"  go-like (& escaped): {tok(go_like_json):>5} tokens, {len(go_like_json):>5} chars")
    print(f"  token difference   : {tok(go_like_json) - tok(default_json):>+5}")

    # Show the escaped portions
    print("\n=== ESCAPED CHARACTERS IN run_shell ===")
    for i, (a, b) in enumerate(zip(default_json, go_like_json)):
        if a != b:
            ctx_start = max(0, i-20)
            ctx_default = default_json[ctx_start:i+30]
            ctx_go      = go_like_json[ctx_start:i+30]
            print(f"  pos {i}: default={repr(ctx_default)}")
            print(f"          go     ={repr(ctx_go)}")
            break
    # Count & occurrences
    amp_count = default_json.count("&")
    print(f"\n  '&' chars in serialized run_shell: {amp_count}")
    print(f"  token cost of '&' alone:          {tok('&')}")
    print(f"  token cost of '\\u0026' alone:      {tok(chr(92)+'u0026')}")

    # Full comparison: count all tokens with go-like serializer
    import aiohttp
    from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL
    tools = [{"type": "function", "function": t.to_ollama_schema()} for t in TOOL_REGISTRY.values()]

    local_default = count_tokens(BASE, tools)
    local_go_like = count_with_custom_json(BASE, tools)

    async with aiohttp.ClientSession() as sess:
        async with sess.post(OLLAMA_CHAT_URL, json={
            "model": MODEL_NAME, "messages": BASE, "tools": tools,
            "stream": False, "options": {"temperature": 0, "num_predict": 1},
        }) as r:
            data = await r.json()
            ollama_count = data["prompt_eval_count"]

    print(f"\n=== ALL TOOLS: LOCAL vs OLLAMA ===")
    print(f"  local (python json.dumps):    {local_default}  delta vs ollama={ollama_count-local_default:+d}")
    print(f"  local (go-like & escaping):   {local_go_like}  delta vs ollama={ollama_count-local_go_like:+d}")
    print(f"  ollama:                        {ollama_count}")


if __name__ == "__main__":
    asyncio.run(main())
