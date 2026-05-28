"""
Investigate the +1/tool residual delta.
Hypothesis: tiktoken and llama.cpp split the \n{"type": boundary differently,
or there's a Go-level character escaping we haven't caught yet.

Steps:
1. Tokenize the exact tool block text character-by-character to find where +1 comes from.
2. Check if Go escapes any char we haven't handled (U+2028, U+2029, etc.)
3. Test whether the delta is in the tool JSON content or in the wrapping/boundary.
"""
import asyncio, json
from tokenizer import _load, render_messages, count_tokens
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from agent.count_token import count_token_with_ollama

BASE = [{"role": "user", "content": "."}]
ALL_NAMES = list(TOOL_REGISTRY.keys())
ALL_TOOLS = get_ollama_tool_list(ALL_NAMES)

enc, _ = _load()
def tok(s): return len(enc.encode(s, allowed_special="all"))


def check_go_escaping():
    """
    Go json.Marshal also escapes U+2028 (line sep) and U+2029 (para sep).
    Check if any tool schema contains these or other non-ASCII Go would escape.
    """
    print("=== GO EXTRA ESCAPING CHECK ===")
    # Go escapes: & &, < <, > >, U+2028  , U+2029
    extra = {' ': '\\u2028', ' ': '\\u2029'}
    for name, tool in TOOL_REGISTRY.items():
        schema = json.dumps({"type": "function", "function": tool.to_ollama_schema()}, ensure_ascii=False)
        hits = [(ch, esc) for ch, esc in extra.items() if ch in schema]
        amp_count = schema.count("&")
        lt_count  = schema.count("<")
        gt_count  = schema.count(">")
        if hits or amp_count or lt_count or gt_count:
            print(f"  {name}: & x{amp_count}  < x{lt_count}  > x{gt_count}  extra={hits}")
        else:
            print(f"  {name}: clean")


def check_tool_json_token_counts():
    """Show local token count for each individual tool JSON and compare."""
    print("\n=== PER-TOOL JSON TOKEN COUNTS (local) ===")
    for name, tool in TOOL_REGISTRY.items():
        schema = {"type": "function", "function": tool.to_ollama_schema()}
        s = json.dumps(schema, ensure_ascii=False)
        s_escaped = s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        print(f"  {name:25s}  raw={tok(s):4d}  go_escaped={tok(s_escaped):4d}  diff={tok(s_escaped)-tok(s):+d}")


def check_boundary_tokens():
    """
    For each tool, measure how many tokens the \n + tool_json chunk contributes,
    vs how many tokens the boundary \n contributes alone.
    The sum should equal the full block count if there's no BPE merge at boundaries.
    """
    print("\n=== BOUNDARY TOKENIZATION ===")
    rendered = render_messages(BASE, ALL_TOOLS, add_generation_prompt=True)
    start = rendered.find("<tools>") + len("<tools>")
    end = rendered.find("</tools>")
    inner = rendered[start:end]  # \n{json1}\n{json2}...\n{jsonN}

    # Split into chunks: each is \n + tool_json
    chunks = ["\n" + t for t in inner.split("\n") if t.startswith("{")]

    print(f"  Sum of chunk tokens: {sum(tok(c) for c in chunks)}")
    print(f"  Full inner block:    {tok(inner)}")
    print(f"  BPE merge effect:    {tok(inner) - sum(tok(c) for c in chunks):+d}")
    print()
    for i, chunk in enumerate(chunks):
        name = ALL_NAMES[i] if i < len(ALL_NAMES) else f"chunk_{i}"
        print(f"  [{name:25s}]  {tok(chunk):4d} tokens  chars={len(chunk)}")


async def check_no_tools_vs_one():
    """Narrow down: does 0→1 tool add exactly 1 extra vs Ollama for every tool?"""
    print("\n=== 0 vs 1 TOOL DELTA PER TOOL ===")
    l0 = count_tokens(BASE, [])
    o0 = await count_token_with_ollama(BASE, tool_names=[])
    print(f"  no tools: local={l0}  ollama={o0}  delta={o0-l0:+d}")
    for name in ALL_NAMES:
        tools = get_ollama_tool_list([name])
        l = count_tokens(BASE, tools)
        o = await count_token_with_ollama(BASE, tool_names=[name])
        print(f"  {name:25s}: local={l}  ollama={o}  delta={o-l:+d}  (adds local+{l-l0} / ollama+{o-o0})")


async def main():
    check_go_escaping()
    check_tool_json_token_counts()
    check_boundary_tokens()
    await check_no_tools_vs_one()


if __name__ == "__main__":
    asyncio.run(main())
