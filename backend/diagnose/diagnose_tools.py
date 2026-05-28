"""
Isolate the tool-schema tokenization difference between local tiktoken and Ollama.
The drift is a fixed constant (~+25 with all tools) that doesn't grow with messages.
"""
import asyncio, json
from agent.count_token import count_token_with_ollama
from agent.tools import get_ollama_tool_list, TOOL_REGISTRY
from tokenizer import count_tokens, render_messages, warmup, _load

BASE = [{"role": "user", "content": "."}]


def loc(msgs, tool_names):
    tools = get_ollama_tool_list(tool_names) if tool_names else []
    return count_tokens(msgs, tools)


async def oll(msgs, tool_names):
    return await count_token_with_ollama(msgs, tool_names=tool_names or [])


async def delta_by_tool_count():
    """Check delta for 0..all tools with a minimal message."""
    print("=== DELTA BY TOOL COUNT ===")
    all_names = list(TOOL_REGISTRY.keys())
    print(f"  {'n_tools':>7}  {'local':>7}  {'ollama':>7}  {'delta':>6}")
    prev_l, prev_o = None, None
    for n in range(len(all_names) + 1):
        names = all_names[:n]
        l = loc(BASE, names)
        o = await oll(BASE, names)
        marker = ""
        if prev_l is not None:
            marker = f"  (+{l-prev_l} local / +{o-prev_o} ollama)"
        print(f"  {n:>7}  {l:>7}  {o:>7}  {o-l:>+6}{marker}")
        prev_l, prev_o = l, o


async def per_tool_delta():
    """For each tool, measure how much it adds to the delta (marginal)."""
    print("\n=== PER-TOOL MARGINAL DELTA ===")
    all_names = list(TOOL_REGISTRY.keys())
    all_tools = get_ollama_tool_list(all_names)
    full_l = loc(BASE, all_names)
    full_o = await oll(BASE, all_names)
    print(f"  {'tool':30s}  {'loc_marginal':>12}  {'oll_marginal':>12}  {'diff':>6}")
    for name in all_names:
        without = [n for n in all_names if n != name]
        l_without = loc(BASE, without)
        o_without = await oll(BASE, without)
        loc_m = full_l - l_without
        oll_m = full_o - o_without
        print(f"  {name:30s}  {loc_m:>12}  {oll_m:>12}  {oll_m-loc_m:>+6}")


async def render_diff():
    """Print the rendered system block for 0 tools vs all tools to spot differences."""
    print("\n=== RENDERED SYSTEM BLOCK ===")
    enc, _ = _load()
    def tok(s): return len(enc.encode(s, allowed_special="all"))

    all_names = list(TOOL_REGISTRY.keys())
    tools = get_ollama_tool_list(all_names)

    r_no_tools  = render_messages(BASE, [],    add_generation_prompt=True)
    r_all_tools = render_messages(BASE, tools, add_generation_prompt=True)

    # The system block is everything before the first <|im_start|>user
    def system_block(r):
        idx = r.find("<|im_start|>user")
        return r[:idx] if idx != -1 else r

    sys_no  = system_block(r_no_tools)
    sys_all = system_block(r_all_tools)

    print(f"  No tools  system block: {tok(sys_no)} tokens, {len(sys_no)} chars")
    print(f"  All tools system block: {tok(sys_all)} tokens, {len(sys_all)} chars")
    print(f"  Tool block adds: {tok(sys_all)-tok(sys_no)} tokens local")

    # Find the tool schema JSON for one tool and check our tojson vs json.dumps
    print("\n=== TOJSON SERIALIZATION CHECK ===")
    for name, tool in TOOL_REGISTRY.items():
        schema = tool.to_ollama_schema()
        our_json = json.dumps({"type": "function", "function": schema}, ensure_ascii=False)
        # Check separators
        compact = json.dumps({"type": "function", "function": schema}, ensure_ascii=False, separators=(',', ':'))
        print(f"  {name}:")
        print(f"    default json.dumps:    {tok(our_json)} tokens, {len(our_json)} chars")
        print(f"    compact separators:    {tok(compact)} tokens, {len(compact)} chars")
        # Show first 120 chars to see formatting
        print(f"    repr[:120]: {repr(our_json[:120])}")
        break  # just first tool for now

    print("\n=== FULL RENDERED TOOL BLOCK ===")
    # Extract just the tool injection block
    start = sys_all.find("<tools>")
    end   = sys_all.find("</tools>") + len("</tools>")
    tool_block = sys_all[start:end]
    print(f"  Tool block: {tok(tool_block)} tokens, {len(tool_block)} chars")
    print(tool_block)


async def main():
    warmup()
    await delta_by_tool_count()
    await per_tool_delta()
    await render_diff()


if __name__ == "__main__":
    asyncio.run(main())
