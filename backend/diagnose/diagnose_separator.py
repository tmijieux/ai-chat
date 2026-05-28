"""
Test what separator/whitespace variant in the tool block matches Ollama's count.
Hypothesis: Ollama's C++ Jinja2 renders slightly different whitespace per tool.
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


def tool_block_variants():
    rendered = render_messages(BASE, ALL_TOOLS, add_generation_prompt=True)
    start = rendered.find("<tools>")
    end = rendered.find("</tools>") + len("</tools>")
    tool_block = rendered[start:end]

    lines = tool_block.split("\n")
    # lines[0]="<tools>", lines[1..N]=tool jsons, lines[-1]="</tools>"

    def mutate(suffix_fn):
        out = []
        for i, line in enumerate(lines):
            if line and line[0] == "{":
                out.append(line + suffix_fn(i, line))
            else:
                out.append(line)
        return "\n".join(out)

    base_tok = tok(tool_block)
    print(f"Original tool block:           {base_tok} tokens")

    for label, suffix_fn in [
        ("trailing comma (all)",           lambda i, l: ","),
        ("trailing comma (non-last)",      lambda i, l: "," if lines[i+1] != "</tools>" else ""),
        ("trailing space",                 lambda i, l: " "),
        ("trailing newline",               lambda i, l: "\n"),
    ]:
        v = mutate(suffix_fn)
        t = tok(v)
        print(f"  {label:35s}  {t} tokens  (delta={t - base_tok:+d})")


async def main():
    tool_block_variants()

    local = count_tokens(BASE, ALL_TOOLS)
    ollama = await count_token_with_ollama(BASE, tool_names=ALL_NAMES)
    print(f"\nFull count — local={local}  ollama={ollama}  delta={ollama - local:+d}")

    # Also test with 1 tool to isolate per-tool overhead
    one_name = ["glob_files"]
    one_tools = get_ollama_tool_list(one_name)
    l1 = count_tokens(BASE, one_tools)
    o1 = await count_token_with_ollama(BASE, tool_names=one_name)
    print(f"1 tool  — local={l1}  ollama={o1}  delta={o1 - l1:+d}")


if __name__ == "__main__":
    asyncio.run(main())
