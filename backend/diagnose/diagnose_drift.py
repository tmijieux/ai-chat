"""
Isolate the per-repetition token drift between local tiktoken and Ollama.

Strategy:
  1. Find which message type causes the drift (binary search over tail composition)
  2. Find exactly at what rep count it first appears
  3. Print the rendered text for 1-rep vs 2-rep to spot the byte difference
"""
import asyncio
from agent.count_token import count_token_with_ollama
from agent.tools import get_ollama_tool_list, TOOL_REGISTRY
from tokenizer import count_tokens, render_messages, warmup, _load

SYS = {"role": "system", "content": "You are a coding assistant. You have access to tools to browse the filesystem."}
USER = {"role": "user", "content": "Find all Python files in the backend directory."}
ASST_TOOL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
        "type": "function",
        "function": {
            "name": "glob_files",
            "arguments": {"pattern": "**/*.py", "path": "backend"},
        },
    }],
}
TOOL_RESULT = {
    "role": "tool",
    "content": '{"tool": "glob_files", "status": "success", "files": ["backend/main.py", "backend/database.py", "backend/tables.py", "backend/agent/agent.py"], "file_count": 4}',
}
ASST_FINAL = {
    "role": "assistant",
    "content": "I found 4 Python files in the backend directory: main.py, database.py, tables.py, and agent/agent.py.",
}

ALL_TOOLS = get_ollama_tool_list(list(TOOL_REGISTRY.keys()))


def loc(msgs):
    return count_tokens(msgs, ALL_TOOLS)


async def oll(msgs):
    return await count_token_with_ollama(msgs, tool_names=list(TOOL_REGISTRY.keys()))


async def test_tail_variants():
    """Try tails of increasing complexity to find which element drifts."""
    print("\n=== TAIL VARIANT DRIFT (10 reps each) ===")
    N = 10
    variants = [
        ("user only",            [USER]),
        ("user+asst",            [USER, ASST_FINAL]),
        ("user+asst_tool",       [USER, ASST_TOOL]),
        ("user+asst_tool+tool",  [USER, ASST_TOOL, TOOL_RESULT]),
        ("full tail",            [USER, ASST_TOOL, TOOL_RESULT, ASST_FINAL]),
    ]
    for label, tail in variants:
        msgs = [SYS] + tail * N
        l = loc(msgs)
        o = await oll(msgs)
        print(f"  {label:30s}  N={N}  local={l}  ollama={o}  delta={o-l:+d}")


async def test_rep_counts():
    """Track delta across rep counts for the full tail."""
    print("\n=== DELTA vs REP COUNT (full tail) ===")
    tail = [USER, ASST_TOOL, TOOL_RESULT, ASST_FINAL]
    print(f"  {'N':>4}  {'local':>7}  {'ollama':>7}  {'delta':>6}")
    for n in [1, 2, 3, 5, 10, 20, 50]:
        msgs = [SYS] + tail * n
        l = loc(msgs)
        o = await oll(msgs)
        print(f"  {n:>4}  {l:>7}  {o:>7}  {o-l:>+6}")


async def test_rep_counts_by_type():
    """Track delta across rep counts for each message type individually."""
    print("\n=== DELTA vs REP COUNT BY MESSAGE TYPE ===")
    variants = [
        ("user",         [USER]),
        ("asst_tool",    [ASST_TOOL]),
        ("tool_result",  [TOOL_RESULT]),
        ("asst_final",   [ASST_FINAL]),
    ]
    for label, tail in variants:
        print(f"\n  [{label}]")
        print(f"    {'N':>4}  {'local':>7}  {'ollama':>7}  {'delta':>6}  {'delta/N':>8}")
        for n in [1, 5, 10, 20]:
            msgs = [SYS] + tail * n
            l = loc(msgs)
            o = await oll(msgs)
            print(f"    {n:>4}  {l:>7}  {o:>7}  {o-l:>+6}  {(o-l)/n:>+8.2f}")


async def diff_rendered_text():
    """Print side-by-side rendered text for N=1 vs N=2 for the culprit message type."""
    print("\n=== RENDERED TEXT DIFF: 1 rep vs 2 reps ===")
    tail = [USER, ASST_TOOL, TOOL_RESULT, ASST_FINAL]

    r1 = render_messages([SYS] + tail * 1, ALL_TOOLS, add_generation_prompt=False)
    r2 = render_messages([SYS] + tail * 2, ALL_TOOLS, add_generation_prompt=False)

    # Find where they diverge after the first rep
    # r2 should be r1 + something; check if suffix matches
    print(f"  len(r1)={len(r1)}  len(r2)={len(r2)}  extra={len(r2)-len(r1)} chars")

    # Check if the second rep is byte-identical to the first rep
    rep_suffix = r2[len(r1):]
    first_tail_text = r1[r1.index("<|im_start|>user\nFind all Python"):]
    print(f"\n  First-rep tail ({len(first_tail_text)} chars):")
    print(repr(first_tail_text))
    print(f"\n  Second-rep tail ({len(rep_suffix)} chars):")
    print(repr(rep_suffix))
    print(f"\n  Byte-identical: {first_tail_text == rep_suffix}")

    # Token count for each tail chunk separately
    enc, _ = _load()
    def tok(s):
        return len(enc.encode(s, allowed_special="all"))

    # Count how many tokens the repeat tail adds vs the first tail
    r1_tok = tok(r1)
    r2_tok = tok(r2)
    first_tail_tok = tok(first_tail_text)
    second_tail_tok = tok(rep_suffix)
    print(f"\n  Tokens in r1: {r1_tok}")
    print(f"  Tokens in r2: {r2_tok}")
    print(f"  r2 - r1 = {r2_tok - r1_tok}  (extra tokens from 2nd rep)")
    print(f"  first tail chunk: {first_tail_tok} tokens")
    print(f"  second tail chunk: {second_tail_tok} tokens")
    print(f"  diff per rep: {second_tail_tok - first_tail_tok:+d}")

    # The BPE boundary issue: tokenize the join point
    boundary_before = r1[-50:]
    boundary_after = rep_suffix[:50]
    joint = boundary_before + boundary_after
    print(f"\n  Boundary chars (last 50 of r1 + first 50 of r2):")
    print(repr(joint))
    print(f"  Tokens in boundary_before (50 chars): {tok(boundary_before)}")
    print(f"  Tokens in boundary_after (50 chars):  {tok(boundary_after)}")
    print(f"  Tokens in joint (100 chars):           {tok(joint)}")
    print(f"  Merge effect: {tok(joint) - tok(boundary_before) - tok(boundary_after):+d}")


async def main():
    warmup()
    await test_tail_variants()
    await test_rep_counts()
    await test_rep_counts_by_type()
    await diff_rendered_text()


if __name__ == "__main__":
    asyncio.run(main())
