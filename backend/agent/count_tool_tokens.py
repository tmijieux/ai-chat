"""
One-time script to measure the token cost of each tool individually and in context.

Run from the backend/ directory:
    python -m agent.count_tool_tokens

Methodology:
  isolated delta:  count_with_1_tool - count_with_0_tools  (original)
  marginal delta:  count_with_all_tools - count_with_all_tools_minus_1  (accurate for full set)
"""

from tokenizer import count_tokens, warmup
from .tools import TOOL_REGISTRY, CONVERSATIONAL_TOOLS, PLAN_MODE_TOOLS

DUMMY_MESSAGES = [{"role": "user", "content": "."}]
MINIMAL_TOOL = [{"type": "function", "function": {"name": "x", "description": "", "parameters": {}}}]


def _count(tools: list[dict], messages=DUMMY_MESSAGES) -> int:
    return count_tokens(messages, tools)


def main():
    warmup()

    baseline = _count([])
    print(f"baseline (no tools, no system): {baseline} tokens")

    minimal_tool = _count(MINIMAL_TOOL)
    system_x = _count([], messages=[{"role": "system", "content": "x"}] + DUMMY_MESSAGES)
    print(f"minimal tool (name='x', empty desc, no params): {minimal_tool} tokens (+{minimal_tool - baseline} from baseline)")
    print(f"system prompt 'x' (no tools): {system_x} tokens (+{system_x - baseline} from baseline)")
    print(f"tool framework overhead vs equivalent system text: {minimal_tool - system_x} tokens")

    minimal_tool_x2 = _count(MINIMAL_TOOL * 2)
    minimal_tool_x3 = _count(MINIMAL_TOOL * 3)
    print(f"2 minimal tools: {minimal_tool_x2} tokens (+{minimal_tool_x2 - baseline} from baseline)")
    print(f"3 minimal tools: {minimal_tool_x3} tokens (+{minimal_tool_x3 - baseline} from baseline)")
    print(f"delta 1→2 tools: {minimal_tool_x2 - minimal_tool} tokens")
    print(f"delta 2→3 tools: {minimal_tool_x3 - minimal_tool_x2} tokens\n")

    # --- isolated delta (1 tool vs no tools) ---
    isolated = {}
    print("--- isolated delta (1 tool vs no tools) ---")
    for name, tool in TOOL_REGISTRY.items():
        schema = {"type": "function", "function": tool.to_ollama_schema()}
        count = _count([schema])
        delta = count - baseline
        isolated[name] = delta
        print(f"  {name}: {count} total, +{delta} isolated delta")

    # --- marginal delta (all tools minus 1) ---
    all_schemas = [{"type": "function", "function": t.to_ollama_schema()} for t in TOOL_REGISTRY.values()]
    full_count = _count(all_schemas)
    print(f"\n--- marginal delta (all {len(TOOL_REGISTRY)} tools = {full_count}, removing one at a time) ---")
    marginal = {}
    for name, tool in TOOL_REGISTRY.items():
        without = [s for n, s in zip(TOOL_REGISTRY.keys(), all_schemas) if n != name]
        count_without = _count(without)
        delta = full_count - count_without
        marginal[name] = delta
        print(f"  {name}: marginal_delta={delta}")

    print("\n--- verification: measured_delta (isolated) vs current tool_count ---")
    from agent.tools.base import TOOL_FRAMEWORK_OVERHEAD
    for name, tool in TOOL_REGISTRY.items():
        iso = isolated[name]
        mar = marginal[name]
        computed = tool.token_count
        expected_iso = iso - TOOL_FRAMEWORK_OVERHEAD
        match = "OK" if computed == expected_iso else f"MISMATCH (got {computed}, expected {expected_iso})"
        print(f"  {name}: isolated={iso}, marginal={mar}, token_count={computed} [{match}]")

    print(f"\n  formula total (OVERHEAD + sum token_count): {TOOL_FRAMEWORK_OVERHEAD + sum(t.token_count for t in TOOL_REGISTRY.values())}")
    print(f"  actual total (all tools):                   {full_count - baseline}")

    # --- always-on tools (conversational + plan mode) ---
    always_on = {**CONVERSATIONAL_TOOLS, **PLAN_MODE_TOOLS}
    print(f"\n--- always-on tools (isolated delta, 1 tool vs no tools) ---")
    for name, tool in always_on.items():
        schema = {"type": "function", "function": tool.to_ollama_schema()}
        count = _count([schema])
        delta = count - baseline
        expected_token_count = delta - TOOL_FRAMEWORK_OVERHEAD
        current = tool.measured_delta
        match = "OK" if current == delta else f"MISMATCH (stored={current}, measured={delta})"
        print(f"  {name}: isolated_delta={delta}, token_count={expected_token_count} [{match}]")


if __name__ == "__main__":
    main()
