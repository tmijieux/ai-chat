"""
Verify: Go's ToolFunctionParameters struct outputs required BEFORE properties.
If we reorder our parameters JSON to match, delta should reach 0.

Also check: ToolProperty struct order is anyOf, type, items, description, enum, ...
Our Python puts enum before description; Go puts description before enum.
"""
import asyncio, json
import aiohttp
from tokenizer import _load, render_messages
from agent.tools import TOOL_REGISTRY
from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL

BASE = [{"role": "user", "content": "."}]
enc, chat_template_str = _load()


def _go_json(x) -> str:
    s = json.dumps(x, ensure_ascii=False)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def reorder_schema(schema: dict) -> dict:
    """
    Reorder to match Go's struct field declaration order:
    - ToolFunctionParameters: type, $defs, items, required, properties
    - ToolProperty: anyOf, type, items, description, enum, properties, required
    """
    import copy
    s = copy.deepcopy(schema)

    params = s.get("function", {}).get("parameters", {})
    if params:
        # Reorder parameters: type first, required before properties
        ordered = {}
        for k in ["type", "$defs", "items"]:
            if k in params: ordered[k] = params[k]
        if "required" in params:
            ordered["required"] = params["required"]
        if "properties" in params:
            # Also reorder each property value to match ToolProperty struct:
            # anyOf, type, items, description, enum, properties, required
            reordered_props = {}
            for prop_name, prop_val in params["properties"].items():
                op = {}
                for k in ["anyOf", "type", "items", "description", "enum", "properties", "required"]:
                    if k in prop_val: op[k] = prop_val[k]
                # catch any leftover keys
                for k, v in prop_val.items():
                    if k not in op: op[k] = v
                reordered_props[prop_name] = op
            ordered["properties"] = reordered_props
        # catch any leftover keys
        for k, v in params.items():
            if k not in ordered: ordered[k] = v
        s["function"]["parameters"] = ordered

    return s


async def compare(label: str, schemas: list[dict], tojson_fn=None):
    from jinja2 import Environment

    def raise_exception(msg): raise ValueError(msg)
    env = Environment()
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = tojson_fn or (lambda x, **kw: _go_json(x))

    rendered = env.from_string(chat_template_str).render(
        messages=BASE, tools=schemas, add_generation_prompt=True,
        add_vision_id=False, enable_thinking=True,
    )
    l = len(enc.encode(rendered, allowed_special="all"))

    async with aiohttp.ClientSession() as sess:
        async with sess.post(OLLAMA_CHAT_URL, json={
            "model": MODEL_NAME, "messages": BASE, "tools": schemas,
            "stream": False, "options": {"temperature": 0, "num_predict": 1},
        }) as r:
            data = await r.json()
            o = data["prompt_eval_count"]
    print(f"  {label:60s}  local={l}  ollama={o}  delta={o-l:+d}")
    return o - l


async def main():
    normal_schemas  = [{"type": "function", "function": t.to_ollama_schema()} for t in TOOL_REGISTRY.values()]
    reordered_schemas = [reorder_schema(s) for s in normal_schemas]

    print("=== ALL TOOLS ===")
    await compare("normal order (baseline, expect +7)",    normal_schemas)
    await compare("Go struct order (required before props)", reordered_schemas)

    print("\n=== PER-TOOL WITH GO STRUCT ORDER ===")
    for name, tool in TOOL_REGISTRY.items():
        s = reorder_schema({"type": "function", "function": tool.to_ollama_schema()})
        await compare(name, [s])

    print("\n=== SHOW DIFF FOR ONE TOOL ===")
    glob_normal   = {"type": "function", "function": TOOL_REGISTRY["glob_files"].to_ollama_schema()}
    glob_reordered = reorder_schema(glob_normal)
    print(f"  normal   params: {_go_json(glob_normal['function']['parameters'])}")
    print(f"  reordered params: {_go_json(glob_reordered['function']['parameters'])}")
    run_normal   = {"type": "function", "function": TOOL_REGISTRY["run_shell"].to_ollama_schema()}
    run_reordered = reorder_schema(run_normal)
    print(f"\n  run_shell normal   params snippet: ...{_go_json(run_normal['function']['parameters'])[-100:]}")
    print(f"  run_shell reordered params snippet: ...{_go_json(run_reordered['function']['parameters'])[-100:]}")


if __name__ == "__main__":
    asyncio.run(main())
