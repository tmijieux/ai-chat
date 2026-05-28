"""
Replicate Go's marshalWithSpaces + api.Tool struct field ordering exactly.

marshalWithSpaces does: json.Marshal (compact, & escaped) then adds space after each : and , outside strings.
api.ToolFunctionParameters struct field order: type, $defs, items, required, properties
api.ToolProperty struct field order: anyOf, type, items, description, enum, properties, required
api.ToolPropertiesMap: preserves INSERTION ORDER (orderedmap)

So the differences vs Python json.dumps are:
  1. & → & (already fixed)
  2. required BEFORE properties in parameters
  3. description BEFORE enum in property values (Go struct: type, items, description, enum)
  4. Python uses ': ' and ', ' separators — same as marshalWithSpaces ✓
"""
import asyncio, json, re
import aiohttp
from tokenizer import _load, render_messages
from agent.tools import TOOL_REGISTRY
from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL

BASE = [{"role": "user", "content": "."}]
enc, chat_template_str = _load()


def marshal_with_spaces(obj) -> str:
    """Replicate Go's marshalWithSpaces: compact JSON + space after : and , outside strings."""
    compact = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    # HTML-escape & < > like Go's json.Marshal
    compact = compact.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    # Add spaces after : and , that are outside strings (like marshalWithSpaces)
    out = []
    in_str = False
    escaped = False
    for c in compact:
        if in_str:
            out.append(c)
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                in_str = False
        else:
            out.append(c)
            if c == '"':
                in_str = True
            elif c in ':,':
                out.append(' ')
    return ''.join(out)


def reorder_to_go_struct(schema: dict) -> dict:
    """Reorder fields to match Go's api.Tool struct declaration order."""
    import copy
    s = copy.deepcopy(schema)
    f = s.get("function", {})
    params = f.get("parameters", {})
    if not params:
        return s

    # ToolFunctionParameters: type, $defs, items, required, properties
    new_params = {}
    for k in ["type", "$defs", "items"]:
        if k in params:
            new_params[k] = params[k]
    if "required" in params and params["required"]:
        new_params["required"] = params["required"]
    if "properties" in params:
        # ToolProperty struct: anyOf, type, items, description, enum, properties, required
        new_props = {}
        for prop_name, prop_val in params["properties"].items():
            pv = {}
            for k in ["anyOf", "type", "items", "description", "enum", "properties", "required"]:
                if k in prop_val:
                    pv[k] = prop_val[k]
            for k, v in prop_val.items():
                if k not in pv:
                    pv[k] = v
            new_props[prop_name] = pv
        new_params["properties"] = new_props
    for k, v in params.items():
        if k not in new_params:
            new_params[k] = v
    s["function"]["parameters"] = new_params
    return s


async def compare_render(label: str, schemas: list[dict], tojson_fn):
    from jinja2 import Environment

    def raise_exception(msg): raise ValueError(msg)
    env = Environment()
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = tojson_fn

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


def current_tojson(x, **kw):
    s = json.dumps(x, ensure_ascii=False)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def marshalwithspaces_tojson(x, **kw):
    return marshal_with_spaces(x)


def go_struct_tojson(x, **kw):
    """Reorder to Go struct field order then apply marshalWithSpaces."""
    reordered = reorder_to_go_struct(x) if isinstance(x, dict) else x
    return marshal_with_spaces(reordered)


async def main():
    normal_schemas = [{"type": "function", "function": t.to_ollama_schema()} for t in TOOL_REGISTRY.values()]

    print("=== SHOW ONE TOOL JSON UNDER EACH APPROACH ===")
    glob = {"type": "function", "function": TOOL_REGISTRY["glob_files"].to_ollama_schema()}
    print(f"  Python json.dumps:         {current_tojson(glob)[:120]}")
    print(f"  marshalWithSpaces:         {marshalwithspaces_tojson(glob)[:120]}")
    print(f"  Go struct+marshalWSpaces:  {go_struct_tojson(glob)[:120]}")
    run = {"type": "function", "function": TOOL_REGISTRY["run_shell"].to_ollama_schema()}
    print(f"\n  run_shell Python:          ...{current_tojson(run)[-80:]}")
    print(f"  run_shell Go struct:       ...{go_struct_tojson(run)[-80:]}")

    print("\n=== ALL TOOLS: DIFFERENT SERIALIZATION APPROACHES ===")
    await compare_render("current (& escaped, Python order)", normal_schemas, current_tojson)
    await compare_render("marshalWithSpaces only (no struct reorder)", normal_schemas, marshalwithspaces_tojson)
    await compare_render("Go struct order + marshalWithSpaces", normal_schemas, go_struct_tojson)

    print("\n=== PER-TOOL: Go struct order + marshalWithSpaces ===")
    for name, tool in TOOL_REGISTRY.items():
        s = [{"type": "function", "function": tool.to_ollama_schema()}]
        await compare_render(name, s, go_struct_tojson)


if __name__ == "__main__":
    asyncio.run(main())
