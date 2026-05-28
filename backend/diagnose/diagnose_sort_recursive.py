"""
Test recursive alphabetical key sorting to match Go's json.Marshal map behavior.
Go sorts all map[string]interface{} keys alphabetically.
Our tool schemas sent as JSON become Go maps at every level.
"""
import asyncio, json
import aiohttp
from tokenizer import count_tokens
from agent.tools import TOOL_REGISTRY
from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL

BASE = [{"role": "user", "content": "."}]


def _go_json_sorted(x) -> str:
    """Serialize with Go-like behavior: & escaping + alphabetical map keys."""
    s = json.dumps(x, ensure_ascii=False, sort_keys=True)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def _go_json_nosort(x) -> str:
    s = json.dumps(x, ensure_ascii=False)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def make_schemas(sort_keys: bool) -> list[dict]:
    schemas = []
    for tool in TOOL_REGISTRY.values():
        s = {"type": "function", "function": tool.to_ollama_schema()}
        # Round-trip through JSON to apply sorting
        serialized = (_go_json_sorted if sort_keys else _go_json_nosort)(s)
        schemas.append(json.loads(serialized))
    return schemas


async def compare(label: str, schemas: list[dict], sort_in_tojson=False):
    """
    count_tokens uses tojson which currently sorts & escapes but not keys.
    If sort_in_tojson=True, we patch to also sort keys.
    """
    from tokenizer import render_messages, _load
    enc, _ = _load()

    import json as _json
    def tojson_sorted(x, **kw):
        s = _json.dumps(x, ensure_ascii=False, sort_keys=True)
        return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    def tojson_normal(x, **kw):
        s = _json.dumps(x, ensure_ascii=False)
        return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

    # Temporarily patch render_messages tojson filter
    from jinja2 import Environment
    orig_render = None

    rendered = render_messages.__wrapped__(BASE, schemas) if hasattr(render_messages, '__wrapped__') else None

    # Use count_tokens with the schemas as-is
    l = count_tokens(BASE, schemas)

    async with aiohttp.ClientSession() as sess:
        async with sess.post(OLLAMA_CHAT_URL, json={
            "model": MODEL_NAME, "messages": BASE, "tools": schemas,
            "stream": False, "options": {"temperature": 0, "num_predict": 1},
        }) as r:
            data = await r.json()
            o = data["prompt_eval_count"]
    print(f"  {label:55s}  local={l}  ollama={o}  delta={o-l:+d}")
    return o - l


async def compare_with_render(label: str, schemas: list[dict], sort_keys: bool):
    """Render the template with sorted or unsorted tojson and count tokens."""
    from tokenizer import _load
    from jinja2 import Environment
    import json as _json

    enc, chat_template_str = _load()

    def raise_exception(msg): raise ValueError(msg)
    env = Environment()
    env.globals["raise_exception"] = raise_exception

    if sort_keys:
        def tojson_fn(x, **kw):
            s = _json.dumps(x, ensure_ascii=False, sort_keys=True)
            return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    else:
        def tojson_fn(x, **kw):
            s = _json.dumps(x, ensure_ascii=False)
            return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
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
    print(f"  {label:55s}  local={l}  ollama={o}  delta={o-l:+d}")
    return o - l


async def main():
    schemas_normal = make_schemas(sort_keys=False)
    schemas_sorted = make_schemas(sort_keys=True)

    print("=== RENDERING WITH sort_keys in tojson filter ===")
    await compare_with_render("normal tojson (& escaped, no sort)", schemas_normal, sort_keys=False)
    await compare_with_render("sorted tojson (& escaped, sort_keys=True)", schemas_normal, sort_keys=True)

    print("\n=== SCHEMAS PRE-SORTED then normal tojson ===")
    # Pre-sort the schema dicts before passing to render
    # (simulates sending pre-sorted JSON to Ollama which then re-sorts)
    await compare_with_render("pre-sorted schemas, normal tojson", schemas_sorted, sort_keys=False)
    await compare_with_render("pre-sorted schemas, sort_keys tojson", schemas_sorted, sort_keys=True)

    print("\n=== PER-TOOL DELTA WITH sort_keys tojson ===")
    for name, tool in TOOL_REGISTRY.items():
        s = [{"type": "function", "function": tool.to_ollama_schema()}]
        d = await compare_with_render(name, s, sort_keys=True)


if __name__ == "__main__":
    asyncio.run(main())
