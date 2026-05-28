"""
Test hypotheses:
H1 - Go omits "required":[] with omitempty (explains -3 from empty required)
H2 - Number of required params matters
H3 - Go sorts property keys alphabetically (maps sort in Go)
"""
import asyncio, json
import aiohttp
from tokenizer import count_tokens
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from agent.count_token import count_token_with_ollama
from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL

BASE = [{"role": "user", "content": "."}]


def _go_json(x) -> str:
    s = json.dumps(x, ensure_ascii=False)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


async def compare(label: str, schemas: list[dict]):
    l = count_tokens(BASE, schemas)
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
    glob_raw = TOOL_REGISTRY["glob_files"].to_ollama_schema()
    edit_raw = TOOL_REGISTRY["edit_file"].to_ollama_schema()
    glob_s = {"type": "function", "function": glob_raw}
    edit_s = {"type": "function", "function": edit_raw}

    print("=== BASELINE ===")
    await compare("glob_files (expect +1)", [glob_s])
    await compare("edit_file  (expect +0)", [edit_s])

    print("\n=== H1: omitempty on required field ===")
    # missing "required" key entirely vs "required": []
    no_req = json.loads(_go_json(glob_s))
    del no_req["function"]["parameters"]["required"]
    await compare("glob, 'required' key absent entirely", [no_req])
    # "required": null
    null_req = json.loads(_go_json(glob_s))
    null_req["function"]["parameters"]["required"] = None
    await compare("glob, required=null", [null_req])

    print("\n=== H2: required count ===")
    # glob with all params required (3 required)
    glob_all_req = json.loads(_go_json(glob_s))
    glob_all_req["function"]["parameters"]["required"] = ["pattern", "path", "include_ignored"]
    await compare("glob, required=[pattern,path,include_ignored] (3)", [glob_all_req])

    # edit with only 1 required
    edit_one_req = json.loads(_go_json(edit_s))
    edit_one_req["function"]["parameters"]["required"] = ["file_path"]
    await compare("edit, required=[file_path] only (1)", [edit_one_req])

    # glob with 1 required (same as current) but forced different count
    glob_2req = json.loads(_go_json(glob_s))
    glob_2req["function"]["parameters"]["required"] = ["pattern", "path"]
    await compare("glob, required=[pattern,path] (2)", [glob_2req])

    print("\n=== H3: sorted property keys ===")
    # glob with alphabetically sorted properties: include_ignored, path, pattern
    glob_sorted = json.loads(_go_json(glob_s))
    props = glob_sorted["function"]["parameters"]["properties"]
    glob_sorted["function"]["parameters"]["properties"] = dict(sorted(props.items()))
    await compare("glob, properties sorted alphabetically", [glob_sorted])

    # edit with alphabetically sorted properties: file_path, new_string, old_string, replace_all
    edit_sorted = json.loads(_go_json(edit_s))
    props = edit_sorted["function"]["parameters"]["properties"]
    edit_sorted["function"]["parameters"]["properties"] = dict(sorted(props.items()))
    await compare("edit, properties sorted alphabetically", [edit_sorted])

    # all tools with sorted properties
    all_schemas = []
    for name, tool in TOOL_REGISTRY.items():
        s = {"type": "function", "function": tool.to_ollama_schema()}
        j = json.loads(_go_json(s))
        if "properties" in j["function"].get("parameters", {}):
            j["function"]["parameters"]["properties"] = dict(sorted(j["function"]["parameters"]["properties"].items()))
        all_schemas.append(j)
    print()
    await compare("ALL tools, properties sorted (expect +0?)", all_schemas)
    await compare("ALL tools, normal order (baseline)", [{"type": "function", "function": t.to_ollama_schema()} for t in TOOL_REGISTRY.values()])

    print("\n=== H3b: also sort required array ===")
    all_sorted = []
    for name, tool in TOOL_REGISTRY.items():
        s = {"type": "function", "function": tool.to_ollama_schema()}
        j = json.loads(_go_json(s))
        params = j["function"].get("parameters", {})
        if "properties" in params:
            params["properties"] = dict(sorted(params["properties"].items()))
        if "required" in params:
            params["required"] = sorted(params["required"])
        all_sorted.append(j)
    await compare("ALL tools, properties+required sorted", all_sorted)


if __name__ == "__main__":
    asyncio.run(main())
