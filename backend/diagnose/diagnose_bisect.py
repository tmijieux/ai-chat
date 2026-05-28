"""
Bisect edit_file's schema to find exactly what makes its delta +0 vs glob_files' +1.
Compare field-by-field: swap descriptions, parameter names, etc.
"""
import asyncio, json
from tokenizer import count_tokens
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from agent.count_token import count_token_with_ollama
import aiohttp
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


def schema(name, desc, params: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": params, "required": required},
    }}


def str_param(desc): return {"type": "string", "description": desc}
def bool_param(desc): return {"type": "boolean", "description": desc}


async def main():
    glob_raw = TOOL_REGISTRY["glob_files"].to_ollama_schema()
    edit_raw = TOOL_REGISTRY["edit_file"].to_ollama_schema()

    glob_s = {"type": "function", "function": glob_raw}
    edit_s = {"type": "function", "function": edit_raw}

    print("=== BASELINE ===")
    await compare("glob_files", [glob_s])
    await compare("edit_file",  [edit_s])

    print("\n=== SWAP NAMES ===")
    # Use edit_file schema but with glob_files name
    edit_named_glob = json.loads(_go_json(edit_s))
    edit_named_glob["function"]["name"] = "glob_files"
    await compare("edit schema, name='glob_files'", [edit_named_glob])

    glob_named_edit = json.loads(_go_json(glob_s))
    glob_named_edit["function"]["name"] = "edit_file"
    await compare("glob schema, name='edit_file'", [glob_named_edit])

    print("\n=== SWAP DESCRIPTIONS ===")
    glob_edit_desc = json.loads(_go_json(glob_s))
    glob_edit_desc["function"]["description"] = edit_raw["description"]
    await compare("glob schema, edit description", [glob_edit_desc])

    edit_glob_desc = json.loads(_go_json(edit_s))
    edit_glob_desc["function"]["description"] = glob_raw["description"]
    await compare("edit schema, glob description", [edit_glob_desc])

    print("\n=== SWAP PARAMETERS ===")
    glob_edit_params = json.loads(_go_json(glob_s))
    glob_edit_params["function"]["parameters"] = json.loads(_go_json(edit_s))["function"]["parameters"]
    await compare("glob name+desc, edit parameters", [glob_edit_params])

    edit_glob_params = json.loads(_go_json(edit_s))
    edit_glob_params["function"]["parameters"] = json.loads(_go_json(glob_s))["function"]["parameters"]
    await compare("edit name+desc, glob parameters", [edit_glob_params])

    print("\n=== MINIMAL SYNTHETIC SCHEMAS ===")
    # Minimal: same structure, vary specific field
    await compare("minimal 1 str param",
        [schema("tool_a", "Does something.", {"x": str_param("A value.")}, ["x"])])
    await compare("minimal 1 bool param",
        [schema("tool_a", "Does something.", {"x": bool_param("A flag.")}, ["x"])])
    await compare("minimal required=[]",
        [schema("tool_a", "Does something.", {"x": str_param("A value.")}, [])])
    await compare("minimal 3 required params",
        [schema("tool_a", "Does X.", {"a": str_param("Param a."), "b": str_param("Param b."), "c": str_param("Param c.")}, ["a", "b", "c"])])

    print("\n=== EDIT_FILE PARAMETER NAMES VS CONTENT ===")
    # edit_file has: file_path, old_string, new_string, replace_all
    # glob_files has: pattern, path, include_ignored
    # Test: edit_file params but with glob-like names
    edit_renamed = json.loads(_go_json(edit_s))
    props = edit_renamed["function"]["parameters"]["properties"]
    renamed_props = {
        "pattern": props.get("file_path", str_param("x")),
        "path":    props.get("old_string", str_param("x")),
        "extra":   props.get("new_string", str_param("x")),
        "flag":    props.get("replace_all", bool_param("x")),
    }
    edit_renamed["function"]["parameters"]["properties"] = renamed_props
    edit_renamed["function"]["parameters"]["required"] = ["pattern", "path", "extra"]
    await compare("edit schema, param names like glob", [edit_renamed])


if __name__ == "__main__":
    asyncio.run(main())
