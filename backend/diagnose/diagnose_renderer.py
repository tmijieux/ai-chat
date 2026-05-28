"""
Replicate Ollama's Qwen35Renderer.Render() + marshalWithSpaces(api.Tool) exactly in Python.
If token counts match, we can replace render_messages in tokenizer.py.
"""
import asyncio, json
import aiohttp
from tokenizer import _load
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from agent.count_token import count_token_with_ollama
from agent.agent import MODEL_NAME, OLLAMA_CHAT_URL

enc, _ = _load()

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"

TOOL_POSTAMBLE = """\
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


def _add_spaces(s: str) -> str:
    """Add space after : and , outside strings (Go marshalWithSpaces)."""
    out = []
    in_str = False
    escaped = False
    for c in s:
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


def marshal_tool(tool: dict) -> str:
    """
    Replicate Go's marshalWithSpaces(api.Tool).
    Field ordering follows Go struct declarations:
      - ToolFunctionParameters: type, $defs, items, required (omitempty), properties
      - ToolProperty: anyOf, type, items, description, enum (all omitempty)
    Properties map is an orderedmap → preserves our insertion order.
    """
    func_d = tool.get("function", {})
    params = func_d.get("parameters", {})

    # Reorder each property value: anyOf, type, items, description, enum
    def reorder_prop(pv: dict) -> dict:
        op = {}
        for k in ["anyOf", "type", "items", "description", "enum", "properties", "required"]:
            if k in pv:
                op[k] = pv[k]
        for k, v in pv.items():
            if k not in op:
                op[k] = v
        return op

    # Build parameters in Go struct order: type, $defs, items, required, properties
    ordered_params = {}
    ordered_params["type"] = params.get("type", "object")
    if "$defs" in params:
        ordered_params["$defs"] = params["$defs"]
    if "items" in params:
        ordered_params["items"] = params["items"]
    required = params.get("required") or []
    if required:  # omitempty
        ordered_params["required"] = required
    if "properties" in params:
        ordered_params["properties"] = {
            name: reorder_prop(pv)
            for name, pv in params["properties"].items()
        }

    # Build tool in Go struct order
    ordered_tool = {
        "type": tool.get("type", "function"),
        "function": {
            "name": func_d.get("name", ""),
            "description": func_d.get("description", ""),
            "parameters": ordered_params,
        },
    }

    compact = json.dumps(ordered_tool, ensure_ascii=False, separators=(',', ':'))
    compact = compact.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return _add_spaces(compact)


def render_qwen35(messages: list, tools: list, add_generation_prompt: bool = True) -> str:
    """Python port of Qwen35Renderer.Render() with isThinking=True, emitEmptyThinkOnNoThink=True."""
    sb = []

    # Tool block (before messages)
    if tools:
        sb.append(IM_START + "system\n")
        sb.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for tool in tools:
            sb.append("\n")
            sb.append(marshal_tool(tool))
        sb.append("\n")
        sb.append(TOOL_POSTAMBLE)
        # System message content (if first message is system)
        if messages and messages[0]["role"] == "system":
            sys_content = messages[0]["content"].strip()
            if sys_content:
                sb.append("\n\n")
                sb.append(sys_content)
        sb.append(IM_END + "\n")
    elif messages and messages[0]["role"] == "system":
        sb.append(IM_START + "system\n" + messages[0]["content"].strip() + IM_END + "\n")

    # Find lastQueryIndex (last non-tool-response user message)
    start_idx = 1 if (messages and messages[0]["role"] == "system") else 0
    msgs = messages[start_idx:]

    last_query_idx = len(msgs) - 1
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m["role"] == "user":
            content = m["content"].strip()
            if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
                last_query_idx = i
                break

    for i, m in enumerate(msgs):
        role = m["role"]
        content = m.get("content", "").strip()
        last = i == len(msgs) - 1
        prefill = last and role == "assistant"

        if role in ("user",) or (role == "system" and i != 0):
            sb.append(IM_START + role + "\n" + content + IM_END + "\n")

        elif role == "assistant":
            thinking = m.get("thinking", "")
            # splitQwen35ReasoningContent: if thinking field present, use it directly
            if thinking:
                reasoning = thinking.strip()
                remaining = content
            else:
                # extract from <think>...</think> in content
                import re
                close = content.find("</think>")
                if close != -1:
                    before = content[:close]
                    open_idx = before.rfind("<think>")
                    if open_idx != -1:
                        reasoning = before[open_idx + len("<think>"):]
                    else:
                        reasoning = before
                    remaining = content[close + len("</think>"):].lstrip("\n")
                else:
                    reasoning = ""
                    remaining = content
                reasoning = reasoning.strip()

            if i > last_query_idx:
                sb.append(IM_START + role + "\n<think>\n" + reasoning + "\n</think>\n\n" + remaining)
            else:
                sb.append(IM_START + role + "\n" + remaining)

            # tool calls
            tool_calls = m.get("tool_calls", [])
            for j, tc in enumerate(tool_calls):
                if j == 0 and remaining.strip():
                    sb.append("\n\n")
                elif j > 0:
                    sb.append("\n")
                fn = tc.get("function", {})
                sb.append("<tool_call>\n<function=" + fn.get("name", "") + ">\n")
                for arg_name, arg_val in (fn.get("arguments") or {}).items():
                    sb.append("<parameter=" + arg_name + ">\n")
                    sb.append(str(arg_val))
                    sb.append("\n</parameter>\n")
                sb.append("</function>\n</tool_call>")

            if not prefill:
                sb.append(IM_END + "\n")

        elif role == "tool":
            if i == 0 or msgs[i - 1]["role"] != "tool":
                sb.append(IM_START + "user")
            sb.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if i == len(msgs) - 1 or msgs[i + 1]["role"] != "tool":
                sb.append(IM_END + "\n")

    # Generation prompt
    if add_generation_prompt:
        sb.append(IM_START + "assistant\n<think>\n")

    return "".join(sb)


def count_tokens_qwen35(messages: list, tools: list | None = None) -> int:
    tools = tools or []
    add_gen = not messages or messages[-1]["role"] != "assistant"
    rendered = render_qwen35(messages, tools, add_generation_prompt=add_gen)
    return len(enc.encode(rendered, allowed_special="all"))


async def main():
    BASE = [{"role": "user", "content": "."}]
    ALL_NAMES = list(TOOL_REGISTRY.keys())
    ALL_TOOLS = get_ollama_tool_list(ALL_NAMES)

    from tokenizer import count_tokens

    print("=== ALL TOOLS ===")
    l_old = count_tokens(BASE, ALL_TOOLS)
    l_new = count_tokens_qwen35(BASE, ALL_TOOLS)
    o = await count_token_with_ollama(BASE, tool_names=ALL_NAMES)
    print(f"  old tokenizer: local={l_old}  ollama={o}  delta={o-l_old:+d}")
    print(f"  new renderer:  local={l_new}  ollama={o}  delta={o-l_new:+d}")

    print("\n=== PER-TOOL ===")
    o0 = await count_token_with_ollama(BASE, tool_names=[])
    for name in ALL_NAMES:
        tools = get_ollama_tool_list([name])
        l = count_tokens_qwen35(BASE, tools)
        o = await count_token_with_ollama(BASE, tool_names=[name])
        print(f"  {name:25s}: local={l}  ollama={o}  delta={o-l:+d}")

    print("\n=== SAMPLE MESSAGES ===")
    from diagnose.tokenize_script import SAMPLE_MESSAGES
    tools = get_ollama_tool_list(["glob_files"])
    l = count_tokens_qwen35(SAMPLE_MESSAGES, tools)
    o = await count_token_with_ollama(SAMPLE_MESSAGES, tool_names=["glob_files"])
    print(f"  sample msgs: local={l}  ollama={o}  delta={o-l:+d}")

    print("\n=== RENDERED TEXT DIFF (no tools, 1 tool) ===")
    from tokenizer import render_messages
    r_old_no  = render_messages(BASE, [], add_generation_prompt=True)
    r_new_no  = render_qwen35(BASE, [],  add_generation_prompt=True)
    r_old_one = render_messages(BASE, get_ollama_tool_list(["glob_files"]), add_generation_prompt=True)
    r_new_one = render_qwen35(BASE, get_ollama_tool_list(["glob_files"]),  add_generation_prompt=True)
    print(f"  No tools — old={len(enc.encode(r_old_no, allowed_special='all'))}  new={len(enc.encode(r_new_no, allowed_special='all'))}")
    print(f"  1 tool   — old={len(enc.encode(r_old_one, allowed_special='all'))}  new={len(enc.encode(r_new_one, allowed_special='all'))}")
    if r_old_one != r_new_one:
        # Find first difference
        for i, (a, b) in enumerate(zip(r_old_one, r_new_one)):
            if a != b:
                print(f"  First diff at char {i}:")
                print(f"    old: {repr(r_old_one[max(0,i-30):i+60])}")
                print(f"    new: {repr(r_new_one[max(0,i-30):i+60])}")
                break
        if len(r_old_one) != len(r_new_one):
            print(f"  Length: old={len(r_old_one)}  new={len(r_new_one)}")


if __name__ == "__main__":
    asyncio.run(main())
