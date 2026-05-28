"""
Diagnostic script: isolate WHERE the token delta between local tiktoken
and Ollama prompt_eval_count comes from.

Run from backend/:
  python diagnose_token_delta.py

Hypotheses tested:
  H1 — constant offset (BOS token or generation-prompt token Ollama always adds)
  H2 — per-message overhead that accumulates (thinking tokens, message separators)
  H3 — add_generation_prompt mismatch (we pass False when last msg is assistant,
        but Ollama still adds it)
  H4 — enable_thinking=True in our render adds tokens Ollama doesn't count
  H5 — assistant messages with a 'thinking' key rendered differently by us vs Ollama
"""
import asyncio
from agent.tools import get_ollama_tool_list, TOOL_REGISTRY
from agent.count_token import count_token_with_ollama
from tokenizer import render_messages, count_tokens, warmup

TOOLS_1 = get_ollama_tool_list(["glob_files"])
TOOLS_ALL = get_ollama_tool_list(list(TOOL_REGISTRY.keys()))

# ---------------------------------------------------------------------------
# Message fixtures
# ---------------------------------------------------------------------------

SYS = {"role": "system", "content": "You are a coding assistant."}
U1  = {"role": "user", "content": "Find all Python files."}
A1  = {"role": "assistant", "content": "Sure, I'll glob for them."}
U2  = {"role": "user", "content": "Now count lines in main.py."}
A2  = {"role": "assistant", "content": "Done. It has 300 lines."}

# Assistant with a thinking key (as produced by agent.py when content is present)
A1_WITH_THINKING = {
    "role": "assistant",
    "content": "Sure, I'll glob for them.",
    "thinking": "The user wants Python files. I should use glob_files.",
}


def local(msgs, tools=None):
    return count_tokens(msgs, tools)


async def ollama(msgs, tools=None):
    tool_names = None
    if tools is None:
        tool_names = []
    return await count_token_with_ollama(msgs, tool_names=tool_names)


def show(label, msgs, tools, ollama_count):
    loc = local(msgs, tools)
    delta = ollama_count - loc
    print(f"  {label}")
    print(f"    local={loc}  ollama={ollama_count}  delta={delta:+d}")


async def h1_constant_offset():
    """Does the delta stay constant as we add more messages?"""
    print("\n=== H1: constant offset (BOS / generation prompt) ===")
    fixtures = [
        ("1 user msg",          [U1],             None),
        ("1 user + 1 assistant",[U1, A1],         None),
        ("2 turns",             [U1, A1, U2, A2], None),
        ("system + 1 user",     [SYS, U1],        None),
        ("system + 2 turns",    [SYS, U1, A1, U2, A2], None),
    ]
    for label, msgs, tools in fixtures:
        o = await ollama(msgs, tools)
        show(label, msgs, tools, o)


async def h2_per_message_growth():
    """Delta should grow linearly with message count if per-message overhead."""
    print("\n=== H2: per-message growth ===")
    base = [SYS, U1]
    msgs = list(base)
    for i in range(1, 6):
        msgs = msgs + [A1, U2] if i > 1 else msgs
        o = await ollama(msgs, None)
        loc = local(msgs, None)
        print(f"  {len(msgs)} messages: local={loc}  ollama={o}  delta={o-loc:+d}")
        msgs = [SYS, U1] + [A1, U2] * i


async def h3_generation_prompt():
    """Compare our rendering with add_generation_prompt True vs False."""
    print("\n=== H3: add_generation_prompt True vs False ===")
    msgs_end_user = [U1]
    msgs_end_asst = [U1, A1]

    rendered_with    = render_messages(msgs_end_user, None, add_generation_prompt=True)
    rendered_without = render_messages(msgs_end_user, None, add_generation_prompt=False)
    rendered_asst    = render_messages(msgs_end_asst, None, add_generation_prompt=False)

    from tokenizer import _load
    enc, _ = _load()
    enc_with    = len(enc.encode(rendered_with,    allowed_special="all"))
    enc_without = len(enc.encode(rendered_without, allowed_special="all"))
    enc_asst    = len(enc.encode(rendered_asst,    allowed_special="all"))

    print(f"  end-user  with gen_prompt   : {enc_with} tokens")
    print(f"  end-user  without gen_prompt: {enc_without} tokens  delta={enc_without - enc_with:+d}")
    print(f"  end-asst  without gen_prompt: {enc_asst} tokens")

    o_user = await ollama([U1], None)
    o_asst = await ollama([U1, A1], None)
    print(f"  ollama end-user: {o_user}  (our with={enc_with} delta={o_user-enc_with:+d}  our without={enc_without} delta={o_user-enc_without:+d})")
    print(f"  ollama end-asst: {o_asst}  (our={enc_asst} delta={o_asst-enc_asst:+d})")


async def h4_enable_thinking():
    """Does enable_thinking=True add tokens compared to False?"""
    print("\n=== H4: enable_thinking True vs False ===")
    from jinja2 import Environment
    import json

    from tokenizer import _load
    enc, chat_template_str = _load()

    def _render(msgs, thinking):
        def raise_exception(msg):
            raise ValueError(msg)
        env = Environment()
        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        return env.from_string(chat_template_str).render(
            messages=msgs,
            tools=[],
            add_generation_prompt=True,
            add_vision_id=False,
            enable_thinking=thinking,
        )

    msgs = [SYS, U1]
    r_thinking = _render(msgs, True)
    r_no_thinking = _render(msgs, False)

    t_thinking    = len(enc.encode(r_thinking,    allowed_special="all"))
    t_no_thinking = len(enc.encode(r_no_thinking, allowed_special="all"))
    o = await ollama(msgs, None)

    print(f"  enable_thinking=True : {t_thinking}")
    print(f"  enable_thinking=False: {t_no_thinking}  delta={t_no_thinking - t_thinking:+d}")
    print(f"  ollama               : {o}")
    print(f"  delta thinking vs ollama: {o - t_thinking:+d}")
    print(f"  delta no-thinking vs ollama: {o - t_no_thinking:+d}")

    print("\n  --- Rendered with thinking ---")
    print(r_thinking)
    print("\n  --- Rendered without thinking ---")
    print(r_no_thinking)


async def h5_thinking_field_in_assistant():
    """Does a 'thinking' key on an assistant message render extra tokens?"""
    print("\n=== H5: assistant message with 'thinking' field ===")
    msgs_no_thinking = [U1, A1]
    msgs_with_thinking = [U1, A1_WITH_THINKING]

    loc_no  = local(msgs_no_thinking, None)
    loc_with = local(msgs_with_thinking, None)
    o_no  = await ollama(msgs_no_thinking, None)
    o_with = await ollama(msgs_with_thinking, None)

    print(f"  without thinking field: local={loc_no}  ollama={o_no}  delta={o_no-loc_no:+d}")
    print(f"  with thinking field   : local={loc_with}  ollama={o_with}  delta={o_with-loc_with:+d}")
    print(f"  extra tokens from thinking field: local+{loc_with-loc_no}  ollama+{o_with-o_no}")

    r_no   = render_messages(msgs_no_thinking,   None, add_generation_prompt=True)
    r_with = render_messages(msgs_with_thinking, None, add_generation_prompt=True)
    print("\n  --- Rendered WITHOUT thinking field ---")
    print(r_no)
    print("\n  --- Rendered WITH thinking field ---")
    print(r_with)


async def main():
    warmup()
    await h1_constant_offset()
    await h2_per_message_growth()
    await h3_generation_prompt()
    await h4_enable_thinking()
    await h5_thinking_field_in_assistant()


if __name__ == "__main__":
    asyncio.run(main())
