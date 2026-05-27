"""
Local Qwen3.5 tokenizer built from the GGUF file — no PyTorch, no downloads.

Exposes:
  count_tokens(messages, tools=None) -> int
"""
import os
from typing import Any

import tiktoken
from jinja2 import Environment
from gguf import GGUFReader

GGUF_PATH = os.environ.get(
    "QWEN_GGUF_PATH",
    "C:\\Users\\tmijieux\\.ollama\\models\\blobs\\"
    "sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
)

# Qwen3.5 tiktoken pre-tokenization regex
_QWEN_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# GPT-2 byte encoding inverse: unicode char → raw byte
def _unicode_to_byte_map() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

_U2B = _unicode_to_byte_map()

def _gpt2_str_to_bytes(s: str) -> bytes:
    return bytes([_U2B[c] for c in s])


# Lazy singleton
_enc: tiktoken.Encoding | None = None
_chat_template: str | None = None


def _load() -> tuple[tiktoken.Encoding, str]:
    global _enc, _chat_template
    if _enc is not None and _chat_template is not None:
        return _enc, _chat_template

    reader = GGUFReader(GGUF_PATH, "r")
    fields = reader.fields

    tokens: list[str] = fields["tokenizer.ggml.tokens"].contents()
    token_types: list[int] = fields["tokenizer.ggml.token_type"].contents()
    chat_template_str: str = fields["tokenizer.chat_template"].contents()

    mergeable_ranks: dict[bytes, int] = {}
    special_tokens: dict[str, int] = {}

    for rank, (tok, tok_type) in enumerate(zip(tokens, token_types)):
        if tok_type in (3, 4):  # CONTROL and USER_DEFINED both need special token handling
            special_tokens[tok] = rank
        else:
            mergeable_ranks[_gpt2_str_to_bytes(tok)] = rank

    _enc = tiktoken.Encoding(
        name="qwen35",
        pat_str=_QWEN_PAT,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    _chat_template = chat_template_str
    return _enc, _chat_template


def render_messages(messages: list[dict[str, Any]], tools: list | None, add_generation_prompt: bool = True) -> str:
    _, chat_template_str = _load()

    def raise_exception(msg):
        raise ValueError(msg)

    import json
    env = Environment()
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)

    return env.from_string(chat_template_str).render(
        messages=messages,
        tools=tools or [],
        add_generation_prompt=add_generation_prompt,
        add_vision_id=False,
        enable_thinking=True,
    )


def count_tokens(messages: list[dict[str, Any]], tools: list | None = None) -> int:
    enc, _ = _load()
    add_generation_prompt = not messages or messages[-1]["role"] != "assistant"
    rendered = render_messages(messages, tools, add_generation_prompt=add_generation_prompt)
    return len(enc.encode(rendered, allowed_special="all"))
