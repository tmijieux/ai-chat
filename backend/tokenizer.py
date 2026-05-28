"""
Local Qwen3.5 tokenizer built from the GGUF file — no PyTorch, no downloads.

Exposes:
  count_tokens(messages, tools=None) -> int
  warmup() -> None   # call at startup to avoid first-request latency
"""
import os
import struct
from typing import Any

import tiktoken
from jinja2 import Environment

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


# ---------------------------------------------------------------------------
# Fast GGUF KV-only reader — skips tensor info entirely
# ---------------------------------------------------------------------------

_GGUF_SCALAR: dict[int, tuple[str, int]] = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
# INT/UINT 32/64 — batch-unpack whole arrays in one struct call
_GGUF_BATCH: dict[int, tuple[str, int]] = {
    4: ("I", 4), 5: ("i", 4), 10: ("Q", 8), 11: ("q", 8),
}


def _read_gguf_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def _read_gguf_value(f, vtype: int):
    if vtype in _GGUF_SCALAR:
        fmt, size = _GGUF_SCALAR[vtype]
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == 8:  # STRING
        return _read_gguf_string(f)
    if vtype == 9:  # ARRAY
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if elem_type in _GGUF_BATCH:
            char, esize = _GGUF_BATCH[elem_type]
            return list(struct.unpack(f"<{count}{char}", f.read(count * esize)))
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def _read_gguf_kv(path: str) -> dict:
    """Read only the KV metadata section from a GGUF file; stop before tensor info."""
    with open(path, "rb", buffering=1 << 20) as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        f.read(4)   # version uint32
        f.read(8)   # tensor_count uint64
        kv_count = struct.unpack("<Q", f.read(8))[0]
        result: dict = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            result[key] = _read_gguf_value(f, vtype)
    return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_enc: tiktoken.Encoding | None = None
_chat_template: str | None = None


def _load() -> tuple[tiktoken.Encoding, str]:
    global _enc, _chat_template
    if _enc is not None and _chat_template is not None:
        return _enc, _chat_template

    kv = _read_gguf_kv(GGUF_PATH)

    tokens: list[str] = kv["tokenizer.ggml.tokens"]
    token_types: list[int] = kv["tokenizer.ggml.token_type"]
    chat_template_str: str = kv["tokenizer.chat_template"]

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


def warmup() -> None:
    """Eagerly load the tokenizer. Call once at backend startup (blocking)."""
    _load()


def render_messages(messages: list[dict[str, Any]], tools: list | None, add_generation_prompt: bool = True) -> str:
    _, chat_template_str = _load()

    def raise_exception(msg):
        raise ValueError(msg)

    import json
    env = Environment()
    env.globals["raise_exception"] = raise_exception
    def _go_json(x, **kw):
        s = json.dumps(x, ensure_ascii=False)
        s = s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        return s
    env.filters["tojson"] = _go_json

    return env.from_string(chat_template_str).render(
        messages=messages,
        tools=tools or [],
        add_generation_prompt=add_generation_prompt,
        add_vision_id=False,
        enable_thinking=True,
    )


def count_tokens(messages: list[dict[str, Any]], tools: list | None = None) -> int:
    enc, _ = _load()
    add_generation_prompt = len(messages) == 0 or messages[-1]["role"] != "assistant"
    rendered = render_messages(messages, tools, add_generation_prompt=add_generation_prompt)
    return len(enc.encode(rendered, allowed_special="all"))
