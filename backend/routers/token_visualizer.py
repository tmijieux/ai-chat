"""Debug page backend: shows llama.cpp's own tokenization of a turn, including
special/control tokens, straight from the llama-server HTTP API (no local
reimplementation of templating or tokenization)."""
import json
import logging
from typing import Any

import aiohttp
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import tokenizer
from agent.tools import get_ollama_tool_list
from agent.tools.base import ToolDict
from llm.llama_server import LLAMA_BASE_URL, LLAMA_COMPLETION_URL
from message_types import PreparedLLMMessage

LLAMA_APPLY_TEMPLATE_URL = f"{LLAMA_BASE_URL}/apply-template"
LLAMA_TOKENIZE_URL = f"{LLAMA_BASE_URL}/tokenize"

router = APIRouter()

logger = logging.getLogger(__name__)


class TokenVisualizerTurnRequest(BaseModel):
    # Pydantic validates each entry against PreparedLLMMessage (role/content required,
    # tool_calls/tool_call_id optional); a legacy field like "name" the debug page may still send
    # on a simulated tool-role message is silently dropped, not rejected — harmless since
    # llama.cpp's own parser accepts its absence and this project's chat template never renders it.
    history: list[PreparedLLMMessage]
    message: str
    system_prompt: str | None = None
    tool_names: list[str] = []


class TokenVisualizerInsertRequest(BaseModel):
    history: list[PreparedLLMMessage]
    messages: list[PreparedLLMMessage]
    system_prompt: str | None = None
    tool_names: list[str] = []


def _ndjson_event(event: dict) -> bytes:
    """Encode one event as a newline-delimited JSON line for streaming to the frontend."""
    return (json.dumps(event) + "\n").encode()


async def _apply_template(
    http: aiohttp.ClientSession, messages: list[PreparedLLMMessage], add_generation_prompt: bool, tools: list[ToolDict]
) -> str:
    """Render a message list through llama-server's own chat template."""
    body: dict[str, Any] = {"messages": messages, "add_generation_prompt": add_generation_prompt}
    if len(tools) > 0:
        body["tools"] = tools
    async with http.post(LLAMA_APPLY_TEMPLATE_URL, json=body) as r:
        return (await r.json())["prompt"]


async def _tokenize_with_special_flags(
    http: aiohttp.ClientSession, text: str, special_token_ids: set[int]
) -> list[dict]:
    """Tokenize text via llama-server's own /tokenize, annotating each token with whether
    its id is marked CONTROL/USER_DEFINED in the model's own GGUF vocab metadata."""
    async with http.post(LLAMA_TOKENIZE_URL, json={"content": text, "with_pieces": True}) as r:
        raw_tokens = (await r.json())["tokens"]

    return [
        {"id": raw_token["id"], "piece": raw_token["piece"], "special": raw_token["id"] in special_token_ids}
        for raw_token in raw_tokens
    ]


def _effective_system_prompt(system_prompt: str | None, tools: list[ToolDict]) -> str | None:
    """llama.cpp's chat template only renders tool definitions when visiting a system-role
    message slot. Without one, selected tools would silently vanish from the prompt even
    though they're still sent in the request body's `tools` field — so force an (empty)
    system message into existence whenever tools are selected, even with no textual prompt."""
    if system_prompt is None and len(tools) > 0:
        return ""
    return system_prompt


def _with_system_prompt(system_prompt: str | None, history: list[PreparedLLMMessage]) -> list[PreparedLLMMessage]:
    if system_prompt is None:
        return history
    system_message: PreparedLLMMessage = {"role": "system", "content": system_prompt}
    return [system_message] + history


# llama.cpp's chat template has two conditions that make calling /apply-template with
# add_generation_prompt=False directly unsafe:
#  1. The template hard-requires at least one role="user" message anywhere in the list — a
#     system-only (or otherwise userless) list is rejected outright ("No user query found").
#  2. When the LAST message is 'assistant', it applies a "continue this turn" special case:
#     injects a bogus empty <think></think> and omits the closing <|im_end|> — treating it as
#     an unfinished turn to continue, not a complete one to build on.
# Both make such a render useless as a "here's everything already shown" baseline for
# string-diffing. Confirmed directly against the live template (not a guess): a plain user-role
# message renders identically as a trailing suffix regardless of what precedes it (with/without
# history, system prompt, or tools), so appending one and trimming its own isolated length
# recovers the correctly-closed rendering in both cases.
_BOUNDARY_PROBE_MESSAGE: PreparedLLMMessage = {"role": "user", "content": " TOKEN_VISUALIZER_PROBE "}


async def _render_closed(http: aiohttp.ClientSession, messages: list[PreparedLLMMessage], tools: list[ToolDict]) -> str:
    """Render `messages` as a complete, closed prompt via llama.cpp's own template — safe even
    when the last message is 'assistant' or there's no user message at all, unlike calling
    /apply-template on it directly."""
    if len(messages) == 0:
        return ""
    has_user_message = any(m.get("role") == "user" for m in messages)
    if messages[-1]["role"] != "assistant" and has_user_message:
        return await _apply_template(http, messages, False, tools)
    probe_alone = await _apply_template(http, [_BOUNDARY_PROBE_MESSAGE], False, [])
    probe_with_prefix = await _apply_template(http, messages + [_BOUNDARY_PROBE_MESSAGE], False, tools)
    return probe_with_prefix[: len(probe_with_prefix) - len(probe_alone)]


@router.post("/api/token-visualizer/turn")
async def token_visualizer_turn(request: TokenVisualizerTurnRequest):
    """Tokenize the new user turn's templated text and generate + tokenize the assistant reply,
    all via llama-server's own /apply-template, /tokenize, and /completion endpoints. Streams
    newline-delimited JSON events to the frontend as each assistant token is generated."""
    history = request.history
    user_message: PreparedLLMMessage = {"role": "user", "content": request.message}
    tools = get_ollama_tool_list(request.tool_names)
    special_token_ids = tokenizer.get_special_token_ids()
    effective_system_prompt = _effective_system_prompt(request.system_prompt, tools)

    async def event_stream():
        async with aiohttp.ClientSession() as http:
            full_history = _with_system_prompt(effective_system_prompt, history)

            all_prefix_tokens: list[dict] = []

            if len(history) == 0 and effective_system_prompt is not None:
                # Can't render [system] alone — llama.cpp's template requires a user message
                # to be present anywhere in the list, so a system-only list is rejected outright.
                system_prompt_rendered = await _render_closed(http, full_history, tools)
                system_tokens = await _tokenize_with_special_flags(http, system_prompt_rendered, special_token_ids)
                yield _ndjson_event({"type": "system_tokens", "tokens": system_tokens})
                all_prefix_tokens += system_tokens

            # The new user turn's own rendering never depends on what precedes it (confirmed
            # against the live template) — tokenizing it in isolation sidesteps llama.cpp's
            # "continue this turn" special case entirely, rather than trying to diff it out.
            user_delta = await _apply_template(http, [user_message], False, [])
            user_tokens = await _tokenize_with_special_flags(http, user_delta, special_token_ids)
            yield _ndjson_event({"type": "user_tokens", "tokens": user_tokens})
            all_prefix_tokens += user_tokens

            # Rendered without the generation prompt: ends right after the user's own turn,
            # with no assistant-preamble tokens (<|im_start|>assistant, <think>, ...) attached.
            # Always safe — this message list's last entry is the user message we just added.
            prompt_user_only = await _apply_template(http, full_history + [user_message], False, tools)
            # Rendered with the generation prompt: the assistant-preamble tokens are appended.
            prompt_after_user = await _apply_template(http, full_history + [user_message], True, tools)

            assistant_preamble = prompt_after_user[len(prompt_user_only):]

            assistant_preamble_tokens = await _tokenize_with_special_flags(http, assistant_preamble, special_token_ids)
            yield _ndjson_event({"type": "assistant_preamble_tokens", "tokens": assistant_preamble_tokens})
            all_prefix_tokens += assistant_preamble_tokens

            # Any control token we've already identified as special in this turn's own
            # template rendering (<|im_start|>, <|im_end|>, <think>, ...) — preserving these
            # makes llama-server include their real piece text in the generation stream
            # instead of the empty string it sends for special tokens by default.
            preserved_tokens = sorted({
                token["piece"]
                for token in all_prefix_tokens
                if token["special"] and isinstance(token["piece"], str)
            })

            completion_body: dict[str, Any] = {
                "prompt": prompt_after_user,
                "temperature": 0.7,
                "cache_prompt": True,
                "stream": True,
                "return_tokens": True,
                "preserved_tokens": preserved_tokens,
            }
            if len(tools) > 0:
                completion_body["tools"] = tools
            async with http.post(LLAMA_COMPLETION_URL, json=completion_body) as r:
                if r.status != 200:
                    logger.error("llama-server token-visualizer completion failed: %s", await r.text())
                else:
                    async for line_bytes in r.content:
                        line = line_bytes.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            chunk = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue

                        content_fragment = chunk.get("content", "")
                        token_ids = chunk.get("tokens") or []
                        if content_fragment and len(token_ids) > 0:
                            token_id = token_ids[0]
                            yield _ndjson_event({
                                "type": "assistant_token",
                                "id": token_id,
                                "piece": content_fragment,
                                "special": token_id in special_token_ids,
                            })

            yield _ndjson_event({"type": "done"})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/api/token-visualizer/insert-messages")
async def token_visualizer_insert_messages(request: TokenVisualizerInsertRequest):
    """Tokenize an arbitrary set of new messages (e.g. a simulated tool_call/tool_result pair)
    appended to the given history, without triggering any generation. Used to let the debug page
    show how llama.cpp's own template renders tool-calling turns."""
    history = request.history
    tools = get_ollama_tool_list(request.tool_names)
    special_token_ids = tokenizer.get_special_token_ids()
    effective_system_prompt = _effective_system_prompt(request.system_prompt, tools)

    async with aiohttp.ClientSession() as http:
        full_history = _with_system_prompt(effective_system_prompt, history)

        prompt_before = await _render_closed(http, full_history, tools)

        system_tokens = None
        if len(history) == 0 and effective_system_prompt is not None:
            system_tokens = await _tokenize_with_special_flags(http, prompt_before, special_token_ids)

        prompt_after = await _render_closed(http, full_history + request.messages, tools)
        inserted_delta = prompt_after[len(prompt_before):]
        inserted_tokens = await _tokenize_with_special_flags(http, inserted_delta, special_token_ids)

    return {"system_tokens": system_tokens, "tokens": inserted_tokens}
