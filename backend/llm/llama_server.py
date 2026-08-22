import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import AsyncIterator, Sequence

import aiohttp
from fastapi import HTTPException

from tokenizer import render_messages
from message_types import LLMMessage, PreparedLLMMessage, PreparedMessages, ToolCall, PreparedToolCall
from .base import (
    LLMBackend, StreamEvent, ThinkingParser,
    ContentEvent, ThinkingEvent, ToolCallStartEvent, ToolCallArgEvent, ToolCallRawEvent, DoneEvent,
    parse_all_tool_calls,
)

MODEL_NAME = "local"
LLAMA_BASE_URL = "http://127.0.0.1:8080"
LLAMA_CHAT_URL = f"{LLAMA_BASE_URL}/v1/chat/completions"
LLAMA_COMPLETION_URL = f"{LLAMA_BASE_URL}/completion"
LLAMA_TOKENIZE_URL = f"{LLAMA_BASE_URL}/tokenize"
LLAMA_HEALTH_URL = f"{LLAMA_BASE_URL}/health"
LLAMA_SERVER_EXE = str(Path.home() / "ai/llama.cpp/build/bin/Release/llama-server.exe")
GGUF_PATH = str(Path.home() / "ai/models/unsloth/Qwen3.5-9B-Q4_K_M.gguf")
MMPROJ_PATH = str(Path.home() / "ai/models/unsloth/mmproj-F16.gguf")
CTX_LIMIT = 2**15 # 14 -> 16K, 15 -> 32K, 16 -> 65k

# Flip to False to instantly revert every tool-enabled turn to the pre-think-gate behavior
# (_stream_completion_legacy) with no other code change, if the think-gated path misbehaves.
USE_THINK_GATED_STREAMING = True

# The special <tool_call> token's textual form, as it appears in raw /completion output — used
# to split phase 2's content stream live into prose (ContentEvent) vs. the tool call's raw text
# (ToolCallRawEvent), without waiting for the whole tail to finish generating.
_TOOL_CALL_OPEN_TAG = "<tool_call>"
_FUNCTION_NAME_RE = re.compile(r"<function=(\w+)>")

logger = logging.getLogger(__name__)

# The think-block sub-grammar llama-server itself generates for eager tool-call grammars (copied
# verbatim from a captured __verbose.generation_settings.grammar) — fixed boilerplate, the same
# regardless of which tools/schema are involved, unlike the tool-call rules below it which are
# schema-dependent and always taken fresh from the server.
_THINK_BLOCK_RULE = (
    'think-block ::= "<think>" "\\n"? ([^<] | "<" [^/] | "</" [^t] | "</t" [^h] | "</th" [^i] '
    '| "</thi" [^n] | "</thin" [^k] | "</think" [^>])* "\\n"? "</think>" "\\n"? "\\n"?'
)


def _to_prepared_tool_call(tc: ToolCall, index: int) -> PreparedToolCall:
    """Build the OpenAI wire-format tool call llama.cpp's server requires: `type` is mandatory
    (its parser throws if absent from message history), and arguments must be the JSON-encoded
    string OpenAI's wire format uses, not the internal dict representation."""
    arguments = tc["function"]["arguments"]
    return {
        "id": tc.get("id", f"tc-{index}"),
        "type": "function",
        "function": {
            "name": tc["function"]["name"],
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
        },
    }


def _cast_tool_call_arguments(parsed_calls: list[dict], tools_by_name: dict[str, dict]) -> list[tuple[str, dict]]:
    """Cast a parsed tool call's raw-string arguments to their declared JSON-Schema type.

    parse_all_tool_calls (llm/base.py) only knows the XML wire format, not any particular tool's
    schema, so it always returns raw strings; this casts them the way the normal
    /v1/chat/completions path's server-side argument conversion would have.
    """
    calls: list[tuple[str, dict]] = []
    for parsed in parsed_calls:
        name = parsed["name"]
        properties = tools_by_name.get(name, {}).get("parameters", {}).get("properties", {})
        arguments: dict = {}
        for param_name, raw_value in parsed["arguments"].items():
            param_type = properties.get(param_name, {}).get("type", "string")
            if param_type == "integer":
                arguments[param_name] = int(raw_value.strip())
            elif param_type == "number":
                arguments[param_name] = float(raw_value.strip())
            elif param_type == "boolean":
                arguments[param_name] = raw_value.strip() == "true"
            elif param_type in ("array", "object"):
                arguments[param_name] = json.loads(raw_value)
            else:
                arguments[param_name] = raw_value
        calls.append((name, arguments))
    return calls


class LlamaServerBackend(LLMBackend):

    async def ensure_running(self) -> None:
        async with aiohttp.ClientSession() as http:
            try:
                async with http.get(LLAMA_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        logger.info("llama-server already running.")
                        return
            except Exception:
                pass

        logger.info("llama-server not detected — launching ...")
        p = subprocess.Popen(
            [
                LLAMA_SERVER_EXE,
                "-m", GGUF_PATH,
                "--mmproj", MMPROJ_PATH,
                "-c", str(CTX_LIMIT),
                "-ngl", "99",
                "--port", "8080",
                "--host", "127.0.0.1",
                # verbosity>9 makes llama-server attach a "__verbose" block (rendered prompt +
                # grammar) to /v1/chat/completions responses — used by _force_tool_call below.
                # --log-disable stops that verbosity from producing any log output.
                "--verbosity", "10",
                "--log-disable",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        

        async with aiohttp.ClientSession() as http:
            for _ in range(120):  # 60s — model loading takes time
                ret = p.poll() 
                if ret is not None:
                    output = p.stdout.read().decode(errors="replace") if p.stdout else ""
                    logger.error("llama-server exited with code %s:\n%s", ret, output)
                    break
                await asyncio.sleep(0.5)
                try:
                    async with http.get(LLAMA_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=1)) as r:
                        if r.status == 200:
                            logger.info("llama-server started successfully.")
                            return
                except Exception:
                    pass

        logger.warning("llama-server did not respond within 60s — continuing anyway.")

    async def check_or_raise(self) -> None:
        async with aiohttp.ClientSession() as http:
            try:
                async with http.get(LLAMA_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
        raise HTTPException(status_code=503, detail="llama-server is not running")

    async def count_tokens(self, messages: PreparedMessages, tools: list) -> int:
        rendered = render_messages(messages, tools)
        return await self.count_text_tokens(rendered)

    async def count_text_tokens(self, text: str) -> int:
        """Count tokens for raw text, bypassing the chat template."""
        async with aiohttp.ClientSession() as http:
            async with http.post(
                LLAMA_TOKENIZE_URL,
                json={"content": text},
            ) as r:
                data = await r.json()
                return len(data["tokens"])

    def prepare_messages(self, messages: Sequence[LLMMessage]) -> Sequence[PreparedLLMMessage]:
        """Convert internal format to OpenAI wire format for llama-server."""
        result: list[PreparedLLMMessage] = []
        for m in messages:
            msg: PreparedLLMMessage = {"role": m["role"], "content": m.get("content", "")}

            if "tool_calls" in m:
                msg["tool_calls"] = [_to_prepared_tool_call(tc, i) for i, tc in enumerate(m["tool_calls"])]

            if m["role"] == "tool":
                # OpenAI requires tool_call_id as a top-level field on tool messages. A caller
                # with plain-text tool content sets LLMMessage.tool_call_id directly; an ordinary
                # tool result instead carries it embedded in its JSON content (ToolResult.tool_call_id).
                if "tool_call_id" in m:
                    msg["tool_call_id"] = m["tool_call_id"]
                else:
                    try:
                        raw = m.get("content")
                        if not isinstance(raw, str):
                            raise ValueError(f"tool message has non-string content: {type(raw)}")
                        content_data = json.loads(raw)
                        tool_call_id = content_data.get("tool_call_id")
                        if tool_call_id:
                            msg["tool_call_id"] = tool_call_id
                    except (json.JSONDecodeError, ValueError):
                        pass

            result.append(msg)
        return result

    async def stream_completion(
        self,
        messages: PreparedMessages,
        tools: list,
        temperature: float,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        tool_choice: dict | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Dispatch to the think-gated path when it applies, else the legacy OAI-streaming path.

        The think-gated path only helps when there's an unclosed <think> block a <tool_call>
        could hide in and tools that could actually be called — so it's skipped whenever there
        are no tools, thinking is disabled (disable_thinking's prompt already contains a
        pre-closed empty <think></think>, confirmed via a live probe, so there's nothing to
        gate), or tool_choice=="none" (no tool-calling outcome to guard against).
        """
        use_think_gate = (
            USE_THINK_GATED_STREAMING
            and len(tools) > 0
            and disable_thinking == False
            and tool_choice != "none"
        )
        if use_think_gate:
            async for event in self._stream_completion_think_gated(messages, tools, temperature, max_tokens, tool_choice):
                yield event
            return

        async for event in self._stream_completion_legacy(messages, tools, temperature, max_tokens, disable_thinking, tool_choice):
            yield event

    async def _stream_completion_legacy(
        self,
        messages: PreparedMessages,
        tools: list,
        temperature: float,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        tool_choice: dict | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        body: dict = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "presence_penalty": 1.5,
            "top_k": 20,
            "top_p": 0.95,
        }
        if tools:
            body["tools"] = tools
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        tool_calls_acc: dict[int, dict] = {}  # index → {id, name, arguments_str}
        finish_reason: str = "stop"
        prompt_tokens: int = 0
        completion_tokens: int = 0
        thinking_acc: str = ""

        #print(json.dumps(body, indent=2, ensure_ascii=False))
        async with aiohttp.ClientSession() as http:
            async with http.post(LLAMA_CHAT_URL, json=body) as response:
                if response.status != 200:
                    body_text = await response.text()
                    logger.error("llama-server error %s: %s", response.status, body_text)
                    return

                async for line_bytes in response.content:
                    line = line_bytes.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                        timings = chunk.get("timings") or {}
                        prompt_tokens = timings.get("prompt_n", 0)
                        completion_tokens = timings.get("predicted_n", 0)

                    # Thinking — llama-server exposes it in reasoning_content, not <think> tags
                    thinking_frag = delta.get("reasoning_content") or ""
                    if thinking_frag:
                        thinking_acc += thinking_frag
                        yield ThinkingEvent(type="thinking", content=thinking_frag)

                    # Content
                    content_frag = delta.get("content") or ""
                    if content_frag:
                        yield ContentEvent(type="content", content=content_frag)

                    # Tool calls — fragmented across chunks, keyed by index
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_calls_acc:
                            tc_id = tc_delta.get("id", f"tc-{idx}")
                            name = (tc_delta.get("function") or {}).get("name", "")
                            tool_calls_acc[idx] = {"id": tc_id, "name": name, "arguments_str": ""}
                            yield ToolCallStartEvent(type="tool_call_start", index=idx, id=tc_id, name=name)

                        args_frag = (tc_delta.get("function") or {}).get("arguments") or ""
                        if args_frag:
                            tool_calls_acc[idx]["arguments_str"] += args_frag
                            yield ToolCallArgEvent(type="tool_call_arg", index=idx, fragment=args_frag)

        if tool_choice is not None and len(tool_calls_acc) == 0:
            # llama-server's own parser doesn't recognize a call the model made without ever
            # closing </think> — check for that (same recovery agent.py does post-hoc) before
            # concluding tool_choice was genuinely ignored and paying for a forced reissue.
            recovered = _cast_tool_call_arguments(parse_all_tool_calls(thinking_acc), {t["function"]["name"]: t["function"] for t in tools})
            if len(recovered) > 0:
                for idx, (name, arguments) in enumerate(recovered):
                    tc_id = f"tc-recovered-{idx}"
                    yield ToolCallStartEvent(type="tool_call_start", index=idx, id=tc_id, name=name)
                    yield ToolCallArgEvent(type="tool_call_arg", index=idx, fragment=json.dumps(arguments, ensure_ascii=False))
                yield DoneEvent(type="done", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason="tool_calls")
                return

            logger.warning(
                "llama-server did not honor tool_choice=%r — retrying with a mechanically forced grammar",
                tool_choice,
            )
            async for event in self._force_tool_call(messages, tools, tool_choice, max_tokens, temperature):
                yield event
            return

        yield DoneEvent(
            type="done",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    async def _stream_completion_think_gated(
        self,
        messages: PreparedMessages,
        tools: list,
        temperature: float,
        max_tokens: int | None,
        tool_choice: dict | str | None,
    ) -> AsyncIterator[StreamEvent]:
        """Two-phase generation that structurally prevents <tool_call> from being emitted before
        </think> closes, instead of recovering from it after the fact like agent.py's regex-based
        extraction (still needed for the legacy path).

        Phase 1 runs an eager grammar that forbids only the special <tool_call> token (a distinct
        vocab entry) and stops at </think> — this blocks real tool-call attempts without blocking
        the model from writing "<tool_call>" as ordinary text, since ordinary tokens spelling the
        same characters aren't the banned token (verified empirically against a live server:
        forcing the special token via logit_bias still gets rejected pre-</think>, while a forced
        prefix ending mid-"<tool_call>" text completes normally via an ordinary token).

        Phase 2 continues generation from the same llama-server slot (cache_prompt reuses phase
        1's KV cache, so this isn't a full prompt reprocess) with a grammar requiring either a
        real, schema-validated tool call or free prose — using the same token-identity exclusion
        for the prose branch, since a bare `.*` alternative there would silently void the schema
        rule's validation (any text the schema rule could produce, `.*` could produce identically,
        so the ever-permissive `.*` branch would make the schema branch's constraints moot).
        """
        n_predict = 1024 if max_tokens is None else max(max_tokens, 1024)

        async with aiohttp.ClientSession() as http:
            probe_body = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "max_tokens": 1,
                "temperature": temperature,
                "tools": tools,
                "tool_choice": "auto" if tool_choice is None else tool_choice,
            }
            async with http.post(LLAMA_CHAT_URL, json=probe_body) as response:
                if response.status != 200:
                    logger.error("llama-server think-gate probe failed: %s", await response.text())
                    yield DoneEvent(type="done", prompt_tokens=0, completion_tokens=0, finish_reason="error")
                    return
                probe = await response.json()

            verbose = probe.get("__verbose")
            if verbose is None:
                logger.error(
                    "llama-server probe response had no __verbose block (server not started with "
                    "--verbosity 10+) — cannot run the think-gated streaming path"
                )
                yield DoneEvent(type="done", prompt_tokens=0, completion_tokens=0, finish_reason="error")
                return

            prompt = verbose["prompt"]
            grammar_lines = [
                line for line in verbose["generation_settings"]["grammar"].splitlines()
                if not line.startswith("root ::=")
            ]

            # Phase 1: think, with the special <tool_call> token banned outright.
            phase1_body = {
                "prompt": prompt,
                "grammar": "root ::= !<tool_call>*\n",
                "stop": ["</think>"],
                "n_predict": n_predict,
                "temperature": temperature,
                "presence_penalty": 1.5,
                "top_k": 20,
                "top_p": 0.95,
                "cache_prompt": True,
                "stream": True,
            }
            thinking_acc = ""
            prompt_tokens = 0
            completion_tokens = 0
            stopped_on_think_close = False
            async with http.post(LLAMA_COMPLETION_URL, json=phase1_body) as response:
                if response.status != 200:
                    logger.error("llama-server think-gate phase 1 failed: %s", await response.text())
                    yield DoneEvent(type="done", prompt_tokens=0, completion_tokens=0, finish_reason="error")
                    return

                async for line_bytes in response.content:
                    line = line_bytes.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        chunk = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    fragment = chunk.get("content", "")
                    if fragment:
                        thinking_acc += fragment
                        yield ThinkingEvent(type="thinking", content=fragment)

                    if chunk.get("stop"):
                        prompt_tokens = chunk.get("tokens_evaluated", 0)
                        completion_tokens = chunk.get("tokens_predicted", 0)
                        stopped_on_think_close = chunk.get("stopping_word") == "</think>"

        if stopped_on_think_close == False:
            logger.error(
                "think-gate phase 1 did not close </think> (likely spent its full %d-token "
                "budget still reasoning): %r", n_predict, thinking_acc,
            )
            yield DoneEvent(type="done", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason="error")
            return

        # Phase 2: think is closed — free prose, optionally followed by a real, schema-validated
        # tool call. This has to be a *sequence* (prose, then optionally tool-call), not a
        # top-level "tool-call | prose" choice: alternation picks a branch once, at the very
        # first phase-2 token, and the model's first tokens are near-always some lead-in prose
        # ("I'll check the weather..."), which would permanently prune the tool-call branch (its
        # literal must start immediately) and strand generation in prose — forbidden from ever
        # using the real token again, so it hallucinates fake pseudo-call syntax instead of a real
        # call. Confirmed both ways empirically: "tool-call | prose" reproduced this 5/5 runs;
        # "prose tool-call?" does not, because the star's lazy exit (same ambiguity that makes the
        # phase-1/phase-2 boundary work) keeps a "the loop already ended, tool-call starts here"
        # derivation alive at every position, not just position 0.
        # Uses a fresh session/connection rather than reusing phase 1's: llama-server closes the
        # connection right after a stop-triggered stream ends, and reusing the pooled connection
        # for the very next request races that close (observed empirically as a
        # ServerDisconnectedError when both requests shared one aiohttp.ClientSession).
        if tool_choice is None or tool_choice == "auto":
            grammar = "\n".join(["root ::= prose tool-call?", "prose ::= !<tool_call>*"] + grammar_lines)
        else:
            # No "prose" prefix here, unlike the optional case above: an unbounded prose prefix
            # in front of a *mandatory* call gives the model room to stall forever when it has no
            # real reason to invoke the forced tool (confirmed empirically: it ran out its full
            # token budget rambling instead of ever committing). Mirrors _force_tool_call's
            # existing mandatory root exactly — the tail must start immediately with the literal
            # tag, no lead-in room to avoid it.
            grammar = "\n".join(["root ::= tool-call"] + grammar_lines)

        # "stop" strings are excluded from the returned content (confirmed empirically), so
        # the closing tag has to be reattached by hand to keep the replayed prompt accurate.
        phase2_prompt = prompt + thinking_acc + "</think>"
        phase2_body = {
            "prompt": phase2_prompt,
            "grammar": grammar,
            "n_predict": n_predict,
            "temperature": temperature,
            "presence_penalty": 1.5,
            "top_k": 20,
            "top_p": 0.95,
            "cache_prompt": True,
            "stream": True,
        }
        thinking_parser = ThinkingParser()
        tail_text = ""
        # Splits phase 2's content stream live into prose (ContentEvent) and, once the special
        # <tool_call> token's text is seen, a cosmetic live preview of the tool call's raw
        # generated text — the display-only ToolCallRawEvent, gated behind the frontend already
        # having a ToolCallStartEvent (tool name) to attach it to. So this waits for
        # "<function=NAME>" to close before emitting anything for the tool-call side: the buffer
        # between the two markers is tiny (the model always writes it immediately), and this way
        # the frontend never has to render a preview with no name. tool_call_carry mirrors
        # ThinkingParser's own carry technique for markers split across chunk boundaries.
        tool_call_carry = ""
        in_tool_call = False
        tool_call_name_known = False
        tool_call_name_buffer = ""

        def split_prose_and_tool_call_live(text: str):
            nonlocal tool_call_carry, in_tool_call, tool_call_name_known, tool_call_name_buffer
            if in_tool_call:
                if tool_call_name_known:
                    if text:
                        yield ToolCallRawEvent(type="tool_call_raw", fragment=text)
                    return
                tool_call_name_buffer += text
                name_match = _FUNCTION_NAME_RE.search(tool_call_name_buffer)
                if name_match is not None:
                    tool_call_name_known = True
                    yield ToolCallStartEvent(type="tool_call_start", index=0, id="tc-0", name=name_match.group(1))
                    yield ToolCallRawEvent(type="tool_call_raw", fragment=tool_call_name_buffer)
                    tool_call_name_buffer = ""
                return
            buf = tool_call_carry + text
            marker_pos = buf.find(_TOOL_CALL_OPEN_TAG)
            if marker_pos == -1:
                safe_len = max(0, len(buf) - (len(_TOOL_CALL_OPEN_TAG) - 1))
                if buf[:safe_len]:
                    yield ContentEvent(type="content", content=buf[:safe_len])
                tool_call_carry = buf[safe_len:]
            else:
                if buf[:marker_pos]:
                    yield ContentEvent(type="content", content=buf[:marker_pos])
                tool_call_carry = ""
                in_tool_call = True
                yield from split_prose_and_tool_call_live(buf[marker_pos:])

        async with aiohttp.ClientSession() as http2:
            async with http2.post(LLAMA_COMPLETION_URL, json=phase2_body) as response:
                if response.status != 200:
                    logger.error("llama-server think-gate phase 2 failed: %s", await response.text())
                    yield DoneEvent(type="done", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason="error")
                    return

                async for line_bytes in response.content:
                    line = line_bytes.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        chunk = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    thinking_frag, content_frag = thinking_parser.feed(chunk.get("content", ""))
                    if thinking_frag:
                        yield ThinkingEvent(type="thinking", content=thinking_frag)
                    tail_text += content_frag
                    for live_event in split_prose_and_tool_call_live(content_frag):
                        yield live_event

                    if chunk.get("stop"):
                        prompt_tokens = chunk.get("tokens_evaluated", 0)
                        completion_tokens = chunk.get("tokens_predicted", 0)

            trailing_thinking, trailing_content = thinking_parser.flush()
            if trailing_thinking:
                yield ThinkingEvent(type="thinking", content=trailing_thinking)
            tail_text += trailing_content
            for live_event in split_prose_and_tool_call_live(trailing_content):
                yield live_event

        tools_by_name = {t["function"]["name"]: t["function"] for t in tools}
        parsed_calls = _cast_tool_call_arguments(parse_all_tool_calls(tail_text), tools_by_name)

        if len(parsed_calls) > 0:
            for idx, (name, arguments) in enumerate(parsed_calls):
                tc_id = f"tc-{idx}"
                # The first call's tool_call_start may have already been sent live (see
                # tool_call_name_known above), once its name became known — this re-sends the
                # same (index, id, name) for it rather than skip it, since consumers key on
                # tool_id and treat a repeated start as "reset this call's entry to a clean
                # slate", which is exactly what should happen right before the final,
                # authoritative argument fragment replaces the raw live preview.
                yield ToolCallStartEvent(type="tool_call_start", index=idx, id=tc_id, name=name)
                yield ToolCallArgEvent(type="tool_call_arg", index=idx, fragment=json.dumps(arguments, ensure_ascii=False))
            yield DoneEvent(type="done", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason="tool_calls")
            return

        mandatory_call = tool_choice is not None and tool_choice != "auto"
        if mandatory_call:
            logger.error(
                "think-gate phase 2 produced no parseable <tool_call> despite a mandatory "
                "grammar (likely spent its full %d-token budget still reasoning): %r", n_predict, tail_text,
            )
            yield DoneEvent(type="done", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason="error")
            return

        # in_tool_call being True here (but with no parseable calls) means the marker appeared
        # yet parsing still found nothing — the raw text already streamed live as
        # ToolCallRawEvent above, so it must not be re-sent as content here too.
        if in_tool_call == False and len(tool_call_carry) > 0:
            yield ContentEvent(type="content", content=tool_call_carry)
        yield DoneEvent(type="done", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason="stop")

    async def _force_tool_call(
        self,
        messages: PreparedMessages,
        tools: list,
        tool_choice: dict | str,
        max_tokens: int | None,
        temperature: float,
    ) -> AsyncIterator[StreamEvent]:
        """Mechanically force a tool call llama-server's tool_choice wiring failed to produce.

        llama.cpp's PEG-native chat format (what Qwen3.5 uses) does not reliably turn tool_choice
        into an eager grammar: either the grammar stays lazy and the model never triggers it, or
        the eager grammar's root allows an unbounded run of ordinary content before the tool-call
        rule, which the model can stall in forever. Both were confirmed empirically against a live
        llama-server. The fix: probe /v1/chat/completions for the grammar and rendered prompt it
        would have used (max_tokens=1 keeps this cheap; requires llama-server run with verbosity>9,
        see ensure_running), rebuild the grammar's root as "think-block? tool-call" — reasoning is
        still allowed, but the only way out of it is a real tool call, no free-text escape hatch —
        and replay the prompt (minus its trailing open <think> tag, since the model reopens it
        itself if it wants to reason) against the raw /completion endpoint, streamed so thinking
        still displays live. The tool-call rules themselves come verbatim from llama-server's own
        grammar — reusing them instead of hand-building GBNF keeps this correct for enum/integer/
        boolean parameters too, not just the flat strings this was validated against.
        """
        # Own budget, decoupled from the caller's max_tokens: that budget was sized for a
        # different generation, and the model is now free to reason at length before the
        # mandatory call — a caller-supplied budget as low as auto_safety.py's 128 measurably
        # risks the model spending it all on reasoning and never reaching the tool call at all.
        n_predict = 1024 if max_tokens is None else max(max_tokens, 1024)

        async with aiohttp.ClientSession() as http:
            probe_body = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "max_tokens": 1,
                "temperature": temperature,
                "tools": tools,
                "tool_choice": tool_choice,
            }
            async with http.post(LLAMA_CHAT_URL, json=probe_body) as response:
                if response.status != 200:
                    logger.error("llama-server tool-call probe failed: %s", await response.text())
                    yield DoneEvent(type="done", prompt_tokens=0, completion_tokens=0, finish_reason="error")
                    return
                probe = await response.json()

            verbose = probe.get("__verbose")
            if verbose is None:
                logger.error(
                    "llama-server probe response had no __verbose block (server not started with "
                    "--verbosity 10+) — cannot mechanically force this tool call"
                )
                yield DoneEvent(type="done", prompt_tokens=0, completion_tokens=0, finish_reason="error")
                return

            prompt = verbose["prompt"]
            if prompt.endswith("<think>\n"):
                prompt = prompt[: -len("<think>\n")]

            grammar_lines = [
                line for line in verbose["generation_settings"]["grammar"].splitlines()
                if not line.startswith("root ::=")
            ]
            grammar = "\n".join(["root ::= think-block? tool-call", _THINK_BLOCK_RULE] + grammar_lines)

            completion_body = {
                "prompt": prompt,
                "grammar": grammar,
                "n_predict": n_predict,
                "temperature": temperature,
                "presence_penalty": 1.5,
                "top_k": 20,
                "top_p": 0.95,
                "cache_prompt": True,
                "stream": True,
            }
            thinking_parser = ThinkingParser()
            tool_call_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            async with http.post(LLAMA_COMPLETION_URL, json=completion_body) as response:
                if response.status != 200:
                    logger.error("llama-server forced tool-call completion failed: %s", await response.text())
                    yield DoneEvent(type="done", prompt_tokens=0, completion_tokens=0, finish_reason="error")
                    return

                async for line_bytes in response.content:
                    line = line_bytes.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        chunk = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    thinking_frag, content_frag = thinking_parser.feed(chunk.get("content", ""))
                    if thinking_frag:
                        yield ThinkingEvent(type="thinking", content=thinking_frag)
                    tool_call_text += content_frag

                    if chunk.get("stop"):
                        prompt_tokens = chunk.get("tokens_evaluated", 0)
                        completion_tokens = chunk.get("tokens_predicted", 0)

            trailing_thinking, trailing_content = thinking_parser.flush()
            if trailing_thinking:
                yield ThinkingEvent(type="thinking", content=trailing_thinking)
            tool_call_text += trailing_content

        tools_by_name = {t["function"]["name"]: t["function"] for t in tools}
        parsed_calls = _cast_tool_call_arguments(parse_all_tool_calls(tool_call_text), tools_by_name)
        if len(parsed_calls) == 0:
            logger.error(
                "forced tool-call completion produced no parseable <tool_call> (likely spent its "
                "full %d-token budget still reasoning): %r", n_predict, tool_call_text,
            )
            yield DoneEvent(type="done", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason="error")
            return

        for idx, (name, arguments) in enumerate(parsed_calls):
            tc_id = f"tc-{idx}"
            yield ToolCallStartEvent(type="tool_call_start", index=idx, id=tc_id, name=name)
            yield ToolCallArgEvent(type="tool_call_arg", index=idx, fragment=json.dumps(arguments, ensure_ascii=False))

        yield DoneEvent(
            type="done",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="tool_calls",
        )
