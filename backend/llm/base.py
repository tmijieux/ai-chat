import re
from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence, TypedDict, Literal, Union

from message_types import LLMMessage


class ContentEvent(TypedDict):
    type: Literal["content"]
    content: str


class ThinkingEvent(TypedDict):
    type: Literal["thinking"]
    content: str


class ToolCallStartEvent(TypedDict):
    type: Literal["tool_call_start"]
    index: int
    id: str
    name: str


class ToolCallArgEvent(TypedDict):
    type: Literal["tool_call_arg"]
    index: int
    fragment: str


class ToolCallRawEvent(TypedDict):
    """Cosmetic, live preview of a tool call's raw generated text (native <function=.../> XML,
    not JSON). Purely for display while streaming — the authoritative, schema-cast arguments
    still arrive afterward via ToolCallStartEvent/ToolCallArgEvent, unaffected by this."""
    type: Literal["tool_call_raw"]
    fragment: str


class DoneEvent(TypedDict):
    type: Literal["done"]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


StreamEvent = Union[ContentEvent, ThinkingEvent, ToolCallStartEvent, ToolCallArgEvent, ToolCallRawEvent, DoneEvent]


# Qwen3's tool-call XML, shared by two callers: agent.py recovers a call the model emitted
# inside an unclosed <think> block, and llama_server.py's forced-tool-call fallback parses a
# grammar-constrained /completion reply. Same wire format both times: <tool_call><function=name>
# <parameter=x>raw value</parameter></function></tool_call>, repeated for multiple calls.
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_embedded_tool_call(block_content: str) -> dict | None:
    """Parse <function=NAME><parameter=K>V</parameter>...</function> into name + raw-string arguments."""
    func_match = _FUNCTION_RE.search(block_content)
    if func_match is None:
        return None
    name = func_match.group(1)
    func_body = func_match.group(2)
    arguments = {m.group(1): m.group(2) for m in _PARAMETER_RE.finditer(func_body)}
    return {"name": name, "arguments": arguments}


def parse_all_tool_calls(text: str) -> list[dict]:
    """Parse every <tool_call>...</tool_call> block in text into {"name", "arguments"} dicts.

    Arguments are raw strings — callers that have the tool's JSON Schema on hand may cast them
    to the declared type (integer/number/boolean); callers that don't can pass them through as-is.
    """
    calls = []
    for block_match in TOOL_CALL_BLOCK_RE.finditer(text):
        parsed = parse_embedded_tool_call(block_match.group(1))
        if parsed is not None and parsed["name"] != "":
            calls.append(parsed)
    return calls


class ThinkingParser:
    """Incremental <think>...</think> extractor, safe across chunk boundaries."""

    def __init__(self) -> None:
        self.in_think: bool = False
        self._carry: str = ""

    def feed(self, fragment: str) -> tuple[str, str]:
        """Returns (thinking_text, content_text) extracted from this fragment."""
        remaining = self._carry + fragment
        self._carry = ""
        thinking_out = ""
        content_out = ""

        while remaining:
            if self.in_think:
                end = remaining.find("</think>")
                if end == -1:
                    # Keep last 8 chars as carry — </think> may be split across chunks
                    safe_len = max(0, len(remaining) - 8)
                    thinking_out += remaining[:safe_len]
                    self._carry = remaining[safe_len:]
                    break
                thinking_out += remaining[:end]
                remaining = remaining[end + 8:]
                self.in_think = False
            else:
                start = remaining.find("<think>")
                if start == -1:
                    content_out += remaining
                    break
                content_out += remaining[:start]
                remaining = remaining[start + 7:]
                self.in_think = True

        return thinking_out, content_out

    def flush(self) -> tuple[str, str]:
        """Emit any buffered carry after the stream ends."""
        remaining = self._carry
        self._carry = ""
        if not remaining:
            return "", ""
        return (remaining, "") if self.in_think else ("", remaining)


class LLMBackend(ABC):

    @abstractmethod
    async def ensure_running(self) -> None: ...

    @abstractmethod
    async def check_or_raise(self) -> None: ...

    @abstractmethod
    async def count_tokens(self, messages: Sequence[LLMMessage], tools: list) -> int: ...

    @abstractmethod
    async def count_text_tokens(self, text: str) -> int: ...

    @abstractmethod
    async def stream_completion(
        self,
        messages: Sequence[LLMMessage],
        tools: list,
        temperature: float,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        tool_choice: dict | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
        yield  # type: ignore[misc]

    def prepare_messages(self, messages: Sequence[LLMMessage]) -> Sequence[LLMMessage]:
        """Convert internal message format to whatever this backend expects on the wire."""
        return messages
