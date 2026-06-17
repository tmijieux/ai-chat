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


class DoneEvent(TypedDict):
    type: Literal["done"]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


StreamEvent = Union[ContentEvent, ThinkingEvent, ToolCallStartEvent, ToolCallArgEvent, DoneEvent]


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
