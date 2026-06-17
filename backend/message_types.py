from typing import Any, TypedDict, NotRequired


class ToolCallFunction(TypedDict):
    """The function payload inside a tool call."""
    name: str
    arguments: dict[str, Any]


class ToolCall(TypedDict):
    """A single tool call as produced by the LLM and consumed by the agent loop."""
    id: str
    function: ToolCallFunction
    _recovered: NotRequired[bool]


class LLMMessage(TypedDict):
    """A message in the conversation format exchanged with the LLM backend."""
    role: str
    content: str | list[dict]
    thinking: NotRequired[str]
    name: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]


class TrackedMessage(LLMMessage):
    """An LLMMessage augmented with an id field for tracking inside the compression pipeline."""
    id: NotRequired[str]
