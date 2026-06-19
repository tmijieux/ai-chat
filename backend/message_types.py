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
    """A message in the conversation history / wire format sent to the LLM.
    thinking is NotRequired because user/tool/system messages don't have it;
    assistant messages that were appended from a generation may carry it,
    but prepare_messages strips it before sending to the server."""
    role: str
    content: str | list[dict]
    thinking: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]


class AssistantMessage(TypedDict):
    """Accumulated output from one LLM generation stream.
    Both content and thinking are always present (initialized to empty string)."""
    role: str
    content: str
    thinking: str
    tool_calls: NotRequired[list[ToolCall]]


class TrackedMessage(LLMMessage):
    """An LLMMessage augmented with an id field for tracking inside the compression pipeline."""
    id: str
