from typing import TypedDict, NotRequired


class LLMMessage(TypedDict):
    """A message in the conversation format exchanged with the LLM backend."""
    role: str
    content: str | list[dict]
    thinking: NotRequired[str]
    name: NotRequired[str]
    tool_calls: NotRequired[list[dict]]


class TrackedMessage(LLMMessage):
    """An LLMMessage augmented with an id field for tracking inside the compression pipeline."""
    id: NotRequired[str]
