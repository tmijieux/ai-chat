from typing import Any, Literal, Sequence, TypedDict, NotRequired


class ToolCallFunction(TypedDict):
    """The function payload inside a tool call."""
    name: str
    arguments: dict[str, Any]


class ToolCall(TypedDict):
    """A single tool call as produced by the LLM and consumed by the agent loop."""
    id: str
    function: ToolCallFunction
    _recovered: NotRequired[bool]


class PreparedToolCallFunction(TypedDict):
    """The function payload inside a prepared tool call: unlike ToolCallFunction, arguments
    is the JSON-encoded string OpenAI's wire format requires, not the internal dict."""
    name: str
    arguments: str


class PreparedToolCall(TypedDict):
    """Prepared tool call shape for a message about to be sent to an OpenAI-compatible LLM
    endpoint. Unlike ToolCall: `type` is mandatory (llama.cpp's server throws if it's absent
    from message history), and internal-only markers such as ToolCall's _recovered must not be
    carried over onto the wire. Built from a ToolCall by each backend's prepare_messages."""
    id: str
    type: Literal["function"]
    function: PreparedToolCallFunction


class LLMMessage(TypedDict):
    """A message in the conversation history — the agent loop's internal representation,
    before a backend's prepare_messages converts it to that backend's wire format.

    tool_call_id is for a role="tool" message whose content is plain text rather than a JSON
    ToolResult envelope (an ordinary tool result instead carries it embedded in that JSON, via
    ToolResult.tool_call_id — see agent.py's tool dispatch loop). prepare_messages checks this
    top-level field first before falling back to extracting one from JSON content."""
    role: str
    content: str | list[dict]
    thinking: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]


class PreparedLLMMessage(TypedDict):
    """Prepared message shape returned by an OpenAI-compatible backend's prepare_messages —
    what's actually sent on the wire. tool_calls (if present) use PreparedToolCall, and
    thinking is never carried (prepare_messages strips it before sending)."""
    role: str

    content: str | list[dict] | None
    """The message's text (or multimodal parts). Nullable because that's the real OpenAI wire
    convention for a tool-only assistant turn (no natural-language text, only tool_calls) —
    distinct from content being an empty string. This project's own prepare_messages never emits
    None (uses "" instead, since nothing internal needs to tell the two cases apart), but
    llama.cpp's parser explicitly accepts null content as long as tool_calls is present (verified
    against its own source), and the token-visualizer debug page does send it."""

    tool_calls: NotRequired[list[PreparedToolCall]]
    tool_call_id: NotRequired[str]


class PreparedOllamaToolCall(TypedDict):
    """Prepared tool call for Ollama's native /api/chat endpoint: id + function only. Reuses
    ToolCallFunction as-is (Ollama keeps arguments as a dict, unlike the OpenAI wire format),
    but not ToolCall itself — its _recovered marker is internal-only and must not reach the
    wire, so OllamaBackend.prepare_messages rebuilds each entry through this shape rather than
    passing ToolCall objects through unchanged."""
    id: str
    function: ToolCallFunction


class PreparedOllamaMessage(TypedDict):
    """Prepared message shape returned by OllamaBackend.prepare_messages. Unlike
    PreparedLLMMessage: tool_calls use PreparedOllamaToolCall (no `type` field, arguments
    stays a dict — Ollama's own API, not the OpenAI-compatible one). thinking is never carried
    (prepare_messages strips it before sending)."""
    role: str
    content: str | list[dict]
    tool_calls: NotRequired[list[PreparedOllamaToolCall]]


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


# What a backend's prepare_messages produces and stream_completion/count_tokens accept: the
# OpenAI-wire-ready PreparedLLMMessage (llama.cpp's OpenAI-compatible endpoint) or Ollama's
# own PreparedOllamaMessage (its native /api/chat endpoint) — see each backend's
# prepare_messages in llm/llama_server.py and llm/ollama.py. Never LLMMessage itself: that's
# the internal, pre-wire representation, and by definition isn't "prepared".
PreparedMessages = Sequence[PreparedLLMMessage] | Sequence[PreparedOllamaMessage]
