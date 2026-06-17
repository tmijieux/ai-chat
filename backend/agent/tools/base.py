from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tool_result_types import ToolResult

if TYPE_CHECKING:
    from agent.agent import AgentSession


def tool_rejected(tool_name: str, reason: str | None = None) -> ToolResult:
    """Return a result indicating the user declined to run this tool."""
    result: ToolResult = {"tool": tool_name, "status": "rejected"}
    if reason is not None:
        result["reason"] = reason
    return result


def tool_error(tool_name: str, error: str, user_message: str | None = None, **extra) -> ToolResult:
    """Return a result indicating the tool failed with an error."""
    result: ToolResult = {
        "tool": tool_name,
        "status": "error",
        "error": {"message": error},
    }
    if user_message is not None:
        result["error"]["user_message"] = user_message
    result.update(extra)
    return result


# Fixed token overhead Ollama adds when any tools are present (measured empirically).
TOOL_FRAMEWORK_OVERHEAD = 223

# Extra tokens each additional tool (2nd, 3rd, ...) adds beyond its schema content.
# Total tool tokens = TOOL_FRAMEWORK_OVERHEAD + sum(t.token_count) + STACKING_OVERHEAD_PER_ADDITIONAL_TOOL * (N - 1)
STACKING_OVERHEAD_PER_ADDITIONAL_TOOL = 25


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict
    requires_confirmation: bool = False
    measured_delta: int  # raw prompt_eval_count delta vs no-tools baseline (1 tool enabled)

    @property
    def token_count(self) -> int:
        return self.measured_delta - TOOL_FRAMEWORK_OVERHEAD

    def to_ollama_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        return str(args)

    def label(self, args: dict) -> str:
        """Short single-line label for log output and UI summaries.
        Non-confirmation tools override this directly.
        Confirmation tools whose confirmation text is already single-line (run_shell, search_web) get
        the right label for free via this delegation. Confirmation tools with multi-line confirmation
        text (write_file, edit_file) override both methods separately."""
        return self.make_validation_text_for_user_confirmation(args)

    @abstractmethod
    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        """Return a ToolResult envelope. Framework serializes to JSON and appends tool_call_id."""
        ...
