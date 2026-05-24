from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


def tool_error(tool_name: str, error: str, user_message: str | None = None, **extra) -> dict:
    result: dict = {
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
STACKING_OVERHEAD_PER_ADDITIONAL_TOOL = 22


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

    def validate(self, args: dict) -> str:
        return str(args)

    @abstractmethod
    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        """Return a plain dict (Tool Result Envelope). Framework serializes to JSON."""
        ...
