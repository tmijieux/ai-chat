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


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict
    requires_confirmation: bool = False

    def to_ollama_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def validate(self, args: dict) -> str:
        return str(args)

    @abstractmethod
    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        """Return a plain dict (Tool Result Envelope). Framework serializes to JSON."""
        ...
