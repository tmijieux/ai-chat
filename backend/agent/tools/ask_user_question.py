import uuid
from .base import BaseTool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class AskUserQuestionTool(BaseTool):
    name = "ask_user_question"
    description = (
        "Ask the user a clarifying question and wait for their reply. "
        "Use this in Plan mode when you need more information before proposing a plan."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user.",
            },
        },
        "required": ["question"],
    }
    requires_confirmation = False
    measured_delta = 275

    def label(self, args: dict) -> str:
        question = args.get("question", "")
        return f"QUESTION: {question[:80]}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        """Emit an agent_question event and wait for the user's text reply."""
        question = args.get("question", "")
        question_id = str(uuid.uuid4())
        reply = await session.request_user_input(question_id, question)
        return {
            "tool": self.name,
            "status": "success",
            "reply": reply,
        }
