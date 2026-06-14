import uuid
from .base import BaseTool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class AskUserQuestionTool(BaseTool):
    name = "ask_user_question"
    description = (
        "Ask the user a clarifying question and wait for their reply. "
        "Optionally provide predefined choices — an 'Other' option is always added automatically. "
        "ALWAYS use this tool instead of asking questions in plain text — "
        "it renders an interactive card in the UI and suspends execution until the user replies."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional predefined choices for the user to pick from.",
            },
        },
        "required": ["question"],
    }
    requires_confirmation = False
    measured_delta = 275

    def label(self, args: dict) -> str:
        return f"QUESTION: {args.get('question', '')[:80]}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        """Emit an agent_question event and wait for the user's reply."""
        question = args.get("question", "")
        options: list[str] | None = args.get("options")
        question_id = str(uuid.uuid4())
        reply = await session.request_user_input(question_id, question, options=options)
        return {
            "tool": self.name,
            "status": "success",
            "reply": reply,
        }
