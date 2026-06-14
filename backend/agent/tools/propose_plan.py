import uuid
from .base import BaseTool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class ProposePlanTool(BaseTool):
    name = "propose_plan"
    description = (
        "Propose a structured plan for the user to review. "
        "The user will choose to accept the plan and continue in Standard, Auto, or YOLO mode. "
        "Only available in Plan mode. Call this once you have enough information to propose a complete plan."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The detailed, step-by-step plan to propose to the user.",
            },
        },
        "required": ["plan"],
    }
    requires_confirmation = False
    measured_delta = 280

    def label(self, args: dict) -> str:
        plan = args.get("plan", "")
        return f"PLAN: {plan[:60]}..." if len(plan) > 60 else f"PLAN: {plan}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        """Emit a plan_proposal event and wait for the user to accept with a chosen mode."""
        plan = args.get("plan", "")
        plan_id = str(uuid.uuid4())
        chosen_mode = await session.request_plan_confirm(plan_id, plan)
        await session.emit({"type": "mode_changed", "mode": chosen_mode})
        return {
            "tool": self.name,
            "status": "accepted",
            "chosen_mode": chosen_mode,
        }
