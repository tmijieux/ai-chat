from agent.tools.base import BaseTool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class BaseFinishTool(BaseTool):
    """Signals stage completion by storing its args in session.finish_result."""

    requires_confirmation = False
    measured_delta = 223  # baseline framework overhead, no content

    def label(self, args: dict) -> str:
        """Return a short label for logging."""
        return f"{self.name}()"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        """Store args on the session so run_stage() can detect stage completion."""
        session.finish_result = args
        return {"tool": self.name, "status": "success"}


class FinishClassify(BaseFinishTool):
    name = "finish_classify"
    description = (
        "Signal whether the user request is simple or complex. "
        "'simple': greetings, factual questions, single-turn explanations — no file access needed. "
        "'complex': requires reading/writing files, code changes, or multi-step work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "complexity": {
                "type": "string",
                "enum": ["simple", "complex"],
            },
        },
        "required": ["complexity"],
    }


class FinishAugmentation(BaseFinishTool):
    name = "finish_augmentation"
    description = (
        "Output the augmented prompt after gathering context from the codebase. "
        "Call this once you have explored enough files to clarify the user's request."
    )
    parameters = {
        "type": "object",
        "properties": {
            "augmented_prompt": {
                "type": "string",
                "description": "Expanded and clarified request with relevant file paths, function names, and context.",
            },
            "context_notes": {
                "type": "string",
                "description": "Key facts discovered: relevant files, patterns, existing code structure.",
            },
        },
        "required": ["augmented_prompt"],
    }


class FinishCritique(BaseFinishTool):
    name = "finish_critique"
    description = "Output critique of the augmented prompt and the final corrected prompt to use for planning."
    parameters = {
        "type": "object",
        "properties": {
            "final_prompt": {
                "type": "string",
                "description": "Final prompt for planning — the augmented prompt unchanged if correct, or a corrected version.",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Issues found in the augmented prompt. Empty list if none.",
            },
        },
        "required": ["final_prompt", "issues"],
    }


class FinishPlan(BaseFinishTool):
    name = "finish_plan"
    description = "Output the complete ordered list of atomic tasks to execute."
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Short unique id like 'task-1'."},
                        "description": {"type": "string", "description": "Concrete task description."},
                        "verification_method": {
                            "type": "string",
                            "description": "Concrete way to verify this task succeeded.",
                        },
                    },
                    "required": ["id", "description", "verification_method"],
                },
                "description": "Ordered list of atomic tasks. Prefer fewer tasks over more.",
            },
        },
        "required": ["tasks"],
    }


class FinishTask(BaseFinishTool):
    name = "finish_task"
    description = "Signal that the current task is complete. Call this when done working on the assigned task."
    parameters = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
            },
            "result": {
                "type": "string",
                "description": "Summary of what was done, or reason for failure.",
            },
        },
        "required": ["status", "result"],
    }


class FinishVerify(BaseFinishTool):
    name = "finish_verify"
    description = "Output the verification result after checking whether the task was completed correctly."
    parameters = {
        "type": "object",
        "properties": {
            "passed": {
                "type": "boolean",
                "description": "True if the task was verified successfully.",
            },
            "issues": {
                "type": "string",
                "description": "Description of issues found. Empty string if passed.",
            },
        },
        "required": ["passed", "issues"],
    }
