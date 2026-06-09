from agent.tools.base import BaseTool, tool_error
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
        "Call this once you have explored enough files to clarify the user's request. "
        "The snippets field is REQUIRED — list every file location you found so the system can read the actual code."
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
            "snippets": {
                "type": "array",
                "description": "File locations needed to plan and implement the requested changes. One entry per relevant code block. Exclude exploration dead-ends.",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path to the file."},
                        "start_line": {"type": "integer", "description": "First line of the block (1-indexed)."},
                        "end_line": {"type": "integer", "description": "Last line of the block (1-indexed)."},
                    },
                    "required": ["file_path", "start_line", "end_line"],
                },
            },
        },
        "required": ["augmented_prompt", "snippets"],
    }


class FinishCritique(BaseFinishTool):
    name = "finish_critique"
    description = (
        "Output critique of the augmented prompt and the final corrected prompt to use for planning. "
        "The snippets field is REQUIRED — return only the snippet coordinates actually needed to plan and implement the changes (prune irrelevant ones, keep all relevant ones unchanged)."
    )
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
            "snippets": {
                "type": "array",
                "description": "Snippet coordinates needed to plan and implement the changes. Prune any that are not relevant. Keep file_path, start_line, end_line unchanged from the input.",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["file_path", "start_line", "end_line"],
                },
            },
        },
        "required": ["final_prompt", "issues", "snippets"],
    }


MIN_PLAN_TASKS = 5


class FinishPlan(BaseFinishTool):
    name = "finish_plan"
    description = "Output the complete ordered list of atomic tasks to execute. Minimum 5 tasks required — the call will be rejected if fewer are submitted."
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
                "description": f"Ordered list of atomic tasks. Minimum {MIN_PLAN_TASKS} tasks required.",
            },
            "compile_command": {
                "type": "string",
                "description": "Shell command to compile/type-check the project after all tasks complete (e.g. 'cd chat-client && npx tsc --noEmit', 'python -m py_compile src/main.py'). Omit if no compilation step applies.",
            },
        },
        "required": ["tasks"],
    }

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        """Reject if too few tasks or if any description contains multiple actions."""
        tasks = args.get("tasks") or []
        if len(tasks) < MIN_PLAN_TASKS:
            return tool_error(
                self.name,
                f"Too few tasks: {len(tasks)} submitted, minimum is {MIN_PLAN_TASKS}. Split further — each file change must be its own task.",
            )
        multi_action_markers = [" and ", " also ", " then ", " but also "]
        offenders = [
            t.get("id", "?")
            for t in tasks
            if any(marker in (t.get("description") or "").lower() for marker in multi_action_markers)
        ]
        if offenders:
            return tool_error(
                self.name,
                f"Tasks {', '.join(offenders)} describe multiple actions (contains 'and'/'also'/'then'). Split each into one action per task.",
            )
        session.finish_result = args
        return {"tool": self.name, "status": "success"}


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


class FinishExplore(BaseFinishTool):
    name = "finish_explore"
    description = "Output the file locations found during exploration. The system will read the actual code — do not copy code content yourself."
    parameters = {
        "type": "object",
        "properties": {
            "snippets": {
                "type": "array",
                "description": "Locations of relevant code blocks found via grep/glob. One entry per block.",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path to the file."},
                        "start_line": {"type": "integer", "description": "First line of the block (1-indexed)."},
                        "end_line": {"type": "integer", "description": "Last line of the block (1-indexed)."},
                    },
                    "required": ["file_path", "start_line", "end_line"],
                },
            },
            "summary": {
                "type": "string",
                "description": "Brief description of what was found and why it is relevant.",
            },
        },
        "required": ["snippets", "summary"],
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
