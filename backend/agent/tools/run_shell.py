import subprocess
from .base import BaseTool, tool_error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class RunShellTool(BaseTool):
    name = "run_shell"
    description = "Execute a shell command (bash). For running npm, git, python scripts, etc. Requires user confirmation. Requires a workspace directory to be configured in conversation settings (sets the shell CWD)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The full command string to execute (e.g. 'npm install', 'git log -10').",
            },
        },
        "required": ["command"],
    }
    requires_confirmation = True
    measured_delta = 316

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        return f"SHELL: {args.get('command', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — shell is disabled.")

        command = args.get("command", "")
        if not command:
            return tool_error(self.name, "command is required")

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"shell-{id(args)}", self.name, args, preview)
        if not approved:
            return tool_error(self.name, "User aborted the command", user_message=user_msg)

        try:
            proc = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=working_directory,
            )
            if proc.returncode == 0:
                return {"tool": self.name, "status": "success", "command": command, "output": proc.stdout}
            else:
                return {"tool": self.name, "status": "error", "command": command, "error": {"message": proc.stderr}}
        except Exception as e:
            return tool_error(self.name, f"Unexpected error: {e}")
