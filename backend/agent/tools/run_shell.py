import asyncio
import shutil
import sys
from .base import BaseTool, tool_error, tool_rejected
from tool_result_types import RunShellResult, ToolResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession

_IS_WINDOWS = sys.platform == "win32"


class RunShellTool(BaseTool):
    name = "run_shell"
    description = (
        "Execute a bash command (pipes, &&, etc.). For running npm, git, python scripts, etc. "
        "Requires user confirmation. Requires a workspace directory to be configured in conversation settings (sets the shell CWD)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The full bash command to execute (e.g. 'npm install', 'git log -10').",
            },
        },
        "required": ["command"],
    }
    requires_confirmation = True
    measured_delta = 316

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        return f"SHELL: {args.get('command', '')}"

    def label(self, args: dict) -> str:
        return f"SHELL: {args.get('command', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — shell is disabled.")

        command = args.get("command", "")
        if not command:
            return tool_error(self.name, "command is required")

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"shell-{id(args)}", self.name, args, preview)
        if not approved:
            return tool_rejected(self.name, reason=user_msg)

        try:
            if _IS_WINDOWS:
                bash_exe = shutil.which("bash")
                if bash_exe is None:
                    return tool_error(self.name, "bash not found on PATH.")
                proc = await asyncio.create_subprocess_exec(
                    bash_exe, "-c", command,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    cwd=working_directory,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    cwd=working_directory,
                )

            stdout, stderr = await proc.communicate()
            return RunShellResult(
                tool=self.name,
                status="success",
                command=command,
                exit_code=proc.returncode,
                output=stdout.decode(),
                stderr=stderr.decode(),
            )
        except Exception as e:
            return tool_error(self.name, f"Unexpected error: {e}")
