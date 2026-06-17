import asyncio
import subprocess
import shutil
import sys
from .base import BaseTool, tool_error, tool_rejected
from tool_result_types import RunShellResult, ToolResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession

_IS_WINDOWS = sys.platform == "win32"

_MEASURED_DELTA_WINDOWS = 427
_MEASURED_DELTA_UNIX = 316


def _build_description() -> str:
    if _IS_WINDOWS:
        return (
            "Execute a shell command on Windows. For running npm, git, python scripts, etc. "
            "Requires user confirmation. Requires a workspace directory to be configured in conversation settings (sets the shell CWD). "
            "Use shell_mode='bash' (default) for Git Bash (POSIX syntax, pipes, &&) "
            "or shell_mode='cmd' for cmd.exe (Windows-native commands)."
        )
    return (
        "Execute a shell command (bash). For running npm, git, python scripts, etc. "
        "Requires user confirmation. Requires a workspace directory to be configured in conversation settings (sets the shell CWD)."
    )


def _build_parameters() -> dict:
    props: dict = {
        "command": {
            "type": "string",
            "description": "The full command string to execute (e.g. 'npm install', 'git log -10').",
        },
    }
    if _IS_WINDOWS:
        props["shell_mode"] = {
            "type": "string",
            "enum": ["bash", "cmd"],
            "description": (
                "'bash' (default): Git Bash — POSIX syntax, pipes, &&, etc. "
                "'cmd': cmd.exe — for Windows-native commands that require cmd syntax."
            ),
        }
    return {"type": "object", "properties": props, "required": ["command"]}


class RunShellTool(BaseTool):
    name = "run_shell"
    description = _build_description()
    parameters = _build_parameters()
    requires_confirmation = True
    measured_delta = _MEASURED_DELTA_WINDOWS if _IS_WINDOWS else _MEASURED_DELTA_UNIX

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        mode = args.get("shell_mode", "bash")
        label = f"[{mode}]" if _IS_WINDOWS else ""
        return f"SHELL{label}: {args.get('command', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — shell is disabled.")

        command = args.get("command", "")
        if not command:
            return tool_error(self.name, "command is required")

        shell_mode = args.get("shell_mode", "bash")

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"shell-{id(args)}", self.name, args, preview)
        if not approved:
            return tool_rejected(self.name, reason=user_msg)

        try:
            if _IS_WINDOWS and shell_mode == "bash":
                bash_exe = shutil.which("bash")
                if bash_exe is None:
                    return tool_error(self.name, "Git Bash not found on PATH. Install Git for Windows or use shell_mode='cmd'.")
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
            if proc.returncode == 0:
                return RunShellResult(
                    tool=self.name,
                    status="success",
                    command=command,
                    output=stdout.decode(),
                    stderr=stderr.decode(),
                )
            else:
                return RunShellResult(
                    tool=self.name,
                    status="error",
                    command=command,
                    exit_code=proc.returncode,
                    output=stdout.decode(),
                    stderr=stderr.decode(),
                    error={"message": f"exit code {proc.returncode}"},
                )
        except Exception as e:
            return tool_error(self.name, f"Unexpected error: {e}")
