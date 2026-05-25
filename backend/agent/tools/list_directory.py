import subprocess
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory, resolve_workspace_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List files and directories within a specific path. Use this to understand the project structure, file hierarchy, and permissions. Requires a workspace directory to be configured in conversation settings."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The directory path to list (e.g. '.', '/src', 'project_name').",
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, lists files in subdirectories. Keep false when just looking at a root folder to save tokens.",
            },
            "maximum_depth": {
                "type": "integer",
                "description": "Maximum depth for recursive listing (default 3). Keep low to prevent context explosion.",
            },
        },
        "required": ["path"],
    }
    requires_confirmation = False
    measured_delta = 375

    def validate(self, args: dict) -> str:
        return f"LIST {args.get('path', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("path", ".")
        is_recursive = args.get("recursive", False)
        maximum_depth = args.get("maximum_depth", 3 if is_recursive else 1)

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Listing outside workspace is forbidden. Workspace: {working_directory}")

        path = absolute_path.relative_to(working_directory)

        exe = "c:\\Program Files\\Git\\usr\\bin\\find.exe"
        cmd = [exe, str(path), "-maxdepth", str(maximum_depth)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0:
            return {"tool": self.name, "path": str(absolute_path), "status": "success", "content": proc.stdout.decode()}
        else:
            return tool_error(self.name, proc.stderr.decode())
