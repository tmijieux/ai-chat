import subprocess
from .base import BaseTool, tool_error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List files and directories within a specific path. Use this to understand the project structure, file hierarchy, and permissions."
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

    def validate(self, args: dict) -> str:
        return f"LIST {args.get('path', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("path", "")
        is_recursive = args.get("recursive", False)
        maximum_depth = args.get("maximum_depth", 3 if is_recursive else 1)

        exe = "c:\\Program Files\\Git\\usr\\bin\\find.exe"
        cmd = [exe, path, "-maxdepth", str(maximum_depth)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0:
            return {"tool": self.name, "path": path, "status": "success", "content": proc.stdout.decode()}
        else:
            return tool_error(self.name, proc.stderr.decode())
