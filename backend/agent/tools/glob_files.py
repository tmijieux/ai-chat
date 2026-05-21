from pathlib import Path
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory, resolve_workspace_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class GlobFilesTool(BaseTool):
    name = "glob_files"
    description = "Find files matching a glob pattern (e.g. '**/*.ts', 'src/**/*.py'). Use this to locate files by name or extension before reading them. Requires a workspace directory to be configured in conversation settings."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.ts', 'src/**/*.py', 'tests/test_*.py').",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search from (default '.').",
            },
        },
        "required": ["pattern"],
    }
    requires_confirmation = False
    measured_delta = 344

    def validate(self, args: dict) -> str:
        return f"GLOB {args.get('pattern', '')} in {args.get('path', '.')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        pattern = args.get("pattern", "")
        path = args.get("path", ".")

        if not pattern:
            return tool_error(self.name, "pattern is required")

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Searching outside workspace is forbidden. Workspace: {working_directory}")

        try:
            files = [str(p) for p in absolute_path.glob(pattern) if p.is_file()]
            return {"tool": self.name, "status": "success", "files": files, "total": len(files)}
        except Exception as e:
            return tool_error(self.name, f"Error during glob: {e}")
