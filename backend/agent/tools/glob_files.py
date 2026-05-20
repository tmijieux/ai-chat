from pathlib import Path
from .base import BaseTool, tool_error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class GlobFilesTool(BaseTool):
    name = "glob_files"
    description = "Find files matching a glob pattern (e.g. '**/*.ts', 'src/**/*.py'). Use this to locate files by name or extension before reading them."
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

    def validate(self, args: dict) -> str:
        return f"GLOB {args.get('pattern', '')} in {args.get('path', '.')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        pattern = args.get("pattern", "")
        path = args.get("path", working_directory)

        if not pattern:
            return tool_error(self.name, "pattern is required")

        try:
            root = Path(path)
            files = [str(p) for p in root.glob(pattern) if p.is_file()]
            return {"tool": self.name, "status": "success", "files": files, "total": len(files)}
        except Exception as e:
            return tool_error(self.name, f"Error during glob: {e}")
