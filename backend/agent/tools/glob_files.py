import asyncio
import uuid
from pathlib import Path
from .base import BaseTool, tool_error
from tool_result_types import GlobFilesResult, ToolResult
from agent.file_utils import file_in_directory, resolve_workspace_path, load_ignore_spec, is_path_ignored
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
            "include_ignored": {
                "type": "boolean",
                "description": "If true, include files normally excluded by .gitignore and common build/dependency directories (venv, node_modules, .git, __pycache__, dist, build). Default false.",
            },
        },
        "required": ["pattern"],
    }
    requires_confirmation = False
    measured_delta = 397

    def label(self, args: dict) -> str:
        return f"GLOB {args.get('pattern', '')} in {args.get('path', '.')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        include_ignored = args.get("include_ignored", False)

        if not pattern:
            return tool_error(self.name, "pattern is required")

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Searching outside workspace is forbidden. Workspace: {working_directory}")

        try:
            spec = None if include_ignored else load_ignore_spec(working_directory)
            files = await asyncio.to_thread(
                lambda: [
                    str(p) for p in absolute_path.glob(pattern)
                    if p.is_file() and (include_ignored or not is_path_ignored(p, working_directory, spec))
                ]
            )
            result_id = str(uuid.uuid4())[:8]
            session._search_result_ids.add(result_id)
            return GlobFilesResult(
                tool=self.name,
                status="success",
                pattern=pattern,
                path=path,
                result_id=result_id,
                files=files,
                file_count=len(files),
            )
        except Exception as e:
            return tool_error(self.name, f"Error during glob: {e}")
