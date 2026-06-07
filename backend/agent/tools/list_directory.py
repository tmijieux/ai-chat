import asyncio
from pathlib import Path
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory, resolve_workspace_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List files and directories within a specific path. Use this to understand the project structure, file hierarchy, and permissions. Requires a workspace directory to be configured in conversation settings. This tool works like the unix 'find' command"
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

    def label(self, args: dict) -> str:
        return f"DIRECTORY {args.get('path', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("path", ".")
        is_recursive = args.get("recursive", False)
        maximum_depth = args.get("maximum_depth", 3 if is_recursive else 1)

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Listing outside workspace is forbidden. Workspace: {working_directory}")

        rel_root = absolute_path.relative_to(working_directory)

        def _walk(base: Path, max_depth: int) -> str:
            lines = [str(rel_root)]
            def _recurse(p: Path, depth: int) -> None:
                if depth > max_depth:
                    return
                try:
                    entries = sorted(p.iterdir())
                except PermissionError:
                    return
                for entry in entries:
                    lines.append(str(entry.relative_to(Path(working_directory))))
                    if entry.is_dir() and depth < max_depth:
                        _recurse(entry, depth + 1)
            _recurse(base, 1)
            return "\n".join(lines)

        try:
            content = await asyncio.to_thread(_walk, absolute_path, maximum_depth)
            return {"tool": self.name, "path": str(rel_root), "status": "success", "content": content}
        except Exception as e:
            return tool_error(self.name, str(e))
