from pathlib import Path
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory, resolve_workspace_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the full content of a file. For large files, use the limit parameter to read only the last N lines. Requires a workspace directory to be configured in conversation settings."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of lines to read from the END of the file (useful for logs). 0 or omit to read the entire file.",
            },
        },
        "required": ["file_path"],
    }
    requires_confirmation = False
    measured_delta = 343

    def validate(self, args: dict) -> str:
        return f"READ {args.get('file_path', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("file_path", "")
        limit = args.get("limit", 0)

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Reading outside workspace is forbidden. Workspace: {working_directory}", path=path)

        try:
            file_content = absolute_path.read_text(encoding="utf-8")
            if limit and limit > 0:
                lines = file_content.splitlines()
                file_content = "\n".join(lines[-limit:])
            return {"tool": self.name, "status": "success", "path": path, "file_content": file_content}
        except Exception as e:
            return tool_error(self.name, f"Error reading file: {e}", path=path)
