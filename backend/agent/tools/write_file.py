import os
from pathlib import Path
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create a new file or overwrite an existing file. Requires user confirmation. Only use when creating a brand-new file or doing a full rewrite. Prefer edit_file to make targeted edits."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path where the file will be written.",
            },
            "content": {
                "type": "string",
                "description": "The content to write.",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append content instead of overwriting.",
            },
        },
        "required": ["file_path", "content"],
    }
    requires_confirmation = True

    def validate(self, args: dict) -> str:
        path = args.get("file_path", "")
        content = args.get("content", "")
        append = args.get("append", False)
        return f"{'APPEND' if append else 'OVERWRITE'} {path}\n\n{content[:500]}{'...' if len(content) > 500 else ''}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("file_path", "")
        content = args.get("content", "")
        append = args.get("append", False)

        real_path = os.path.realpath(path)
        if not file_in_directory(real_path, working_directory):
            return tool_error(self.name, f"Writing outside workspace is forbidden. Workspace: {working_directory}")

        preview = self.validate(args)
        approved, user_msg = await session.request_confirm(f"write-{path}", self.name, args, preview)
        if not approved:
            return tool_error(self.name, "User aborted the file write", user_message=user_msg)

        try:
            mode = "ab" if append else "wb"
            with open(Path(path), mode=mode) as f:
                f.write(content.encode())
            return {"tool": self.name, "status": "success", "path": path}
        except Exception as e:
            return tool_error(self.name, f"Unexpected error: {e}")
