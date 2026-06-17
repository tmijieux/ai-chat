import aiofiles
from pathlib import Path
from .base import BaseTool, tool_error, tool_rejected
from tool_result_types import WriteFileResult, ToolResult
from agent.file_utils import file_in_directory, resolve_workspace_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create a new file or overwrite an existing file. Requires user confirmation. Only use when creating a brand-new file or doing a full rewrite. Prefer edit_file to make targeted edits. Requires a workspace directory to be configured in conversation settings."
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
    measured_delta = 359

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        path = args.get("file_path", "")
        content = args.get("content", "")
        append = args.get("append", False)
        return f"{'APPEND' if append else 'OVERWRITE'} {path}\n\n{content}"

    def label(self, args: dict) -> str:
        path = args.get("file_path", "")
        append = args.get("append", False)
        return f"{'APPEND' if append else 'WRITE'} {path}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("file_path", "")
        content = args.get("content", "")
        append = args.get("append", False)

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Writing outside workspace is forbidden. Workspace: {working_directory}")

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"write-{path}", self.name, args, preview)
        if not approved:
            return tool_rejected(self.name, reason=user_msg)

        try:
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            async with aiofiles.open(absolute_path, mode=mode, encoding="utf-8") as f:
                await f.write(content)
            return WriteFileResult(
                tool=self.name,
                status="success",
                path=str(absolute_path),
            )
        except Exception as e:
            return tool_error(self.name, f"Unexpected error: {e}")
