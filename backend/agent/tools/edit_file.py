from pathlib import Path
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory, resolve_workspace_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class EditFileTool(BaseTool):
    name = "edit_file"
    description = "Replace a specific string in a file with a new string. Requires user confirmation. Read the file first. Will fail if old_string is not found or not unique — use replace_all for multiple occurrences or widen the context string to make it unique. Requires a workspace directory to be configured in conversation settings."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to replace. Must be unique in the file unless replace_all is true.",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences of old_string (default false).",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }
    requires_confirmation = True
    measured_delta = 413

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        return f"EDIT {path}\n--- OLD ---\n{old_string[:300]}\n--- NEW ---\n{new_string[:300]}"

    def label(self, args: dict) -> str:
        return f"EDIT {args.get('file_path', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = args.get("replace_all", False)

        if not path:
            return tool_error(self.name, "file_path is required")
        if not old_string:
            return tool_error(self.name, "old_string is required")
        if new_string is None:
            return tool_error(self.name, "new_string is required")
        if old_string == new_string:
            return tool_error(self.name, "old_string and new_string must be different")

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Editing outside workspace is forbidden. Workspace: {working_directory}")
        if not absolute_path.is_file():
            return tool_error(self.name, f"File '{path}' does not exist or is not a regular file")

        try:
            current_content = absolute_path.read_text(encoding="utf-8")
        except Exception as e:
            return tool_error(self.name, f"Error reading file: {e}")

        if old_string not in current_content:
            return tool_error(self.name, "old_string not found in file")

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"edit-{path}", self.name, args, preview)
        if not approved:
            return tool_error(self.name, "User aborted the edit", user_message=user_msg)

        try:
            if replace_all:
                new_content = current_content.replace(old_string, new_string)
            else:
                idx = current_content.find(old_string)
                new_content = current_content[:idx] + new_string + current_content[idx + len(old_string):]
            absolute_path.write_text(new_content, encoding="utf-8")
            return {"tool": self.name, "status": "success", "path": str(absolute_path), "message": f"Edition succeed"}
        except Exception as e:
            return tool_error(self.name, f"Unexpected error: {e}")
