import aiofiles
import difflib
import json
import re as _re
from pathlib import Path
from .base import BaseTool, tool_error, tool_rejected
from tool_result_types import EditFileResult
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

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> EditFileResult:
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
            async with aiofiles.open(absolute_path, encoding="utf-8") as f:
                current_content = await f.read()
        except Exception as e:
            return tool_error(self.name, f"Error reading file: {e}")

        if old_string not in current_content:
            # The model sometimes copies strings verbatim from read_file tool results, where
            # content is JSON-encoded inside the outer JSON response. This causes one extra
            # level of escaping (e.g. actual " becomes \" in old_string). Try decoding once.
            try:
                decoded_old = json.loads('"' + old_string + '"')
            except (json.JSONDecodeError, ValueError):
                return tool_error(self.name, "old_string not found in file")
            if decoded_old == old_string or decoded_old not in current_content:
                return tool_error(self.name, "old_string not found in file")
            old_string = decoded_old
            args = {**args, "old_string": old_string}
            try:
                new_string = json.loads('"' + new_string + '"')
                args = {**args, "new_string": new_string}
            except (json.JSONDecodeError, ValueError):
                pass

        idx = current_content.find(old_string)

        diff_lines = None
        start_line = current_content[:idx].count('\n') + 1
        old_sl = old_string.splitlines(True)
        new_sl = new_string.splitlines(True)
        raw_diff = list(difflib.unified_diff(old_sl, new_sl, n=3, lineterm=''))
        if raw_diff:
            diff_lines = []
            cur_old = start_line
            for raw in raw_diff:
                if raw.startswith('---') or raw.startswith('+++'):
                    continue
                if raw.startswith('@@'):
                    m = _re.match(r'@@ -(\d+)', raw)
                    if m:
                        cur_old = start_line + int(m.group(1)) - 1
                    diff_lines.append({'type': 'header', 'text': raw.rstrip('\n')})
                elif raw.startswith('-'):
                    diff_lines.append({'type': 'removed', 'line': cur_old, 'text': raw[1:].rstrip('\n')})
                    cur_old += 1
                elif raw.startswith('+'):
                    diff_lines.append({'type': 'added', 'line': None, 'text': raw[1:].rstrip('\n')})
                else:
                    diff_lines.append({'type': 'context', 'line': cur_old, 'text': raw[1:].rstrip('\n')})
                    cur_old += 1

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"edit-{path}", self.name, args, preview, diff_lines=diff_lines)
        if not approved:
            return tool_rejected(self.name, reason=user_msg)

        try:
            if replace_all:
                new_content = current_content.replace(old_string, new_string)
            else:
                new_content = current_content[:idx] + new_string + current_content[idx + len(old_string):]
            async with aiofiles.open(absolute_path, "w", encoding="utf-8") as f:
                await f.write(new_content)
            return EditFileResult(
                tool=self.name,
                status="success",
                path=str(absolute_path),
            )
        except Exception as e:
            return tool_error(self.name, f"Unexpected error: {e}")
