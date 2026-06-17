import aiofiles
from pathlib import Path
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory, resolve_workspace_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession

FILE_MAX_BYTES = 40_000  # ~10k tokens — refuse full reads above this


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read the full content of a file. "
        "Only use this as a last resort after multiple grep_files attempts failed to return useful results. "
        "Refuses files larger than 40 000 bytes — use grep_files with -A/-B to extract relevant sections, "
        "then read_file_range for specific line ranges. "
        "Use the limit parameter to tail a log file (reads the last N lines regardless of size). "
        "Requires a workspace directory to be configured in conversation settings."
    )
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
    measured_delta = 399

    def label(self, args: dict) -> str:
        return f"READ {args.get('file_path', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        path = args.get("file_path", "")
        limit = args.get("limit", 0)

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Reading outside workspace is forbidden. Workspace: {working_directory}", path=path)

        posix_key = absolute_path.relative_to(Path(working_directory)).as_posix()
        grep_count = session._grepped_files.get(posix_key, 0)
        if grep_count < 2:
            return tool_error(
                self.name,
                f"Cannot read file before grepping it at least twice. Wildcard patterns \".*\"  are not allowed! Search for what you need."
                f"grep_files has matched '{posix_key}' {grep_count} time(s) so far. "
                "Use grep_files with -B/-A to extract the relevant section. "
                "Only call read_file if multiple grep_files attempts on this file did not yield useful results.",
                path=path,
            )

        try:
            file_size = absolute_path.stat().st_size
        except Exception as e:
            return tool_error(self.name, f"Error reading file: {e}", path=path)

        # Refuse full reads of large files — the agent must narrow down first.
        # limit (tail) is exempt: reading the last N lines of a large log is always fine.
        if file_size > FILE_MAX_BYTES and not (limit and limit > 0):
            approx_lines = file_size // 80
            return tool_error(
                self.name,
                f"File too large to read in full: {file_size:,} bytes (~{approx_lines:,} lines). "
                "If you don't know the file's content yet, use grep_files with -A/-B flags to locate "
                "the relevant section, then read_file_range (requires the grep result_id) to read "
                "specific line ranges. If you need the last N lines of a log, use the limit parameter.",
                path=path,
            )

        try:
            async with aiofiles.open(absolute_path, encoding="utf-8") as f:
                file_content = await f.read()
            if limit and limit > 0:
                lines = file_content.splitlines()
                file_content = "\n".join(lines[-limit:])
            return {"tool": self.name, "status": "success", "path": path, "file_content": file_content}
        except Exception as e:
            return tool_error(self.name, f"Error reading file: {e}", path=path)
