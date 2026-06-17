from pathlib import Path
from typing import TYPE_CHECKING
from .base import BaseTool, tool_error
from tool_result_types import ReadFileRangeResult
from agent.file_utils import file_in_directory, resolve_workspace_path

if TYPE_CHECKING:
    from agent.agent import AgentSession

MAX_LINES = 40


class ReadFileRangeTool(BaseTool):
    name = "read_file_range"
    description = (
        "Read a specific line range from a file. "
        "You MUST call glob_files or grep_files first and pass its result_id here — "
        "this enforces that you have located the file before reading. "
        f"Maximum {MAX_LINES} lines per call. Read tight ranges (10-20 lines) — do not read the maximum unless truly needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed, inclusive).",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (1-indexed, inclusive).",
            },
            "search_result_id": {
                "type": "string",
                "description": "The result_id returned by a previous grep_files call. Required.",
            },
        },
        "required": ["file_path", "start_line", "end_line", "search_result_id"],
    }
    requires_confirmation = False
    measured_delta = 426

    def label(self, args: dict) -> str:
        return f"READ {args.get('file_path', '')} lines {args.get('start_line')}–{args.get('end_line')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ReadFileRangeResult:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        search_result_id = args.get("search_result_id", "")
        if search_result_id not in session._search_result_ids:
            return tool_error(
                self.name,
                "Invalid or missing search_result_id. Call glob_files or grep_files first and pass its result_id.",
            )

        file_path = args.get("file_path", "")
        start_line = int(args.get("start_line"))
        end_line = int(args.get("end_line"))

        if not file_path or start_line is None or end_line is None:
            return tool_error(self.name, "file_path, start_line, and end_line are required.")

        if start_line < 1 or end_line < start_line:
            return tool_error(self.name, "start_line must be >= 1 and end_line must be >= start_line.")

        if (end_line - start_line + 1) > MAX_LINES:
            return tool_error(self.name, f"Range too large: max {MAX_LINES} lines, requested {end_line - start_line + 1}.")

        absolute_path = resolve_workspace_path(file_path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, "Reading outside workspace is forbidden.")

        try:
            lines = Path(absolute_path).read_text(encoding="utf-8", errors="strict").splitlines()
        except FileNotFoundError:
            return tool_error(self.name, f"File not found: {file_path}")
        except Exception as e:
            return tool_error(self.name, f"Could not read file: {e}")

        slice_start = start_line - 1
        slice_end = min(end_line, len(lines))
        selected = lines[slice_start:slice_end]
        numbered = "\n".join(f"{slice_start + i + 1}: {line}" for i, line in enumerate(selected))

        return ReadFileRangeResult(
            tool=self.name,
            status="success",
            file_path=file_path,
            start_line=start_line,
            end_line=slice_end,
            content=numbered,
        )
