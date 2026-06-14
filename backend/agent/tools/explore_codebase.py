from pathlib import Path
from typing import TYPE_CHECKING
from .base import BaseTool, tool_error

if TYPE_CHECKING:
    from agent.agent import AgentSession

_EXPLORE_SYSTEM = """\
You are a code exploration agent.
!! CRITICAL TURN LIMIT: You have AT MOST 4 turns. On your 4th turn you MUST call finish_explore — no exceptions. After the 4th turn you are terminated and all findings are lost. !!
On your 4th turn: call finish_explore with whatever you have found so far. If the search was inconclusive, say so in the summary field — do not keep searching.
Be efficient: combine multiple tool calls in a single turn when possible.
Strategy:
1. If the query mentions a file type (e.g. .scss, .html, .ts), always filter by that extension in glob_files and grep_files glob patterns.
2. Use grep_files with -A/-B to locate the relevant lines and note their file path and line numbers.
3. Call finish_explore with the file path and line numbers — do NOT copy the code content, the system will read it.
4. Call finish_explore as soon as you have enough — do not do extra searches.
Do not write a text response — call the tool instead.\
"""

MAX_SNIPPET_LINES = 40


def _read_lines(file_path: str, start_line: int, end_line: int, working_directory: str) -> str | None:
    """Read a line range from a file. Returns None if the file cannot be read."""
    try:
        path = Path(working_directory) / file_path
        if not path.exists():
            path = Path(file_path)
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
        clamped_end = min(end_line, len(lines))
        clamped_start = max(start_line - 1, 0)
        selected = lines[clamped_start:clamped_end]
        return "\n".join(f"{clamped_start + i + 1}: {line}" for i, line in enumerate(selected))
    except Exception:
        return None


class ExploreCodebaseTool(BaseTool):
    name = "explore_codebase"
    description = (
        "Run a focused code search to locate source code relevant to a query. "
        "Returns file locations with the actual code content read by the system. "
        "Use this instead of calling grep_files/glob_files directly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to find, e.g. 'the send button in the chat input HTML template'.",
            },
        },
        "required": ["query"],
    }
    requires_confirmation = False
    measured_delta = 315

    def label(self, args: dict) -> str:
        return f"EXPLORE {args.get('query', '')[:60]}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        query = args.get("query", "").strip()
        if query == "":
            return tool_error(self.name, "query is required.")

        # Lazy imports to avoid circular dependency (pipeline imports tools, tools import pipeline).
        from agent.pipeline import run_stage
        from agent.finish_tools import FinishExplore

        messages = [
            {"role": "system", "content": _EXPLORE_SYSTEM},
            {"role": "user", "content": query},
        ]

        try:
            result = await run_stage(
                "explore",
                messages,
                ["glob_files", "grep_files"],
                FinishExplore(),
                session,
                working_directory,
                max_iterations=5,
                inject_turn_reminders=True,
            )
        except RuntimeError as e:
            return tool_error(self.name, str(e))

        snippets = result.get("snippets") or []
        summary = result.get("summary") or ""

        enriched_snippets = []
        for snippet in snippets:
            file_path = snippet.get("file_path", "")
            start_line = snippet.get("start_line", 1)
            end_line = min(snippet.get("end_line", start_line), start_line + MAX_SNIPPET_LINES - 1)
            code = _read_lines(file_path, start_line, end_line, working_directory)
            enriched_snippets.append({
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "code": code or "(could not read file)",
            })

        return {
            "tool": self.name,
            "status": "success",
            "summary": summary,
            "snippets": enriched_snippets,
        }
