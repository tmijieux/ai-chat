import asyncio
import re
import uuid
from pathlib import Path
from .base import BaseTool, tool_error
from agent.file_utils import file_in_directory, resolve_workspace_path, load_ignore_spec, is_path_ignored
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from agent.agent import AgentSession

logger = logging.getLogger(__name__)

class GrepFilesTool(BaseTool):
    name = "grep_files"
    description = "Search file contents with a regex pattern. Returns matching lines with file path and line numbers. Use -B/-A to read the surrounding lines of a match — this is the primary way to read file content: grep for a known symbol/class/function and use -B/-A to extract the full block. Requires a workspace directory to be configured in conversation settings."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for (e.g. 'def my_func', 'import.*react', 'TODO').",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search from (default '.').",
            },
            "glob": {
                "type": "string",
                "description": "Glob filter for which files to search (e.g. '**/*.py', '**/*.ts'). Never use the global wildcard '**/*' because it is too long.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "If true, match case-insensitively (default false).",
            },
            "-A": {
                "type": "integer",
                "description": "Number of lines to include after each match. Default 0.",
            },
            "-B": {
                "type": "integer",
                "description": "Number of lines to include before each match. Default 0.",
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum matches to return (default 50). Increase only if you need more.",
            },
            "include_ignored": {
                "type": "boolean",
                "description": "If true, include files normally excluded by .gitignore and common build/dependency directories (venv, node_modules, .git, __pycache__, dist, build). Default false.",
            },
        },
        "required": ["pattern", "glob"],
    }
    requires_confirmation = False
    measured_delta = 574

    def label(self, args: dict) -> str:
        return f"GREP '{args.get('pattern', '')}' in {args.get('path', '.')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        glob_pattern = args.get("glob", None)
        if glob_pattern is None:
            return tool_error(self.name, "the 'glob' argument is missing")
        case_insensitive = args.get("case_insensitive", False)
        lines_after = int(args.get("-A", 0))
        lines_before = int(args.get("-B", 0))
        max_matches = args.get("max_matches", 50)
        include_ignored = args.get("include_ignored", False)

        if not pattern:
            return tool_error(self.name, "pattern is required")

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return tool_error(self.name, f"Invalid regex: {e}")

        absolute_path = resolve_workspace_path(path, working_directory)
        if not file_in_directory(str(absolute_path), working_directory):
            return tool_error(self.name, f"Searching outside workspace is forbidden. Workspace: {working_directory}")

        spec = None if include_ignored else load_ignore_spec(working_directory)

        def _do_grep() -> tuple[list, int]:
            matches = []
            total_match_count = 0
            for file_path in absolute_path.glob(glob_pattern):
                if not file_path.is_file():
                    continue
                if not include_ignored and is_path_ignored(file_path, working_directory, spec):
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="strict")
                except Exception:
                    logger.warning("skipping unreadable file: %s", file_path)
                    continue
                all_lines = content.splitlines()
                rel_path = file_path.relative_to(working_directory)
                match_indices: set[int] = set()
                show_indices: set[int] = set()
                for i, line in enumerate(all_lines):
                    if regex.search(line):
                        match_indices.add(i)
                        for j in range(max(0, i - lines_before), min(len(all_lines), i + lines_after + 1)):
                            show_indices.add(j)
                        if total_match_count + len(match_indices) >= max_matches:
                            break

                for idx in sorted(show_indices):
                    entry: dict = {
                        "file": str(rel_path),
                        "line": idx + 1,
                        "content": all_lines[idx].rstrip(),
                    }
                    if idx in match_indices:
                        entry["match"] = True
                    matches.append(entry)

                total_match_count += len(match_indices)
                if total_match_count >= max_matches:
                    break
            return matches, total_match_count

        try:
            matches, total_match_count = await asyncio.to_thread(_do_grep)
        except Exception as e:
            return tool_error(self.name, f"Error during grep: {e}")

        result_id = str(uuid.uuid4())[:8]
        session._search_result_ids.add(result_id)
        return {
            "tool": self.name,
            "pattern": pattern,
            "path": path,
            "glob_pattern": glob_pattern,
            "status": "success",
            "result_id": result_id,
            "matches": matches,
            "total": total_match_count,
            "truncated": total_match_count >= max_matches,
        }
