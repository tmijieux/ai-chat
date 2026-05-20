import re
from pathlib import Path
from .base import BaseTool, tool_error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class GrepFilesTool(BaseTool):
    name = "grep_files"
    description = "Search file contents with a regex pattern. Returns matching lines with file path and line numbers. Prefer this over reading whole files when looking for a symbol, function, or string."
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
                "description": "Glob filter for which files to search (e.g. '**/*.py', '**/*.ts'). Default '**/*'.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "If true, match case-insensitively (default false).",
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum matches to return (default 50). Increase only if you need more.",
            },
        },
        "required": ["pattern"],
    }
    requires_confirmation = False

    def validate(self, args: dict) -> str:
        return f"GREP '{args.get('pattern', '')}' in {args.get('path', '.')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — file tools are disabled.")

        pattern = args.get("pattern", "")
        path = args.get("path", working_directory)
        glob_pattern = args.get("glob", "**/*")
        case_insensitive = args.get("case_insensitive", False)
        max_matches = args.get("max_matches", 50)

        if not pattern:
            return tool_error(self.name, "pattern is required")

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return tool_error(self.name, f"Invalid regex: {e}")

        matches = []
        root = Path(path)
        try:
            for file_path in root.glob(glob_pattern):
                if not file_path.is_file():
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        matches.append({"file": str(file_path), "line": i, "content": line.strip()})
                        if len(matches) >= max_matches:
                            break
                if len(matches) >= max_matches:
                    break
        except Exception as e:
            return tool_error(self.name, f"Error during grep: {e}")

        return {
            "tool": self.name,
            "status": "success",
            "matches": matches,
            "total": len(matches),
            "truncated": len(matches) >= max_matches,
        }
