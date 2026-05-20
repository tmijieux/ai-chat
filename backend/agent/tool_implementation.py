import os
import re
import json
import glob as glob_module
import aiohttp
import subprocess
from typing import Any, TYPE_CHECKING
from pathlib import Path

from .file_utils import file_in_directory

if TYPE_CHECKING:
    from .agent import AgentSession

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"


def _tool_error(call_name: str, call_id: str, error_message: str, user_message: str | None = None, extra_values: dict[str,Any]|None=None) -> str:
    response: dict = {
        "tool": call_name,
        "tool_call_id": call_id,
        "status": "error",
        "error": {"message": error_message},
    }
    if user_message is not None:
        response["error"]["user_message"] = user_message

    if extra_values is not None:
        for k,v in extra_values.items():
            response[k] = v
    return json.dumps(response)


async def _request_confirm(
    session: "AgentSession",
    call_id: str,
    call_name: str,
    arguments: dict,
    preview: str,
) -> tuple[bool, str | None]:
    return await session.request_confirm(call_id, call_name, arguments, preview)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def tool_write_file(call_name: str, call_id: str, arguments: dict[str, Any], session: "AgentSession") -> str:
    path = arguments.get("file_path", "")
    content = arguments.get("content", "")
    append = arguments.get("append", False)
    try:
        path_obj = Path(path)

        current_working_directory = os.path.realpath(os.getcwd())
        file_real_path = os.path.realpath(path)
        if not file_in_directory(file_real_path, current_working_directory):
            return _tool_error(
                call_name, call_id,
                f"Writing outside working directory is forbidden. CWD: {current_working_directory}",
            )

        preview = f"{'APPEND' if append else 'OVERWRITE'} {path}\n\n{content[:500]}{'...' if len(content) > 500 else ''}"
        
        approved, user_msg = await _request_confirm(session, call_id, call_name, arguments, preview)
        if not approved:
            return _tool_error(call_name, call_id, "User aborted the file write", user_message=user_msg)

        mode = "ab" if append else "wb"
        with open(path_obj, mode=mode) as f:
            f.write(content.encode())

        return json.dumps({"tool": call_name, "tool_call_id": call_id, "status": "success"})

    except Exception as e:
        return _tool_error(call_name, call_id, f"Unexpected error: {e}")


def tool_read_file(call_name: str, call_id: str, arguments: dict[str, Any]) -> str:
    path = arguments.get("file_path", "")
    limit = arguments.get("limit", 0)
    try:
        file_content = Path(path).read_text(encoding="utf-8")
        if limit and limit > 0:
            lines = file_content.splitlines()
            file_content = "\n".join(lines[-limit:])
        return json.dumps({
            "tool": call_name,
            "tool_call_id": call_id,
            "status": "success",
            "path": path,
            "content": file_content,
        })
    except Exception as e:
        return _tool_error(call_name, call_id, f"Error reading file: {e}", extra_values={
            "path":path
        })


async def tool_edit_file(call_name: str, call_id: str, arguments: dict[str, Any], session: "AgentSession") -> str:
    """Replace old_string with new_string in a file."""
    path = arguments.get("file_path", "")
    old_string = arguments.get("old_string", "")
    new_string = arguments.get("new_string", "")
    replace_all = arguments.get("replace_all", False)

    if not path:
        return _tool_error(call_name, call_id, "file_path is required")
    if not old_string:
        return _tool_error(call_name, call_id, "old_string is required")
    if new_string is None:
        return _tool_error(call_name, call_id, "new_string is required")
    if old_string == new_string:
        return _tool_error(call_name, call_id, "old_string and new_string must be different")

    try:
        path_obj = Path(path)
        if not path_obj.is_file():
            return _tool_error(call_name, call_id, f"File '{path}' does not exist or is not a regular file")

        current_working_directory = os.path.realpath(os.getcwd())
        file_real_path = os.path.realpath(path)
        if not file_in_directory(file_real_path, current_working_directory):
            return _tool_error(
                call_name, call_id,
                f"Editing outside working directory is forbidden. CWD: {current_working_directory}",
            )

        current_content = path_obj.read_text(encoding="utf-8")
        if old_string not in current_content:
            return _tool_error(call_name, call_id, f"old_string not found in file")

        preview = f"EDIT {path}\n--- OLD ---\n{old_string[:300]}\n--- NEW ---\n{new_string[:300]}"
        approved, user_msg = await _request_confirm(session, call_id, call_name, arguments, preview)
        if not approved:
            return _tool_error(call_name, call_id, "User aborted the edit", user_message=user_msg)

        if replace_all:
            new_content = current_content.replace(old_string, new_string)
        else:
            idx = current_content.find(old_string)
            new_content = current_content[:idx] + new_string + current_content[idx + len(old_string):]

        path_obj.write_text(new_content, encoding="utf-8")
        return json.dumps({"tool": call_name, "tool_call_id": call_id, "status": "success", "message": f"Edited {path}"})

    except Exception as e:
        return _tool_error(call_name, call_id, f"Unexpected error: {e}")


def tool_list_directory(call_name: str, call_id: str, arguments: dict[str, Any]) -> str:
    path = arguments.get("path", "")
    is_recursive = arguments.get("recursive", False)
    maximum_depth = arguments.get("maximum_depth", 3 if is_recursive else 1)

    exe = "c:\\Program Files\\Git\\usr\\bin\\find.exe"
    cmd = [exe, path, "-maxdepth", str(maximum_depth)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0:
        return json.dumps({
            "tool": call_name, "tool_call_id": call_id,
            "path": path, "status": "success",
            "content": proc.stdout.decode(),
        })
    else:
        return _tool_error(call_name, call_id, proc.stderr.decode())


async def tool_run_shell(call_name: str, call_id: str, arguments: dict[str, Any], session: "AgentSession") -> str:
    """Execute a shell command after user confirmation."""
    command = arguments.get("command", "")
    if not command:
        return _tool_error(call_name, call_id, "command is required")

    try:
        preview = f"SHELL: {command}"
        approved, user_msg = await _request_confirm(session, call_id, call_name, arguments, preview)
        if not approved:
            return _tool_error(call_name, call_id, "User aborted the command", user_message=user_msg)

        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            return json.dumps({
                "tool": call_name, "tool_call_id": call_id,
                "status": "success", "output": proc.stdout,
            })
        else:
            return json.dumps({
                "tool": call_name, "tool_call_id": call_id,
                "status": "error", "error": {"message": proc.stderr},
            })
    except Exception as e:
        return _tool_error(call_name, call_id, f"Unexpected error: {e}")


async def tool_search_web(call_name: str, call_id: str, arguments: dict[str, Any]) -> str:
    """Search DuckDuckGo and extract page content with trafilatura."""
    from ddgs import DDGS
    import trafilatura

    query = arguments.get("query", "")
    max_results = 5
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r["href"]
                try:
                    downloaded = trafilatura.fetch_url(url)
                    content = trafilatura.extract(downloaded)
                except Exception:
                    content = None
                results.append({"title": r.get("title"), "url": url, "snippet": r.get("body"), "content": content})

        return json.dumps({
            "tool": call_name, "tool_call_id": call_id, "status": "success",
            "content": json.dumps({"query": query, "results": results, "total_results": len(results)}),
        })
    except Exception as e:
        return _tool_error(call_name, call_id, f"Search error: {e}")


def tool_grep_files(call_name: str, call_id: str, arguments: dict[str, Any]) -> str:
    """Search file contents with a regex pattern using pure Python."""
    pattern = arguments.get("pattern", "")
    path = arguments.get("path", ".")
    glob_pattern = arguments.get("glob", "**/*")
    case_insensitive = arguments.get("case_insensitive", False)
    max_matches = arguments.get("max_matches", 50)

    if not pattern:
        return _tool_error(call_name, call_id, "pattern is required")

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return _tool_error(call_name, call_id, f"Invalid regex: {e}")

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
        return _tool_error(call_name, call_id, f"Error during grep: {e}")

    return json.dumps({
        "tool": call_name, "tool_call_id": call_id, "status": "success",
        "matches": matches, "total": len(matches),
        "truncated": len(matches) >= max_matches,
    })


def tool_glob_files(call_name: str, call_id: str, arguments: dict[str, Any]) -> str:
    """Find files matching a glob pattern using pathlib."""
    pattern = arguments.get("pattern", "")
    path = arguments.get("path", ".")

    if not pattern:
        return _tool_error(call_name, call_id, "pattern is required")

    try:
        root = Path(path)
        files = [str(p) for p in root.glob(pattern) if p.is_file()]
        return json.dumps({
            "tool": call_name, "tool_call_id": call_id, "status": "success",
            "files": files, "total": len(files),
        })
    except Exception as e:
        return _tool_error(call_name, call_id, f"Error during glob: {e}")


async def tool_summarize_subtask(call_name: str, call_id: str, arguments: dict[str, Any]) -> str:
    """Make a one-shot Ollama call to summarize content relevant to a specific task."""
    task = arguments.get("task", "")
    content = arguments.get("content", "")

    if not task or not content:
        return _tool_error(call_name, call_id, "Both 'task' and 'content' are required")

    prompt = f"Task: {task}\n\nContent:\n{content}\n\nProvide a concise summary focused on the task above."
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            ) as response:
                result = await response.json()
                summary = result.get("message", {}).get("content", "")
                return json.dumps({
                    "tool": call_name, "tool_call_id": call_id,
                    "status": "success", "summary": summary,
                })
    except Exception as e:
        return _tool_error(call_name, call_id, f"Summarization error: {e}")


def _not_implemented(call_name: str, call_id: str) -> str:
    return _tool_error(call_name, call_id, f"Tool `{call_name}` is not yet implemented.")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(call_name: str, call_id: str, arguments: dict[str, Any], session: "AgentSession") -> str:
    """Dispatch a tool call to the appropriate implementation."""
    if call_name == "read_file":
        return tool_read_file(call_name, call_id, arguments)
    elif call_name == "list_directory":
        return tool_list_directory(call_name, call_id, arguments)
    elif call_name == "write_file":
        return await tool_write_file(call_name, call_id, arguments, session)
    elif call_name == "edit_file":
        return await tool_edit_file(call_name, call_id, arguments, session)
    elif call_name == "run_shell":
        return await tool_run_shell(call_name, call_id, arguments, session)
    elif call_name == "search_web":
        return await tool_search_web(call_name, call_id, arguments)
    elif call_name == "grep_files":
        return tool_grep_files(call_name, call_id, arguments)
    elif call_name == "glob_files":
        return tool_glob_files(call_name, call_id, arguments)
    elif call_name == "summarize_subtask":
        return await tool_summarize_subtask(call_name, call_id, arguments)
    else:
        return _not_implemented(call_name, call_id)
