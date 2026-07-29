from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.file_utils import file_in_directory, resolve_workspace_path

if TYPE_CHECKING:
    from agent.agent import AgentSession

logger = logging.getLogger(__name__)

CHUNK_FILE_DEFAULT_SIZE = 10
CHUNK_JSON_ENTRIES_DEFAULT_SIZE = 10


async def run_coordinator_action(
    action: str,
    inputs: dict[str, Any],
    session: "AgentSession",
    working_directory: str | None,
    workflow_directory: Path | None = None,
) -> Any:
    """Dispatch a named coordinator action. Returns the action's output value."""
    if action == "read_code_context":
        return await _read_code_context(inputs, working_directory)
    if action == "run_compile":
        return await _run_compile(inputs, working_directory, session)
    if action == "chunk_file":
        return await _chunk_file(inputs, working_directory)
    if action == "append_text":
        return await _append_text(inputs, working_directory)
    if action == "run_script":
        return await _run_script(inputs, working_directory, workflow_directory)
    if action == "chunk_json_entries":
        return await _chunk_json_entries(inputs, working_directory)
    if action == "append_json_entries":
        return await _append_json_entries(inputs, working_directory)
    raise ValueError(f"Unknown coordinator action: {action!r}")


async def _read_code_context(inputs: dict[str, Any], working_directory: str | None) -> str:
    """Read file lines for a list of snippet coordinates and return formatted code blocks."""
    from agent.pipeline import _build_code_context
    snippets = inputs.get("snippets") or []
    if not snippets or working_directory is None:
        return ""
    return _build_code_context(snippets, working_directory)


async def _run_compile(
    inputs: dict[str, Any],
    working_directory: str | None,
    session: "AgentSession",
) -> dict[str, Any]:
    """Run a compile/type-check command with user confirmation. Returns {success, output}."""
    command: str | None = inputs.get("command")
    if command is None or command == "" or working_directory is None:
        return {"success": True, "output": ""}

    approved, _ = await session.request_confirm(
        tool_id=f"compile-{id(command)}",
        tool_name="compile_check",
        arguments={"command": command},
        preview=f"$ {command}",
    )
    if not approved:
        return {"success": True, "output": "Compile check skipped by user."}

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=working_directory,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode("utf-8", errors="replace").strip()
    logger.info("[coordinator:run_compile] exit=%d output_len=%d", proc.returncode, len(output))
    return {"success": proc.returncode == 0, "output": output}


async def _chunk_file(inputs: dict[str, Any], working_directory: str | None) -> list[dict]:
    """Split a file into fixed-size line chunks. Returns [{index, start_line, end_line, text}], deterministically — no model involved."""
    path = inputs.get("path")
    if path is None or working_directory is None:
        return []

    absolute_path = resolve_workspace_path(path, working_directory)
    if not file_in_directory(str(absolute_path), working_directory):
        raise ValueError(f"Reading outside workspace is forbidden: {path}")

    lines = Path(absolute_path).read_text(encoding="utf-8").splitlines()
    chunks = []
    for start in range(0, len(lines), CHUNK_FILE_DEFAULT_SIZE):
        chunk_lines = lines[start:start + CHUNK_FILE_DEFAULT_SIZE]
        chunks.append({
            "index": start // CHUNK_FILE_DEFAULT_SIZE,
            "start_line": start + 1,
            "end_line": start + len(chunk_lines),
            "text": "\n".join(chunk_lines),
        })
    logger.info("[coordinator:chunk_file] %s -> %d chunks of up to %d lines", path, len(chunks), CHUNK_FILE_DEFAULT_SIZE)
    return chunks


async def _append_text(inputs: dict[str, Any], working_directory: str | None) -> dict:
    """Append text plus a trailing newline to a file, creating parent dirs if needed. No confirmation — the file was already confirmed at creation."""
    path = inputs.get("path")
    text = inputs.get("text") or ""
    if path is None or working_directory is None:
        return {"success": False}

    absolute_path = resolve_workspace_path(path, working_directory)
    if not file_in_directory(str(absolute_path), working_directory):
        raise ValueError(f"Writing outside workspace is forbidden: {path}")

    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    with open(absolute_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return {"success": True}


async def _run_script(
    inputs: dict[str, Any],
    working_directory: str | None,
    workflow_directory: Path | None,
) -> dict:
    """Run a script bundled with the workflow (path relative to the workflow's own directory).

    Generic on purpose: any workflow can carry its own scripts/ next to its YAML and invoke them
    this way, the same way `agents/` already lets a workflow carry its own agent definitions. The
    subprocess runs with the workspace as its cwd, so script arguments can use workspace-relative
    paths the same way an LLM stage would. If stdout is valid JSON it is also exposed as `data`,
    so a script can hand back structured values (e.g. a file list) for later stages to consume —
    otherwise `data` is None and only the raw text `output` is available.
    """
    script_rel = inputs.get("script")
    args = inputs.get("args") or []
    if script_rel is None or workflow_directory is None or working_directory is None:
        return {"success": False, "output": "run_script: missing script, workflow_directory, or working_directory", "data": None}

    script_path = (Path(workflow_directory) / script_rel).resolve()
    if not script_path.is_file():
        return {"success": False, "output": f"run_script: script not found: {script_path}", "data": None}

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script_path), *[str(a) for a in args],
        cwd=working_directory,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode("utf-8", errors="replace").strip()

    data = None
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        pass

    logger.info("[coordinator:run_script] %s %s -> exit=%d output_len=%d", script_rel, args, proc.returncode, len(output))
    return {"success": proc.returncode == 0, "output": output, "data": data}


async def _chunk_json_entries(inputs: dict[str, Any], working_directory: str | None) -> list[dict]:
    """Split a JSON key-value file into fixed-size groups of entries, deterministically — no model involved.

    Each chunk carries `entries` ({key, value} pairs, shown to the model as translation input) and
    `keys` (the same keys alone, kept aside as the trusted list append_json_entries validates the
    count against and writes with — the model's own copy of a key, echoed back in its response, is
    never used for the actual write).
    """
    path = inputs.get("path")
    if path is None or working_directory is None:
        return []

    absolute_path = resolve_workspace_path(path, working_directory)
    if not file_in_directory(str(absolute_path), working_directory):
        raise ValueError(f"Reading outside workspace is forbidden: {path}")

    raw = Path(absolute_path).read_text(encoding="utf-8")
    data = json.loads(raw) if raw.strip() != "" else {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object at the top level")

    items = list(data.items())
    chunks = []
    for start in range(0, len(items), CHUNK_JSON_ENTRIES_DEFAULT_SIZE):
        group = items[start:start + CHUNK_JSON_ENTRIES_DEFAULT_SIZE]
        chunks.append({
            "index": start // CHUNK_JSON_ENTRIES_DEFAULT_SIZE,
            "entries": [{"key": key, "value": value} for key, value in group],
            "keys": [key for key, _ in group],
        })
    logger.info("[coordinator:chunk_json_entries] %s -> %d chunks of up to %d entries", path, len(chunks), CHUNK_JSON_ENTRIES_DEFAULT_SIZE)
    return chunks


async def _append_json_entries(inputs: dict[str, Any], working_directory: str | None) -> dict:
    """Merge one chunk's translated entries into the accumulated JSON output, rewriting the whole
    file so it stays valid JSON after every chunk rather than only at the very end.

    Only the entry COUNT is checked against the chunk's trusted `keys` — a mismatch means the model
    dropped or merged entries, and the stage fails so the loop retries the chunk. The model's own
    echoed key in each entry (present so the model has something to stay aligned to, and so the run
    view reads legibly) is never used for the write or the validation; the coordinator always writes
    using its own trusted `keys`, matched to `entries` by position.
    """
    path = inputs.get("path")
    entries = inputs.get("entries") or []
    keys = inputs.get("keys") or []
    if path is None or working_directory is None:
        return {"success": False, "written": 0, "expected": len(keys), "received": len(entries)}

    if len(entries) != len(keys):
        logger.warning(
            "[coordinator:append_json_entries] count mismatch for %s: expected %d, got %d",
            path, len(keys), len(entries),
        )
        return {"success": False, "written": 0, "expected": len(keys), "received": len(entries)}

    absolute_path = resolve_workspace_path(path, working_directory)
    if not file_in_directory(str(absolute_path), working_directory):
        raise ValueError(f"Writing outside workspace is forbidden: {path}")

    raw = absolute_path.read_text(encoding="utf-8") if absolute_path.exists() else ""
    accumulated: dict[str, Any] = json.loads(raw) if raw.strip() != "" else {}

    for key, entry in zip(keys, entries):
        accumulated[key] = entry.get("translated_value") if isinstance(entry, dict) else entry

    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(json.dumps(accumulated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("[coordinator:append_json_entries] %s -> wrote %d entries (total now %d)", path, len(keys), len(accumulated))
    return {"success": True, "written": len(keys), "expected": len(keys), "received": len(entries)}
