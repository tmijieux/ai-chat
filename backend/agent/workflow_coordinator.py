from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.file_utils import file_in_directory, resolve_workspace_path

if TYPE_CHECKING:
    from agent.agent import AgentSession

logger = logging.getLogger(__name__)

CHUNK_FILE_DEFAULT_SIZE = 10


async def run_coordinator_action(
    action: str,
    inputs: dict[str, Any],
    session: "AgentSession",
    working_directory: str | None,
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
