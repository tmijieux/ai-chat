from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.agent import AgentSession

logger = logging.getLogger(__name__)


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
