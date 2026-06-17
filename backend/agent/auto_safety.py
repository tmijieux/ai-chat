import json
import logging
import os

logger = logging.getLogger(__name__)

_ALWAYS_SAFE_TOOLS = frozenset({
    "read_file",
    "read_file_range",
    "glob_files",
    "grep_files",
    "list_directory",
    "explore_codebase",
})

_FILE_WRITE_TOOLS = frozenset({"write_file", "edit_file"})

_EVAL_SYSTEM = """\
You are a safety evaluator for an autonomous coding agent.
Given a tool call and the user's current task, decide whether running this tool is safe.

Safe: the action is reversible or clearly scoped to the task, limited blast radius.
Dangerous: destructive, out-of-scope, could cause data loss, unintended network calls,
or shell commands not clearly related to the user task.

Respond with JSON only — no prose, no markdown fences.
Example: {"verdict": "safe", "reason": "installs a dev dependency listed in the task"}
"""


def is_path_inside_workspace(path: str, working_directory: str) -> bool:
    """Return True if path resolves to a location inside working_directory."""
    try:
        p = os.path.abspath(path) if os.path.isabs(path) else os.path.join(working_directory, path)
        abs_path = os.path.realpath(p)
        abs_workspace = os.path.realpath(os.path.abspath(working_directory))
        return abs_path == abs_workspace or abs_path.startswith(abs_workspace + os.sep)
    except (TypeError, ValueError):
        return False


async def evaluate_tool_safety(
    tool_name: str,
    arguments: dict,
    working_directory: str | None,
    last_user_message: str,
    llm_backend,
    safe_command_prefixes: list[str] | None = None,
) -> tuple[str, str]:
    """
    Return ("safe"|"dangerous", reason).

    Rule-based checks run first (no LLM call).  Falls back to an LLM
    evaluation only for run_shell, search_web, and out-of-workspace writes.
    """
    if tool_name in _ALWAYS_SAFE_TOOLS:
        return "safe", "read-only tool"

    if tool_name == "run_shell" and safe_command_prefixes is not None:
        command = arguments.get("command", "").strip()
        excluded = any(
            command.startswith(p[1:])
            for p in safe_command_prefixes
            if p.startswith("!")
        )
        if not excluded:
            for prefix in safe_command_prefixes:
                if not prefix.startswith("!") and command.startswith(prefix):
                    return "safe", "workflow whitelist"

    if tool_name in _FILE_WRITE_TOOLS:
        path = arguments.get("file_path", "")
        if working_directory is not None and is_path_inside_workspace(path, working_directory):
            return "safe", "in-workspace write"

    prompt = (
        f"User task: {last_user_message[:500]}\n"
        f"Tool: {tool_name}\n"
        f"Arguments: {json.dumps(arguments, ensure_ascii=False)[:800]}"
    )
    messages = [{"role": "user", "content": prompt}]
    prepared = llm_backend.prepare_messages(messages)

    content = ""
    async for event in llm_backend.stream_completion(
        prepared, [], temperature=0.0, max_tokens=128, disable_thinking=True
    ):
        if event["type"] == "content":
            content += event["content"]

    content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
            verdict = parsed.get("verdict", "dangerous")
            reason = str(parsed.get("reason", ""))
            if verdict not in ("safe", "dangerous"):
                verdict = "dangerous"
            logger.info("safety eval %s → %s: %s", tool_name, verdict, reason[:80])
            return verdict, reason
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("safety evaluator unparseable response for %s: %r", tool_name, content[:200])
    return "dangerous", "could not parse evaluator response — defaulting to dangerous"
