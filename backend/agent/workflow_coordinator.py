from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.file_utils import file_in_directory, resolve_workspace_path, load_ignore_spec, is_path_ignored

if TYPE_CHECKING:
    from agent.agent import AgentSession

logger = logging.getLogger(__name__)

CHUNK_JSON_ENTRIES_DEFAULT_SIZE = 10

# chunk_file targets a token budget per chunk, estimated from character count (~4 chars/token for
# English/code text) rather than a fixed line count — a fixed line count is a poor proxy for chunk
# size since a single line can be a handful of characters or several thousand (minified code, a
# long string literal, a big one-line JSON blob).
CHUNK_FILE_TARGET_TOKENS = 12000
CHUNK_FILE_CHARS_PER_TOKEN_ESTIMATE = 4
CHUNK_FILE_MAX_CHUNK_CHARS = CHUNK_FILE_TARGET_TOKENS * CHUNK_FILE_CHARS_PER_TOKEN_ESTIMATE

# Extensions treated as non-text for enumerate_files — no useful summary comes from reading these
# as source, and several (images, archives) can be large enough to matter.
_ENUMERATE_FILES_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".mov", ".avi", ".webm",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pdf", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".wasm",
    ".gguf", ".safetensors", ".onnx", ".pt", ".pth",
    ".sqlite", ".sqlite3", ".db",
}

# Generated lockfiles: authored by tooling, not humans, and disproportionately large relative to
# their information content — never worth a per-file summary.
_ENUMERATE_FILES_LOCKFILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "uv.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum",
}


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
    if action == "enumerate_files":
        return await _enumerate_files(inputs, working_directory)
    if action == "append_jsonl_record":
        return await _append_jsonl_record(inputs, working_directory)
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


def _split_oversized_line(line: str, max_chars: int) -> list[str]:
    """Split one line that alone exceeds the chunk budget into fixed-size character slices."""
    return [line[i:i + max_chars] for i in range(0, len(line), max_chars)] or [""]


async def _chunk_file(inputs: dict[str, Any], working_directory: str | None) -> list[dict]:
    """Split a file into chunks targeting a token budget (estimated from character count, see
    CHUNK_FILE_TARGET_TOKENS), not a fixed line count. Returns [{index, start_line, end_line,
    text}], deterministically — no model involved.

    Lines are packed into a chunk until the next line would push it over the budget, at which
    point the chunk closes and a new one starts. A single line that alone exceeds the budget
    (a long minified line, a big one-line JSON blob) is split into its own character-bounded
    pieces rather than left in one oversized chunk — every piece keeps that line's own
    start_line/end_line since it's still the same source line.
    """
    path = inputs.get("path")
    if path is None or working_directory is None:
        return []

    absolute_path = resolve_workspace_path(path, working_directory)
    if not file_in_directory(str(absolute_path), working_directory):
        raise ValueError(f"Reading outside workspace is forbidden: {path}")

    lines = Path(absolute_path).read_text(encoding="utf-8").splitlines()

    chunks: list[dict] = []
    current_lines: list[str] = []
    current_chars = 0
    current_start_line = 1

    def flush(end_line: int) -> None:
        nonlocal current_lines, current_chars, current_start_line
        if len(current_lines) == 0:
            return
        chunks.append({
            "index": len(chunks),
            "start_line": current_start_line,
            "end_line": end_line,
            "text": "\n".join(current_lines),
        })
        current_lines = []
        current_chars = 0

    for line_number, line in enumerate(lines, start=1):
        line_chars = len(line) + 1  # +1 for the newline packed back in via "\n".join

        if line_chars > CHUNK_FILE_MAX_CHUNK_CHARS:
            flush(line_number - 1)
            current_start_line = line_number
            for piece in _split_oversized_line(line, CHUNK_FILE_MAX_CHUNK_CHARS):
                chunks.append({
                    "index": len(chunks),
                    "start_line": line_number,
                    "end_line": line_number,
                    "text": piece,
                })
            current_start_line = line_number + 1
            continue

        if len(current_lines) > 0 and current_chars + line_chars > CHUNK_FILE_MAX_CHUNK_CHARS:
            flush(line_number - 1)
            current_start_line = line_number

        current_lines.append(line)
        current_chars += line_chars

    flush(len(lines))
    logger.info("[coordinator:chunk_file] %s -> %d chunk(s) targeting ~%d chars each", path, len(chunks), CHUNK_FILE_MAX_CHUNK_CHARS)
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

    A plain string arg (a path, a flag like "--apply") is passed through unchanged, exactly as
    before. Anything else — a list, dict, None, bool, number, e.g. from a bare {{slot.field}} that
    resolved to a real Python value rather than a string — is turned into JSON text via json.dumps
    (None -> "null", a missing slot resolving to None included) so a script can always json.loads
    a non-string arg rather than receiving a Python repr it can't parse.
    """
    script_rel = inputs.get("script")
    args = inputs.get("args") or []
    if script_rel is None or workflow_directory is None or working_directory is None:
        return {"success": False, "output": "run_script: missing script, workflow_directory, or working_directory", "data": None}

    script_path = (Path(workflow_directory) / script_rel).resolve()
    if not script_path.is_file():
        return {"success": False, "output": f"run_script: script not found: {script_path}", "data": None}

    argv = [a if isinstance(a, str) else json.dumps(a, ensure_ascii=False) for a in args]
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script_path), *argv,
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


async def _enumerate_files(inputs: dict[str, Any], working_directory: str | None) -> list[dict]:
    """Recursively list files under `root`, deterministically — no model involved.

    Reuses the SAME gitignore/hardcoded-dir filtering glob_files already applies. Excludes
    non-text/binary files (by extension) and generated lockfiles — everything else is included
    regardless of size, since the workflow that consumes this always chunks rather than reading a
    file whole. `exclude` (optional list of workspace-relative directory paths) additionally skips
    anything under those directories — e.g. a workflow writing its own output back into the
    scanned tree needs to exclude that output directory, or a rerun ends up documenting its own
    previous output. Returns [{path, extension, size_bytes}], workspace-relative posix paths,
    capped at `max_files` (default 300); a truncated result is reported honestly rather than
    silently.
    """
    root = inputs.get("root") or "."
    max_files = inputs.get("max_files") or 300
    exclude = inputs.get("exclude") or []
    if working_directory is None:
        return []

    absolute_root = resolve_workspace_path(root, working_directory)
    if not file_in_directory(str(absolute_root), working_directory):
        raise ValueError(f"Scanning outside workspace is forbidden: {root}")

    exclude_roots = [resolve_workspace_path(e, working_directory) for e in exclude]
    spec = load_ignore_spec(working_directory)

    def _walk() -> list[dict]:
        results = []
        for p in sorted(absolute_root.rglob("*")):
            if not p.is_file():
                continue
            if is_path_ignored(p, working_directory, spec):
                continue
            if any(p.is_relative_to(ex) for ex in exclude_roots):
                continue
            if p.suffix.lower() in _ENUMERATE_FILES_BINARY_EXTENSIONS:
                continue
            if p.name in _ENUMERATE_FILES_LOCKFILE_NAMES:
                continue
            results.append({
                "path": p.relative_to(working_directory).as_posix(),
                "extension": p.suffix,
                "size_bytes": p.stat().st_size,
            })
        return results

    files = await asyncio.to_thread(_walk)
    truncated = len(files) > max_files
    if truncated:
        files = files[:max_files]
    logger.info("[coordinator:enumerate_files] %s -> %d file(s)%s", root, len(files), " (truncated)" if truncated else "")
    return {"files": files, "truncated": truncated}


async def _append_jsonl_record(inputs: dict[str, Any], working_directory: str | None) -> dict:
    """Append one compact JSON line to `path`, creating parent dirs if needed.

    Generic on purpose (like append_text/append_json_entries): any workflow building a structured,
    line-delimited log can use this, not just codebase mapping. No confirmation — same reasoning as
    append_text, the file was already confirmed at creation.
    """
    path = inputs.get("path")
    record = inputs.get("record")
    if path is None or working_directory is None:
        return {"success": False}

    absolute_path = resolve_workspace_path(path, working_directory)
    if not file_in_directory(str(absolute_path), working_directory):
        raise ValueError(f"Writing outside workspace is forbidden: {path}")

    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    with open(absolute_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"success": True}
