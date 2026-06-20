import datetime
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from database import AsyncSession
import tables as db
from llm.base import LLMBackend
from message_types import LLMMessage, TrackedMessage
from tool_result_types import (
    GlobFilesResult,
    GrepFilesResult,
    ReadFileResult,
    RunShellResult,
    SearchWebResult,
    ToolResult
)
from conv_helpers import _now, _build_active_branch_path, _parse_conv_settings
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list

logger = logging.getLogger(__name__)


@dataclass
class ClassifiablePair:
    """One tool message prepared for the classifier: index into candidate_messages plus LLM prompt metadata."""
    index: int
    message_id: str
    tool_call_id: str
    tool: str
    args_summary: str
    result_metadata: dict
    following_thinking: str


@dataclass
class Compression:
    """A single compressed tool message: which message was compressed, the summary text, and the classifier label."""
    message_id: str
    compressed_summary: str
    compression_label: str


@dataclass
class CompressionResult:
    """Outcome of a compress_messages call: the list of per-message compressions and the updated conversation summary."""
    compressions: list[Compression]
    new_summary: str


KEEP_SUMMARIZE_THRESHOLD_CHARS = 800  # ~200 tokens — threshold for auto-summarizing "keep" items
CHUNK_MAX_CHARS = 32_000  # ~8k tokens per chunk, leaves headroom for agent context
_GLOB_MAX_FILES = 80
_GREP_MAX_CHARS = 4000

_SKIP_CLASSIFY = {"write_file", "edit_file", "ask_user_question", "propose_plan"}


def _following_thinking(all_messages: list[TrackedMessage], tool_id: str) -> str:
    """Return up to 600 chars of thinking from the first assistant message after the given tool result."""
    found = False
    for m in all_messages:
        if not found:
            if m["role"] == "tool":
                assert isinstance(m["content"], str)
                try:
                    c = json.loads(m["content"])
                    if c.get("tool_call_id") == tool_id:
                        found = True
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            continue

        if m.get("role") == "assistant":
            assert isinstance(m["content"], str)
            text: str = m["thinking"] or m["content"] or ""
            return text[:600]
    return ""


def _key_args(result: ToolResult) -> str:
    tool = result.get("tool", "")
    if tool in ("read_file", "list_directory"):
        return repr(result.get("path", ""))
    if tool == "glob_files":
        return repr(result.get("pattern", ""))
    if tool == "grep_files":
        return repr(result.get("pattern", ""))
    if tool == "run_shell":
        return repr((result.get("command") or "")[:80])
    if tool == "search_web":
        return repr((result.get("query") or "")[:80])
    return ""


def _result_metadata(result: ToolResult) -> dict[str, Any]:
    tool = result.get("tool", "")
    meta: dict[str,str|int] = {"status": result.get("status", "unknown")}
    if tool == "glob_files":
        meta["file_count"] = result.get("file_count", 0)
    elif tool == "grep_files":
        matches = result.get("matches", [])
        meta["match_count"] = len(matches) if isinstance(matches, list) else 0
    elif tool == "list_directory":
        content = result.get("content") or ""
        meta["entry_count"] = len(content.splitlines())
    elif tool == "read_file":
        content = result.get("file_content") or ""
        meta["line_count"] = len(content.splitlines())
    elif tool == "run_shell":
        meta["exit_code"] = result.get("exit_code", 0)
        output = (result.get("output") or "") + (result.get("stderr") or "")
        meta["line_count"] = len(output.splitlines())
    elif tool == "search_web":
        results_list = result.get("results") or []
        meta["result_count"] = len(results_list)
        meta["total_chars"] = sum(len(r.get("content") or "") for r in results_list)
    return meta


def _compact_summary(result: ToolResult) -> str:
    tool = result.get("tool", "unknown")
    key = _key_args(result)
    status = result.get("status", "unknown")

    if status == "rejected":
        reason = result.get("reason") or ""
        suffix = f": {reason[:120]}" if reason else ""
        return f'{tool}({key}) → rejected{suffix}'

    if status == "error":
        error_msg = (result.get("error") or {}).get("message", "error")
        return f'{tool}({key}) → error: {error_msg[:120]}'

    meta = _result_metadata(result)

    if tool == "glob_files":
        n = meta.get("file_count", 0)
        return f'glob_files({key}) → {n} file{"s" if n != 1 else ""}'
    if tool == "grep_files":
        n = meta.get("match_count", 0)
        return f'grep_files({key}) → {n} match{"es" if n != 1 else ""}'
    if tool == "list_directory":
        n = meta.get("entry_count", 0)
        return f'list_directory({key}) → {n} entr{"ies" if n != 1 else "y"}'
    if tool == "read_file":
        n = meta.get("line_count", 0)
        return f'read_file({key}) → {n} line{"s" if n != 1 else ""}'
    if tool == "run_shell":
        code = meta.get("exit_code", "?")
        n = meta.get("line_count", 0)
        return f'run_shell({key}) → exit {code}, {n} line{"s" if n != 1 else ""}'
    if tool == "search_web":
        n = meta.get("result_count", 0)
        return f'search_web({key}) → {n} result{"s" if n != 1 else ""}'
    return f'{tool}({key}) → {meta.get("status", "?")}'


def _extract_json(text: str) -> str:
    """Strip markdown fences and find the outermost JSON object."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()

    # Find the outermost { ... } in case there's surrounding prose
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


async def _llm_complete(prompt: str, backend: LLMBackend, system: str | None = None) -> str:
    """Call the LLM and return the response content.

    Falls back to reasoning_content (thinking) if content is empty or a bare
    period — Qwen3 thinking models sometimes place the answer there.
    Thinking is disabled (budget_tokens=0) to avoid wasting tokens on reasoning.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    prepared = backend.prepare_messages(messages)
    content = ""
    thinking = ""
    async for event in backend.stream_completion(prepared, [], temperature=0.1, max_tokens=1024, disable_thinking=True):
        if event["type"] == "content":
            content += event["content"]
        elif event["type"] == "thinking":
            thinking += event["content"]

    result = content.strip()
    if not result or result in (".", ".."):
        logger.debug(
            "_llm_complete: content empty/bare-dot (%r), falling back to thinking (%d chars)",
            result,
            len(thinking),
        )
        result = thinking.strip()
    else:
        logger.debug(
            "_llm_complete: content=%d chars, thinking=%d chars (discarded)",
            len(result),
            len(thinking),
        )

    return result


_REPORT_CLASSIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "report_classification",
        "description": "Report the compression classification for each tool call result.",
        "parameters": {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "description": "One entry per tool call, in any order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The id of the tool result being classified (copy it verbatim from the input).",
                            },
                            "label": {
                                "type": "string",
                                "enum": ["drop", "1-line-summary", "summarize", "keep"],
                            },
                            "reason": {
                                "type": "string",
                                "description": "One sentence explaining why this label was chosen for this specific tool result.",
                            },
                            "line_summary": {
                                "type": "string",
                                "description": 'Required when label is "1-line-summary". One factual line: specific names, numbers, paths. No prose.',
                            },
                        },
                        "required": ["id", "label", "reason"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "One sentence describing what the agent accomplished (or is trying to accomplish for mid-run compression).",
                },
            },
            "required": ["classifications", "summary"],
        },
    },
}


async def _llm_classify(prompt: str, backend: LLMBackend, system: str) -> dict:
    """Call the LLM and return the parsed report_classification tool call arguments."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    prepared = backend.prepare_messages(messages)
    tool_args_str = ""
    async for event in backend.stream_completion(
        prepared, [_REPORT_CLASSIFICATION_TOOL], temperature=0.1, max_tokens=1024, disable_thinking=True,
    ):
        if event["type"] == "tool_call_arg":
            tool_args_str += event["fragment"]
    if not tool_args_str:
        return {}
    try:
        return json.loads(tool_args_str)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("_llm_classify: failed to parse tool args: %s — raw: %r", e, tool_args_str[:300])
        return {}


_CLASSIFY_LABELS_DOC = """\
Assign exactly one label per tool call:

  drop            — result is fully consumed; nothing useful remains.
                    Use for: navigation globs/greps, directory listings used to pick a file,
                    files read then immediately edited, commands that succeeded and the agent
                    already acted on them.
                    Never drop a run_shell with non-zero exit code — the error output is
                    diagnostic and must be summarized or kept.
                    The orchestrator keeps only a metadata stub (tool + status).

  1-line-summary  — result is consumed but one factual line is worth preserving.
                    Write the summary yourself in the "line_summary" field.
                    One line, facts only — specific names, numbers, paths. No prose.
                    Examples:
                      grep  → "matched in src/auth.ts:42 and src/user.ts:87 (3 files total)"
                      glob  → "12 files matched: src/components/Button.tsx, Modal.tsx, …"
                      shell → "exit 0 — 3 warnings: unused import in auth.ts:12,14,18"
                      file  → "UserService class — getUser/createUser/deleteUser, imports prisma"

  summarize       — result is still relevant but too large or verbose to keep verbatim.
                    The orchestrator will generate a paragraph summary.
                    Use for: large files the agent needs context from, web search results,
                    long shell outputs with errors still being investigated.

  keep            — agent will reference exact lines or content in the very next step.
                    Use sparingly. The orchestrator may still shorten it if very large.

Prefer drop > 1-line-summary > summarize > keep. Use keep only when exact content matters.\
"""

_CLASSIFY_EXAMPLE = """\
Example input:
User goal: "add a new route to the app config"
Tool calls:
[
  {"id": "msg_a1b2", "tool": "glob_files", "args_summary": "'**/*.config.ts'", "result_metadata": {"file_count": 3}, "following_thinking": "Found 3 files. Let me read the right one."},
  {"id": "msg_c3d4", "tool": "read_file",  "args_summary": "'src/app/app.config.ts'",  "result_metadata": {"line_count": 42}, "following_thinking": "provideRouter is called with the routes array — I need to add the new route here."},
  {"id": "msg_e5f6", "tool": "grep_files", "args_summary": "'provideRouter'",           "result_metadata": {"match_count": 1}, "following_thinking": "Found it at line 18. I will edit the file now."}
]

Example report_classification call:
  classifications: [
    {"id": "msg_a1b2", "label": "drop",           "reason": "Glob used only for navigation — agent moved on immediately."},
    {"id": "msg_c3d4", "label": "summarize",       "reason": "Agent needs the file structure but not verbatim lines."},
    {"id": "msg_e5f6", "label": "1-line-summary",  "reason": "Single fact worth preserving — exact location already noted.", "line_summary": "provideRouter call at src/app/app.config.ts:18"}
  ]
  summary: "Agent located the route config and is about to add the new route."\
"""

_CLASSIFY_SYSTEM_POST_RUN = f"""\
You are a context compression subagent. An AI agent has just completed a run. \
Classify each tool result so the context stays as small as possible while keeping \
what the agent might still need. Call report_classification with your results.

{_CLASSIFY_LABELS_DOC}

Primary signal: following_thinking.
  If the next thought cites specific lines/content verbatim → keep.
  If it uses the result conceptually → 1-line-summary or summarize.
  If it moved on entirely → drop.

{_CLASSIFY_EXAMPLE}\
"""

_CLASSIFY_SYSTEM_MID_RUN = f"""\
You are a context compression subagent. An AI agent hit the context limit mid-task \
and must continue after compression. Classify each tool result to free as much space \
as possible. Call report_classification with your results.

The agent has NOT finished. Judge each result by whether its content will be needed \
in upcoming steps — not by whether it was useful in past steps.

{_CLASSIFY_LABELS_DOC}

Primary signal: following_thinking.
  If the next thought cites specific lines/content verbatim → keep.
  If it uses the result conceptually → 1-line-summary or summarize.
  If it moved on entirely → drop.

{_CLASSIFY_EXAMPLE}\
"""


@dataclass
class ClassifyResult:
    """Output of _classify_and_summarize."""
    labels: dict[str, str]
    line_summaries: dict[str, str]
    reasonings: dict[str, str]
    conversation_summary: str


async def _classify_and_summarize(
    pairs: list[ClassifiablePair],
    user_message: str,
    conversation_summary: str | None,
    backend: LLMBackend,
    is_mid_run: bool = False,
) -> ClassifyResult:
    """Classify tool results and return labels, line summaries, per-label reasonings, and an updated conversation summary."""
    system = _CLASSIFY_SYSTEM_MID_RUN if is_mid_run else _CLASSIFY_SYSTEM_POST_RUN
    conv_line = f"Conversation so far: {conversation_summary}\n" if conversation_summary else ""
    llm_pairs = [
        {
            "id": p.message_id,
            "tool": p.tool,
            "args_summary": p.args_summary,
            "result_metadata": p.result_metadata,
            "following_thinking": p.following_thinking,
        }
        for p in pairs
    ]

    prompt = f"""\
User's goal: {user_message}
{conv_line}
Tool calls:
{json.dumps(llm_pairs, ensure_ascii=False, indent=2)}\
"""

    t0 = time.perf_counter()
    parsed = await _llm_classify(prompt, backend, system=system)
    elapsed = time.perf_counter() - t0
    logger.info("classify LLM call: %.1fs", elapsed)

    if not parsed:
        logger.warning("compress classify: empty result — defaulting all to drop")
        return ClassifyResult(
            labels={p.message_id: "drop" for p in pairs},
            line_summaries={},
            reasonings={},
            conversation_summary=conversation_summary or "",
        )

    labels: dict[str, str] = {}
    reasonings: dict[str, str] = {}
    line_summaries: dict[str, str] = {}
    for item in parsed.get("classifications") or []:
        message_id = str(item.get("id", ""))
        if message_id == "":
            continue
        labels[message_id] = str(item.get("label", "drop"))
        reason = str(item.get("reason", "")).strip()
        if reason != "":
            reasonings[message_id] = reason
        line_summary = str(item.get("line_summary", "")).strip()
        if line_summary != "":
            line_summaries[message_id] = line_summary

    summary = str(parsed.get("summary", ""))
    logger.info("classify summary: %s", summary[:120])
    for message_id, label in labels.items():
        reason = reasonings.get(message_id, "")
        logger.info("  [%s] %s — %s", message_id, label, reason[:100])
    return ClassifyResult(
        labels=labels,
        line_summaries=line_summaries,
        reasonings=reasonings,
        conversation_summary=summary,
    )


_SUMMARIZE_FILE_SYSTEM = """\
Summarize the given file for use as context in a coding agent. Output:
1. One-sentence module purpose
2. All public functions/classes: full signature with type hints + one-line description
3. Key constants and config values
4. External imports

Be terse. No examples. No prose beyond descriptions.\
"""


_SUMMARIZE_SHELL_SYSTEM = """\
Summarize the following shell command output (one chunk of a potentially larger output).
Focus on: errors, warnings, key results, important file paths or values, exit signals.
Be terse. Preserve exact error messages and stack traces verbatim. No prose beyond the summary.\
"""


_SUMMARIZE_SEARCH_SYSTEM = """\
Summarize the following web search results for use in a coding agent context.
Keep only facts, code snippets, API references, and technical details relevant to the query.
Be terse. No prose beyond the summary. Output plain text.\
"""


def _split_by_lines(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars characters, breaking only on line boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > max_chars and len(current) > 0:
            chunks.append("".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if len(current) > 0:
        chunks.append("".join(current))
    return chunks


def _compact_glob_result(result: GlobFilesResult) -> str:
    """Truncate a glob result to _GLOB_MAX_FILES paths, sorted shallowest-first."""
    files: list[str] = result.get("files") or []
    total = len(files)
    sorted_files = sorted(files, key=lambda p: (p.count("/") + p.count("\\"), p))
    kept = sorted_files[:_GLOB_MAX_FILES]
    lines = "\n".join(kept)
    suffix = f"\n… {total - _GLOB_MAX_FILES} more files not shown" if total > _GLOB_MAX_FILES else ""
    pattern = result.get("pattern", "")
    return f"[truncated: glob_files({repr(pattern)}) → {total} files]\n{lines}{suffix}"


def _compact_grep_result(result: GrepFilesResult) -> str:
    """Keep all match lines verbatim; drop context lines until under _GREP_MAX_CHARS."""
    matches: list[dict] = result.get("matches") or []
    pattern = result.get("pattern", "")
    total = result.get("total", len(matches))

    match_lines = [m for m in matches if m.get("match")]
    context_lines = [m for m in matches if not m.get("match")]

    def _render(items: list[dict]) -> str:
        parts = []
        current_file = None
        for m in items:
            if m["file"] != current_file:
                current_file = m["file"]
                parts.append(f"--- {current_file}")
            parts.append(f"{m['line']:>6} | {m['content']}")
        return "\n".join(parts)

    # Start with all match lines; add context lines until budget exceeded
    kept_context: list[dict] = []
    for ctx in context_lines:
        candidate = _render(match_lines + kept_context + [ctx])
        if len(candidate) <= _GREP_MAX_CHARS:
            kept_context.append(ctx)
        else:
            break

    omitted_ctx = len(context_lines) - len(kept_context)
    suffix = f"\n… {omitted_ctx} context lines omitted" if omitted_ctx > 0 else ""
    all_kept = sorted(match_lines + kept_context, key=lambda m: (m["file"], m["line"]))
    header = f"[truncated: grep_files({repr(pattern)}) → {total} matches]\n"
    return header + _render(all_kept) + suffix


async def _summarize_shell_output(result: RunShellResult, backend: LLMBackend) -> str:
    """Summarize a run_shell tool result, splitting large output into chunks if needed."""
    command_raw = result.get("command")
    command = (command_raw if command_raw is not None else "")[:80]

    raw_output = result.get("output")
    if raw_output is None or raw_output == "":
        error_dict = result.get("error")
        if error_dict is None:
            error_dict = {}
        raw_output = error_dict.get("message")
        if raw_output is None:
            raw_output = ""

    exit_code = 0 if result.get("status") == "success" else 1
    chunks = _split_by_lines(raw_output, CHUNK_MAX_CHARS)
    chunk_count = len(chunks)
    summaries = []
    for i, chunk in enumerate(chunks):
        header = f"[chunk {i + 1}/{chunk_count}]\n" if chunk_count > 1 else ""
        summary = await _llm_complete(header + chunk, backend, system=_SUMMARIZE_SHELL_SYSTEM)
        summaries.append(summary)

    combined = "\n\n---\n\n".join(summaries) if chunk_count > 1 else summaries[0]
    estimated_tokens = len(combined) // 4
    logger.info(
        "summarize_shell %r: %d chunk(s) → %d chars (~%d tokens)",
        command, chunk_count, len(combined), estimated_tokens,
    )
    return (
        f"[compressed: run_shell({repr(command)}) → exit {exit_code}, "
        f"{len(raw_output.splitlines())} lines, {chunk_count} chunk(s) → ~{estimated_tokens} tokens]\n"
        + combined
    )


async def _summarize_search_results(result: SearchWebResult, backend: LLMBackend) -> str:
    query = result.get("query", "unknown")
    results_list = result.get("results") or []

    # Build one text block per result, then chunk the whole lot.
    parts = []
    for r in results_list:
        url = r.get("url", "")
        body = r.get("content") or ""
        if body:
            parts.append(f"URL: {url}\n{body}")

    full_text = f"Query: {query}\n\n" + "\n\n---\n\n".join(parts)
    chunks = _split_by_lines(full_text, CHUNK_MAX_CHARS)
    chunk_count = len(chunks)
    summaries = []
    for i, chunk in enumerate(chunks):
        header = f"[chunk {i + 1}/{chunk_count}]\n" if chunk_count > 1 else ""
        summaries.append(await _llm_complete(header + chunk, backend, system=_SUMMARIZE_SEARCH_SYSTEM))

    combined = "\n\n---\n\n".join(summaries) if chunk_count > 1 else summaries[0]
    est_tokens = len(combined) // 4
    logger.info(
        "summarize_search %r: %d chunk(s) → %d chars (~%d tokens)",
        query[:40], chunk_count, len(combined), est_tokens,
    )

    return f"[compressed: search_web({repr(query)}) → {len(results_list)} result(s), {chunk_count} chunk(s) → ~{est_tokens} tokens]\n{combined}"


async def _summarize_file(result: ReadFileResult, backend: LLMBackend) -> str:
    path = result.get("path", "unknown")
    content = result.get("file_content") or ""
    line_count = len(content.splitlines())

    chunks = _split_by_lines(f"File: {path}\n---\n{content}", CHUNK_MAX_CHARS)
    chunk_count = len(chunks)
    summaries = []
    for i, chunk in enumerate(chunks):
        header = f"[chunk {i + 1}/{chunk_count}]\n" if chunk_count > 1 else ""
        summaries.append(await _llm_complete(header + chunk, backend, system=_SUMMARIZE_FILE_SYSTEM))

    combined = "\n\n---\n\n".join(summaries) if chunk_count > 1 else summaries[0]
    est_tokens = len(combined) // 4
    logger.info(
        "summarize_file %s: %d chunk(s) → %d chars (~%d tokens)",
        path, chunk_count, len(combined), est_tokens,
    )

    return f"[compressed: read_file({repr(path)}) → {line_count} lines, {chunk_count} chunk(s) → ~{est_tokens} tokens — prefer grep_files to extract snippets, read_file as last resort]\n{combined}"


def _build_classifiable_pairs(
    candidate_messages: list[TrackedMessage],
    all_messages: list[TrackedMessage],
) -> list[ClassifiablePair]:
    """Build the list of classifiable pairs from candidate tool messages.
    Filters out non-tool messages and tools in _SKIP_CLASSIFY, and attaches metadata needed by the classifier."""

    pairs: list[ClassifiablePair] = []
    for i, msg in enumerate(candidate_messages):
        if msg["role"] != "tool":
            continue
        try:
            content = msg["content"]
            if isinstance(content, str):
                result = json.loads(content)
            else:
                logger.warning("unexpected multipart content in tool message; skipping")
                continue
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("_build_classifiable_pairs() unexpected error=", exc_info=e)
            continue

        tool: str = result.get("tool", "")
        if tool in _SKIP_CLASSIFY:
            logger.debug("skipping classify for %s (tool=%s)", msg.get("id"), tool)
            continue
        tool_call_id = result.get("tool_call_id", "")
        pairs.append(ClassifiablePair(
            index=i,
            message_id=msg["id"],
            tool_call_id=tool_call_id,
            tool=tool,
            args_summary=_key_args(result),
            result_metadata=_result_metadata(result),
            following_thinking=_following_thinking(all_messages, tool_call_id),
        ))
    return pairs


async def _apply_compression_label(
    label: str,
    tool: str,
    result: ToolResult,
    content_len: int,
    line_summary: str,
    reasoning: str,
    backend: LLMBackend,
) -> str | None:
    """Produce a compressed_summary for one tool message given its classifier label.
    Returns None for keep items that don't exceed any size threshold (full content is preserved)."""
    if label == "drop":
        # Never silently drop a shell command that exited non-zero — the error output is diagnostic.
        if tool == "run_shell" and result.get("exit_code", 0) != 0:
            label = "summarize"
        else:
            summary = _compact_summary(result)
            if reasoning != "":
                summary += f"\nReason: {reasoning}"
            logger.info("drop → compact: %s", summary)
            return summary

    if label == "1-line-summary":
        provided = line_summary.strip()
        summary = f"{_compact_summary(result)} — {provided}" if provided != "" else _compact_summary(result)
        if reasoning != "":
            summary += f"\nReason: {reasoning}"
        logger.info("1-line-summary: %s", summary)
        return summary

    if label == "summarize":
        if tool == "read_file":
            return await _summarize_file(result, backend)
        if tool == "search_web":
            return await _summarize_search_results(result, backend)
        if tool == "run_shell":
            return await _summarize_shell_output(result, backend)
        if tool == "glob_files":
            return _compact_glob_result(result)
        if tool == "grep_files":
            return _compact_grep_result(result)
        logger.info("summarize → done for %s", tool)
        return _compact_summary(result)

    if label == "keep":
        if tool == "read_file":
            file_content = result.get("file_content") or ""
            if len(file_content) > KEEP_SUMMARIZE_THRESHOLD_CHARS:
                logger.info("keep read_file over threshold (%d chars) → summarizing", len(file_content))
                return await _summarize_file(result, backend)
        elif tool == "search_web":
            results_list = result.get("results") or []
            total_chars = sum(len(r.get("content") or "") for r in results_list)
            if total_chars > KEEP_SUMMARIZE_THRESHOLD_CHARS:
                logger.info("keep search_web over threshold (%d chars) → summarizing", total_chars)
                return await _summarize_search_results(result, backend)
        elif tool == "run_shell":
            raw_output = result.get("output") or (result.get("error") or {}).get("message") or ""
            if len(raw_output) > KEEP_SUMMARIZE_THRESHOLD_CHARS:
                logger.info("keep run_shell over threshold (%d chars) → summarizing", len(raw_output))
                return await _summarize_shell_output(result, backend)
        elif tool == "glob_files":
            files = result.get("files") or []
            if len(files) > _GLOB_MAX_FILES:
                logger.info("keep glob_files over %d files → truncating", _GLOB_MAX_FILES)
                return _compact_glob_result(result)
        elif tool == "grep_files":
            if content_len > _GREP_MAX_CHARS:
                logger.info("keep grep_files over threshold (%d chars) → truncating", content_len)
                return _compact_grep_result(result)

    return None


async def compress_messages(
    candidate_messages: list[TrackedMessage],
    all_messages: list[TrackedMessage],
    user_message: str,
    conversation_summary: str | None,
    backend: LLMBackend,
    protect_last: bool = False,
    is_mid_run: bool = False,
) -> CompressionResult:
    """Classify and compress a list of tool result messages.

    candidate_messages: tool-role messages to consider for compression.
    all_messages: full ordered branch (for following_thinking lookup).

    Labels from classifier:
      drop           → metadata one-liner stub
      1-line-summary → LLM-provided one-line description
      summarize      → LLM paragraph summary
      keep           → full content, LLM-summarized only if over KEEP_SUMMARIZE_THRESHOLD_CHARS

    protect_last: promotes the last classifiable item from drop/1-line-summary to summarize."""
    t_total = time.perf_counter()

    pairs = _build_classifiable_pairs(candidate_messages, all_messages)
    logger.info(
        "compress_messages: %d candidate messages → %d classifiable pairs",
        len(candidate_messages), len(pairs),
    )

    if len(pairs) == 0:
        return CompressionResult(compressions=[], new_summary=conversation_summary or "")

    classify = await _classify_and_summarize(pairs, user_message, conversation_summary, backend, is_mid_run=is_mid_run)
    labels = classify.labels
    new_summary = classify.conversation_summary

    if protect_last and pairs:
        last_key = pairs[-1].message_id
        if labels.get(last_key) in ("drop", "1-line-summary"):
            labels[last_key] = "summarize"

    compressions: list[Compression] = []
    for p in pairs:
        label = labels.get(p.message_id, "drop")
        msg = candidate_messages[p.index]
        try:
            result = json.loads(msg.get("content") or "{}")
        except (json.JSONDecodeError, ValueError):
            continue

        tool = result.get("tool", "")
        content_len = len(msg.get("content") or "")
        logger.debug("pair index=%d tool=%s label=%s content_len=%d", p.index, tool, label, content_len)

        compressed_summary = await _apply_compression_label(
            label=label,
            tool=tool,
            result=result,
            content_len=content_len,
            line_summary=classify.line_summaries.get(p.message_id, ""),
            reasoning=classify.reasonings.get(p.message_id, ""),
            backend=backend,
        )
        if compressed_summary is not None:
            compressions.append(Compression(
                message_id=p.message_id,
                compressed_summary=compressed_summary,
                compression_label=label,
            ))

    elapsed_total = time.perf_counter() - t_total
    logger.info("compress_messages done: %d compressions in %.1fs", len(compressions), elapsed_total)
    return CompressionResult(
        compressions=compressions, 
        new_summary=new_summary or conversation_summary or "",
    )


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------

DIGEST_TOKEN_BUDGET = 6000
_DIGEST_PIECE_MAX_CHARS = 1600    # ~400 tokens cap per piece before budget truncation
_THINKING_EXCERPT_MAX_CHARS = 300
_ASSISTANT_RESPONSE_MAX_CHARS = 800
WORKING_MEMORY_ITERATION_THRESHOLD = 10


@dataclass
class MessageSnapshot:
    """Lightweight snapshot of a DB message passed to build_digest — no DB dependency in compress.py."""
    role: str
    content: str | None
    thinking: str | None
    compressed_summary: str | None


@dataclass
class DigestEntry:
    """Slim representation of one message for the working memory writer."""
    role: str
    content: str


@dataclass
class WorkingMemoryResult:
    """Output of write_working_memory."""
    working_memory_json: dict
    rendered: str


def build_digest(snapshots: list[MessageSnapshot]) -> list[DigestEntry]:
    """Build a token-bounded digest from message snapshots for the working memory writer.

    Tool messages use compressed_summary when available. Assistant messages get a thinking excerpt
    and a response excerpt. Budget is enforced via proportional truncation (4 chars ≈ 1 token).
    """
    entries: list[DigestEntry] = []

    for snapshot in snapshots:
        if snapshot.role == "user":
            content = (snapshot.content or "")[:_DIGEST_PIECE_MAX_CHARS]
            if content != "":
                entries.append(DigestEntry(role="user", content=content))

        elif snapshot.role == "assistant":
            thinking_excerpt = (snapshot.thinking or "")[:_THINKING_EXCERPT_MAX_CHARS]
            response = snapshot.content or ""
            # Messages whose content is only a <think> wrapper drive tool calls — not useful in the digest
            if response.startswith("<think>"):
                response = ""
            response = response[:_ASSISTANT_RESPONSE_MAX_CHARS]
            parts: list[str] = []
            if thinking_excerpt != "":
                parts.append(f"[thinking] {thinking_excerpt}")
            if response != "":
                parts.append(response)
            if parts:
                entries.append(DigestEntry(role="assistant", content="\n".join(parts)))

        elif snapshot.role == "tool":
            if snapshot.compressed_summary is not None and snapshot.compressed_summary != "":
                content = snapshot.compressed_summary[:_DIGEST_PIECE_MAX_CHARS]
            else:
                try:
                    result = json.loads(snapshot.content or "{}")
                except (json.JSONDecodeError, ValueError):
                    result = {}
                content = _compact_summary(result)
            if content != "":
                entries.append(DigestEntry(role="tool", content=content))

    # Budget enforcement via proportional scaling when over-budget
    budget_chars = DIGEST_TOKEN_BUDGET * 4
    total_chars = sum(len(e.content) for e in entries)
    if total_chars > budget_chars:
        scale = budget_chars / total_chars
        for entry in entries:
            entry.content = entry.content[:max(1, int(len(entry.content) * scale))]

    return entries


_WORKING_MEMORY_WRITER_SYSTEM = """\
You are writing a structured working memory summary for an AI agent session.
Your output replaces a long sequence of messages, preserving only what matters for continuing the work.

Output a JSON object with these optional keys (include only those that apply):
  goal        — what the user is currently trying to accomplish (string)
  learned     — key facts discovered: file locations, architecture, patterns (list of strings)
  done        — completed actions: files edited, tasks finished (list of strings)
  rejected    — approaches tried and abandoned, with reason (list of strings)
  dead_ends   — paths explored that were not useful (list of strings)

Rules:
- Be concise and factual. No prose.
- Preserve specific file paths, function names, line numbers.
- If a previous working memory is provided, incorporate its information and update it.
- Output valid JSON only, no markdown fences or surrounding text.\
"""


def _render_working_memory(working_memory_json: dict) -> str:
    """Render working memory JSON to the markdown string injected into the LLM context."""
    lines = ["[Context: agent working memory]", ""]

    if "goal" in working_memory_json:
        lines.append(f"**Goal:** {working_memory_json['goal']}")

    for key, label in [("learned", "Learned"), ("done", "Done"), ("rejected", "Rejected"), ("dead_ends", "Dead ends")]:
        value = working_memory_json.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            lines.append(f"**{label}:**")
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(f"**{label}:** {value}")

    return "\n".join(lines)


async def write_working_memory(
    previous_working_memory_json: dict | None,
    digest: list[DigestEntry],
    backend: LLMBackend,
) -> WorkingMemoryResult:
    """Write or update the working memory from a digest of recent agent activity.

    previous_working_memory_json folds prior context forward; pass None for the first run.
    Returns new structured JSON and its rendered markdown form.
    """
    parts: list[str] = []

    if previous_working_memory_json is not None:
        parts.append(
            "Previous working memory:\n"
            + json.dumps(previous_working_memory_json, ensure_ascii=False, indent=2)
        )

    parts.append("Recent activity (in order):")
    for entry in digest:
        parts.append(f"[{entry.role}] {entry.content}")

    prompt = "\n\n".join(parts)
    raw = await _llm_complete(prompt, backend, system=_WORKING_MEMORY_WRITER_SYSTEM)

    try:
        working_memory_json = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "write_working_memory: failed to parse JSON response — using empty dict. raw=%r",
            raw[:300],
        )
        working_memory_json = {}

    rendered = _render_working_memory(working_memory_json)
    logger.info(
        "write_working_memory: %d chars rendered, sections=%s",
        len(rendered),
        list(working_memory_json.keys()),
    )
    return WorkingMemoryResult(working_memory_json=working_memory_json, rendered=rendered)


# ---------------------------------------------------------------------------
# Compression-aware inference context assembly
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def load_system_prompt(prompt_id: str, sess: AsyncSession) -> LLMMessage | None:
    """Load a system prompt by slug, prepend today's date, return as a system LLMMessage.

    Returns None if the prompt file does not exist.
    """
    import yaml as _yaml
    prompt_path = _PROMPTS_DIR / f"{prompt_id}.yaml"
    if not prompt_path.exists():
        return None
    data = _yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    content = f"Today's date: {today}\n\n{data.get('content') or ''}"
    return {"role": "system", "content": content}


async def fetch_images_by_message(branch: list[db.Message], sess: AsyncSession) -> dict[str, list]:
    """Fetch image attachments for all messages in the branch, grouped by message_id."""
    branch_ids = [m.id for m in branch]
    if not branch_ids:
        return {}
    img_rows = (await sess.execute(
        select(db.MessageImageAttachment, db.Image)
        .join(db.Image, db.MessageImageAttachment.image_id == db.Image.id)
        .where(db.MessageImageAttachment.message_id.in_(branch_ids))
        .order_by(db.MessageImageAttachment.position)
    )).all()
    images_by_msg: dict[str, list] = {}
    for att, img in img_rows:
        images_by_msg.setdefault(att.message_id, []).append(img)
    return images_by_msg


def assemble_message(
    m: db.Message,
    images_by_msg: dict[str, list],
    interrupted_id: str | None,
) -> LLMMessage:
    """Assemble one non-excluded, non-summary DB message into an LLMMessage.

    Handles thinking prepend for interrupted tool-calling turns, tool_calls reconstruction,
    and multipart image content.
    """
    content = m.content or ""
    if m.id == interrupted_id:
        if m.thinking is not None and not content.startswith("<think>"):
            content = f"<think>{m.thinking}</think>{content}"
        try:
            stored_calls = json.loads(m.tool_calls)
            tool_calls_for_context = [
                {
                    "id": tc.get("id", f"tc-{i}"),
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc.get("args", {})},
                }
                for i, tc in enumerate(stored_calls)
            ]
        except (json.JSONDecodeError, ValueError, KeyError):
            tool_calls_for_context = []
    else:
        tool_calls_for_context = []

    msg: LLMMessage = {"role": m.role, "content": content}
    if len(tool_calls_for_context) > 0:
        msg["tool_calls"] = tool_calls_for_context
    imgs = images_by_msg.get(m.id, [])
    if imgs:
        multipart_content: list[dict] = [{"type": "text", "text": content}]
        for img in imgs:
            multipart_content.append({"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"}})
        msg["content"] = multipart_content
    return msg


async def build_inference_context(
    branch: list[db.Message],
    prompt_id: str | None,
    sess: AsyncSession,
) -> list[LLMMessage]:
    """Build the LLM message list for a branch, applying the compression view.

    Excluded messages with a compressed_summary are injected as compact stubs.
    Excluded messages without a summary are dropped entirely.
    context_summary messages (working memory) are injected as user-role messages.
    """
    messages: list[LLMMessage] = []

    if prompt_id is not None:
        system_msg = await load_system_prompt(prompt_id, sess)
        if system_msg is not None:
            messages.append(system_msg)

    images_by_msg = await fetch_images_by_message(branch, sess)

    non_excluded = [m for m in branch if not m.context_excluded]
    last_assistant = next((m for m in reversed(non_excluded) if m.role == "assistant"), None)
    interrupted_id = last_assistant.id if (last_assistant is not None and last_assistant.tool_calls is not None) else None

    for m in branch:
        if m.context_excluded:
            if m.exclusion_reason in ("working_memory", "working_memory_superseded"):
                continue
            if m.compressed_summary is not None:
                try:
                    original: ToolResult = json.loads(m.content)
                except (json.JSONDecodeError, ValueError, TypeError):
                    original = {"tool": "tool", "status": "unknown"}
                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "tool": original.get("tool", "tool"),
                        "status": "compressed",
                        "summary": m.compressed_summary,
                        "tool_call_id": original.get("tool_call_id", ""),
                    })
                })
            continue
        if m.content is None or m.content.strip() == "":
            continue
        if m.role == "context_summary":
            messages.append({"role": "user", "content": m.content})
            continue
        messages.append(assemble_message(m, images_by_msg, interrupted_id))
    return messages


# ---------------------------------------------------------------------------
# DB-level compression application (mid-run patch and working memory insertion)
# ---------------------------------------------------------------------------

async def apply_db_compressions(
    sess: AsyncSession, messages: list[LLMMessage], conv_id: str
) -> list[LLMMessage]:
    """Patch in-memory tool messages with their compressed summaries after mid-run compression.

    Matches by tool_call_id so assistant messages (not persisted to DB) are preserved.
    """
    compressed_rows = (await sess.execute(
        select(db.Message)
        .where(db.Message.conversation_id == conv_id)
        .where(db.Message.context_excluded == True)
        .where(db.Message.compressed_summary.isnot(None))
    )).scalars().all()

    call_id_to_summary: dict[str, str] = {}
    for m in compressed_rows:
        try:
            content: ToolResult = json.loads(m.content)
            call_id = content.get("tool_call_id")
            if call_id is not None and m.compressed_summary is not None:
                call_id_to_summary[call_id] = m.compressed_summary
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("apply_db_compressions: failed to parse message content, skipping")

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        raw_content = msg.get("content")
        if not isinstance(raw_content, str):
            continue
        try:
            content_dict: ToolResult = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("apply_db_compressions: failed to parse in-memory message content, skipping")
            continue
        call_id = content_dict.get("tool_call_id")
        if call_id is not None and call_id in call_id_to_summary:
            msg["content"] = json.dumps({
                "tool": content_dict.get("tool", "tool"),
                "status": "compressed",
                "summary": call_id_to_summary[call_id],
                "tool_call_id": call_id,
            })

    return messages


async def apply_working_memory(
    conv: db.Conversation,
    branch: list[db.Message],
    conversation_id: str,
    sess: AsyncSession,
    backend: LLMBackend,
) -> None:
    """Synthesize or update the working memory message for a conversation.

    Covers all messages before the last user message with a structured summary.
    Inserts the working memory message into the tree between the last covered message
    and the first live message (the last user message). Marks covered messages excluded.
    """

    last_user_index: int | None = None
    for i, m in enumerate(branch):
        if m.role == "user":
            last_user_index = i

    if last_user_index is None or last_user_index == 0:
        logger.debug("apply_working_memory: nothing to cover (no preceding messages)")
        return

    first_live = branch[last_user_index]

    previous_working_memory: db.Message | None = None
    digest_start_index = 0
    for i, m in enumerate(branch):
        if m.role == "context_summary" and not m.context_excluded:
            previous_working_memory = m
            digest_start_index = i + 1
            break

    messages_to_cover = branch[digest_start_index:last_user_index]
    if not messages_to_cover:
        logger.debug("apply_working_memory: no new messages to cover since last working memory")
        return

    snapshots = [
        MessageSnapshot(
            role=m.role,
            content=m.content,
            thinking=m.thinking,
            compressed_summary=m.compressed_summary,
        )
        for m in messages_to_cover
    ]
    digest = build_digest(snapshots)

    previous_json: dict | None = None
    if previous_working_memory is not None:
        try:
            previous_json = json.loads(previous_working_memory.working_memory_json or "{}")
        except (json.JSONDecodeError, ValueError):
            previous_json = None

    working_memory_result = await write_working_memory(previous_json, digest, backend)

    last_covered = branch[last_user_index - 1]
    new_working_memory_message = db.Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        parent_id=last_covered.id,
        role="context_summary",
        content=working_memory_result.rendered,
        working_memory_json=json.dumps(working_memory_result.working_memory_json, ensure_ascii=False),
        created_at=_now(),
        context_excluded=False,
        is_degenerate=False,
    )
    sess.add(new_working_memory_message)

    first_live.parent_id = new_working_memory_message.id

    for m in messages_to_cover:
        m.context_excluded = True
        if m.exclusion_reason is None or m.exclusion_reason == "":
            m.exclusion_reason = "working_memory"

    if previous_working_memory is not None:
        previous_working_memory.context_excluded = True
        previous_working_memory.exclusion_reason = "working_memory_superseded"

    await sess.flush()
    logger.info(
        "apply_working_memory: covered %d messages, working memory id=%s",
        len(messages_to_cover),
        new_working_memory_message.id,
    )


# ---------------------------------------------------------------------------
# Top-level compression entry point
# ---------------------------------------------------------------------------

@dataclass
class RunCompressionResult:
    """Result of run_compression: per-message compressions, updated summary, and new context token count."""
    compressions: list[Compression]
    new_summary: str
    ctx_tokens: int


async def run_compression(
    conv: db.Conversation,
    branch: list[db.Message],
    conversation_id: str,
    sess: AsyncSession,
    backend: LLMBackend,
    protect_last: bool = False,
    is_mid_run: bool = False,
) -> RunCompressionResult:
    """Run the full compression pipeline on a conversation branch.

    Stage 1/2: classify and summarize tool results, writing compressed_summary to DB rows.
    Stage 3: synthesize a working memory message covering all messages before the last user message.
    Always runs Stage 3 regardless of whether there are tool candidates.
    """
    compressions: list[Compression] = []
    new_summary: str = ""

    candidates = [m for m in branch if m.role == "tool" and not m.context_excluded]

    if candidates:
        user_messages_goal = [m.content for m in reversed(branch) if m.role == "user"][:3]
        user_message = "\n---\n".join(reversed(user_messages_goal)) if user_messages_goal else ""

        all_dicts: list[TrackedMessage] = [{"id": m.id, "role": m.role, "content": m.content, "thinking": m.thinking} for m in branch]
        candidate_dicts: list[TrackedMessage] = [{"id": m.id, "role": m.role, "content": m.content, "thinking": m.thinking} for m in candidates]

        compression_result = await compress_messages(
            candidate_dicts,
            all_dicts,
            user_message,
            conversation_summary=None,
            backend=backend,
            protect_last=protect_last,
            is_mid_run=is_mid_run,
        )

        for c in compression_result.compressions:
            msg = next((m for m in candidates if m.id == c.message_id), None)
            if msg is not None:
                msg.context_excluded = True
                msg.exclusion_reason = "compressed"
                msg.compressed_summary = c.compressed_summary
                msg.compression_label = c.compression_label
                try:
                    original: ToolResult = json.loads(msg.content)
                except (json.JSONDecodeError, ValueError, TypeError):
                    original = {"tool": "tool", "status": "unknown"}
                compressed_content = json.dumps({
                    "tool": original.get("tool", "tool"),
                    "status": "compressed",
                    "summary": c.compressed_summary,
                    "tool_call_id": original.get("tool_call_id", ""),
                })
                msg.compressed_token_count = await backend.count_text_tokens(compressed_content)

        compressions = compression_result.compressions
        new_summary = compression_result.new_summary
        await sess.flush()

    try:
        await apply_working_memory(conv, branch, conversation_id, sess, backend)
    except Exception:
        logger.exception("Working memory synthesis failed — skipping")

    final_all_msgs = list((await sess.execute(
        select(db.Message).where(db.Message.conversation_id == conversation_id)
    )).scalars().all())
    final_branch = _build_active_branch_path(final_all_msgs, conv.active_message_id)
    settings = _parse_conv_settings(conv)
    inference_messages = await build_inference_context(final_branch, settings.active_prompt_id, sess)
    tools_list = get_ollama_tool_list([tool.name for tool in TOOL_REGISTRY.values()])
    ctx_tokens = await backend.count_tokens(backend.prepare_messages(inference_messages), tools_list)

    return RunCompressionResult(compressions=compressions, new_summary=new_summary, ctx_tokens=ctx_tokens)
