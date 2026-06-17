import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llm.base import LLMBackend
from message_types import LLMMessage, TrackedMessage
from tool_result_types import ToolResult

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
            if m.get("role") == "tool":
                try:
                    c = json.loads(m.get("content") or "{}")
                    if c.get("tool_call_id") == tool_id:
                        found = True
                except (json.JSONDecodeError, ValueError):
                    pass
            continue
        if m.get("role") == "assistant":
            text = m.get("thinking") or m.get("content") or ""
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
    meta: dict = {"status": result.get("status", "unknown")}
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
        output = result.get("output") or result.get("stderr") or ""
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
                    files read then immediately edited, failed/retried attempts.
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


def _compact_glob_result(result: dict) -> str:
    """Truncate a glob result to _GLOB_MAX_FILES paths, sorted shallowest-first."""
    files: list[str] = result.get("files") or []
    total = len(files)
    sorted_files = sorted(files, key=lambda p: (p.count("/") + p.count("\\"), p))
    kept = sorted_files[:_GLOB_MAX_FILES]
    lines = "\n".join(kept)
    suffix = f"\n… {total - _GLOB_MAX_FILES} more files not shown" if total > _GLOB_MAX_FILES else ""
    pattern = result.get("pattern", "")
    return f"[truncated: glob_files({repr(pattern)}) → {total} files]\n{lines}{suffix}"


def _compact_grep_result(result: dict) -> str:
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


async def _summarize_shell_output(result: dict, backend: LLMBackend) -> str:
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


async def _summarize_search_results(result: dict, backend: LLMBackend) -> str:
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


async def _summarize_file(result: dict, backend: LLMBackend) -> str:
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
    pairs = []
    for i, msg in enumerate(candidate_messages):
        if msg.get("role") != "tool":
            continue
        try:
            result = json.loads(msg.get("content") or "{}")
        except (json.JSONDecodeError, ValueError):
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

    if not pairs:
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
    return CompressionResult(compressions=compressions, new_summary=new_summary or conversation_summary or "")
