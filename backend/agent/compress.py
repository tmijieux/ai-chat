import json
import logging
import time

from llm.base import LLMBackend

logger = logging.getLogger(__name__)

COMPRESS_THRESHOLD_CHARS = 2000  # ~500 tokens

_SKIP_CLASSIFY = {"write_file", "edit_file"}


def _following_thinking(all_messages: list[dict], tool_id: str) -> str:
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


def _key_args(result: dict) -> str:
    tool = result.get("tool", "")
    if tool in ("read_file", "list_directory"):
        return repr(result.get("path", ""))
    if tool == "glob_files":
        return repr(result.get("pattern", ""))
    if tool == "grep_files":
        return repr(result.get("pattern", ""))
    if tool == "run_shell":
        return repr((result.get("command") or "")[:80])
    return ""


def _result_metadata(result: dict) -> dict:
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
        output = result.get("output") or result.get("stdout") or ""
        meta["line_count"] = len(output.splitlines())
    return meta


def _compact_summary(result: dict) -> str:
    tool = result.get("tool", "unknown")
    key = _key_args(result)
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


_CLASSIFY_SYSTEM = """\
You are a context compression subagent. Another AI agent has just completed a run and you must decide which tool results can be discarded to keep its context small.

For each tool call, answer: is this result ABSOLUTELY REQUIRED for the agent to continue working toward the user goal — or was it just an intermediate computation step the agent had to take to reach the next subgoal?

Label "keep" only if the result content would be directly referenced or needed in future reasoning.
Label "compress" if the result was an intermediate step whose purpose is already consumed (navigation, exploration, failed attempts, location searches, directory listings used to decide where to look next).

When in doubt, prefer "compress". Context economy is the primary goal.

Primary signal: following_thinking. If the agent's next thought directly uses or cites the result content → keep. If the agent just moved on to the next step → compress.

Also write a one-sentence summary of what the agent accomplished overall.

Respond with JSON only, no prose, no markdown fences.

Example input:
User goal: "add a new route to the app config"
Tool calls:
[
  {"index": 0, "tool": "glob_files", "args_summary": "'**/*.config.ts'", "result_metadata": {"status": "ok", "file_count": 3}, "following_thinking": "Found 3 files. Let me list src/ to find the right one."},
  {"index": 1, "tool": "read_file", "args_summary": "'src/app/app.config.ts'", "result_metadata": {"status": "ok", "line_count": 42}, "following_thinking": "This file defines the providers. provideRouter is called with the routes array — I need to add the new route here."}
]

Example output:
{"labels": {"0": "compress", "1": "keep"}, "summary": "Agent located and read the app config file to add a new route."}\
"""


async def _classify_and_summarize(
    pairs: list[dict],
    user_message: str,
    conversation_summary: str | None,
    backend: LLMBackend,
) -> tuple[dict[str, str], str]:
    conv_line = f"Conversation so far: {conversation_summary}\n" if conversation_summary else ""
    llm_pairs = [
        {k: v for k, v in p.items() if k not in ("message_id", "tool_call_id")}
        for p in pairs
    ]

    prompt = f"""\
User's goal: {user_message}
{conv_line}
Tool calls:
{json.dumps(llm_pairs, ensure_ascii=False, indent=2)}\
"""

    t0 = time.perf_counter()
    raw = await _llm_complete(prompt, backend, system=_CLASSIFY_SYSTEM)
    elapsed = time.perf_counter() - t0
    logger.info("classify LLM call: %.1fs, raw length=%d", elapsed, len(raw))
    logger.debug("classify raw response: %s", raw[:500])

    json_str = _extract_json(raw)
    try:
        parsed = json.loads(json_str)
        labels = {str(k): str(v) for k, v in parsed.get("labels", {}).items()}
        summary = str(parsed.get("summary", ""))
        logger.info("classify labels: %s | summary: %s", labels, summary[:80])
        return labels, summary
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "compress classify parse error: %s — json_str: %r — raw: %r",
            e,
            json_str[:200],
            raw[:300],
        )
        return {str(i): "useful" for i in range(len(pairs))}, ""


_SUMMARIZE_FILE_SYSTEM = """\
Summarize the given file for use as context in a coding agent. Output:
1. One-sentence module purpose
2. All public functions/classes: full signature with type hints + one-line description
3. Key constants and config values
4. External imports

Be terse. No examples. No prose beyond descriptions.\
"""


async def _summarize_file(result: dict, backend: LLMBackend) -> str:
    path = result.get("path", "unknown")
    content = result.get("file_content") or ""
    line_count = len(content.splitlines())

    prompt = f"File: {path}\n---\n{content}"

    t0 = time.perf_counter()
    summary = await _llm_complete(prompt, backend, system=_SUMMARIZE_FILE_SYSTEM)
    elapsed = time.perf_counter() - t0
    est_tokens = len(summary) // 4
    logger.info(
        "summarize_file %s: %.1fs → %d chars (~%d tokens)",
        path,
        elapsed,
        len(summary),
        est_tokens,
    )

    return f"[compressed: read_file({repr(path)}) → {line_count} lines → ~{est_tokens} tokens — prefer grep_files to extract snippets, read_file as last resort]\n{summary}"


async def compress_messages(
    candidate_messages: list[dict],
    all_messages: list[dict],
    user_message: str,
    conversation_summary: str | None,
    backend: LLMBackend,
) -> tuple[list[dict[str, str]], str]:
    """
    Classify and compress a list of tool result messages.

    candidate_messages: tool-role messages to consider for compression.
      Each dict must have: id, role, content.
    all_messages: full ordered branch (for following_thinking lookup).
      Each dict must have: role, content, thinking.

    Returns:
      compressions: list of {message_id, compressed_summary}
      new_summary:  updated one-sentence conversation summary
    """
    t_total = time.perf_counter()

    pairs = []
    for i, msg in enumerate(candidate_messages):
        if msg.get("role") != "tool":
            continue
        try:
            result = json.loads(msg.get("content") or "{}")
        except (json.JSONDecodeError, ValueError):
            continue
        tool = result.get("tool", "")
        if tool in _SKIP_CLASSIFY:
            logger.debug("skipping classify for %s (tool=%s)", msg.get("id"), tool)
            continue
        tool_call_id = result.get("tool_call_id", "")
        pairs.append({
            "index": i,
            "message_id": msg["id"],
            "tool_call_id": tool_call_id,
            "tool": tool,
            "args_summary": _key_args(result),
            "result_metadata": _result_metadata(result),
            "following_thinking": _following_thinking(all_messages, tool_call_id),
        })

    logger.info(
        "compress_messages: %d candidate messages → %d classifiable pairs",
        len(candidate_messages),
        len(pairs),
    )

    if not pairs:
        return [], conversation_summary or ""

    labels, new_summary = await _classify_and_summarize(
        pairs, user_message, conversation_summary, backend
    )

    compressions: list[dict[str, str]] = []

    for p in pairs:
        label = labels.get(str(p["index"]), "useful")
        msg = candidate_messages[p["index"]]
        try:
            result = json.loads(msg.get("content") or "{}")
        except (json.JSONDecodeError, ValueError):
            continue

        compressed_summary: str | None = None

        logger.debug(
            "pair index=%d tool=%s label=%s content_len=%d",
            p["index"],
            p["tool"],
            label,
            len(msg.get("content") or ""),
        )

        if label == "compress":
            compressed_summary = _compact_summary(result)
            logger.info("compress → compact: %s", compressed_summary)
        elif label == "keep" and result.get("tool") == "read_file":
            file_content = result.get("file_content") or ""
            if len(file_content) > COMPRESS_THRESHOLD_CHARS:
                logger.info(
                    "useful read_file over threshold (%d chars) → summarizing",
                    len(file_content),
                )
                compressed_summary = await _summarize_file(result, backend)

        if compressed_summary is not None:
            compressions.append({
                "message_id": p["message_id"],
                "compressed_summary": compressed_summary,
            })

    elapsed_total = time.perf_counter() - t_total
    logger.info(
        "compress_messages done: %d compressions in %.1fs",
        len(compressions),
        elapsed_total,
    )
    return compressions, new_summary or conversation_summary or ""
