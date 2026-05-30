# ADR-0006: Post-Iteration Context Compression Pipeline

**Date:** 2026-05-30  
**Status:** Accepted

## Context

After a single agentic run asking "what does file X do?", context was at 24 882 / 32 768 tokens (75.9%) — before any follow-up question. Two independent sources of bloat were identified:

1. **Exploration noise** — intermediate tool calls the agent made while locating the target file (wrong globs, directory listings, grep with zero matches, successful-but-not-useful listings). These carry no informational value once the file is found, but each tool result stays in context verbatim.

2. **Large file reads** — the target file itself was 19 k tokens raw. A single `read_file` result dominates the context and makes multi-turn follow-up impossible.

Without compression, any agent run that reads a large file cannot continue past 1–2 iterations.

## Decision

Implement a two-stage **post-iteration compression pipeline** that runs in the framework layer after each main agent iteration, before the next one begins.

### Stage 1 — LLM-based usefulness classification + deterministic compaction

For each tool call in the iteration, a **stage 1 sub-agent** classifies it as `useful` or `noise`.

**Input to the classifier** (per tool call — never the full tool output):
```
user_message: "Dis-moi ce que fait le fichier data-push-ome-box."
conversation_summary: null  # first agent loop — omitted
tool_calls:
  - tool: glob_files
    args: {"pattern": "**/*ome-box*"}
    result_metadata: {status: "success", file_count: 0}
    following_thinking: "No match. The file name might be different. Let me browse..."
  - tool: list_directory
    args: {"path": "server"}
    result_metadata: {status: "success", entry_count: 47}
    following_thinking: "Let me go deeper into the data subfolder..."
  - tool: run_shell
    args: {"command": "npm test"}
    result_metadata: {status: "success", exit_code: 0, line_count: 47}
    following_thinking: "Tests passed. Moving on..."
  ...
```

`run_shell` is treated like any other tool — its `result_metadata` includes `exit_code` and `line_count` (never the full stdout). The classifier decides usefulness from the following thinking, same as for file tools.

On the first agent loop, `conversation_summary` is omitted. On subsequent loops, it is a one-sentence summary of what the conversation has accomplished so far (e.g. *"User asked about data_push_AUMBOX.py; agent summarised its sync logic and identified the main entry point."*). This mirrors the kind of compact handoff summary Claude produces during context compaction, and gives the classifier the intent context needed to judge whether a tool call contributed to the user's goal.

The same summary is also used to update the conversation title via `PUT /api/conversations/{id}` — reusing the same LLM call output for both purposes.

The classifier reads `(user_message, conversation_summary?, tool_name, key_args, result_metadata, following_thinking)` — the following thinking is the primary signal, but the user message and conversation summary anchor what "useful" means for this specific task. The full tool output content is never passed.

All tool pairs from the iteration are sent in a **single batch LLM call**, returning a `useful | noise` label per tool call. This keeps stage 1 cheap (a few dozen tokens of input per tool call, no file contents).

**After classification**, noise results are handled via the existing `excluded_from_context` mechanism:
- `message.excluded_from_context` is set to `true` — the inference context builder already skips these
- `message.compressed_summary` (new nullable column) stores the compact one-liner shown in the UI instead of the full content
- `message.content` is **never modified** — original content is preserved for potential future re-inclusion

Compact summary format:
```
glob_files("**/*ome-box*") → 0 files
list_directory("server") → 47 entries
grep_files("data-push-ome-box") → 0 matches
run_shell("npm test") → exit 0, 47 lines
glob_files("**/*push*") → 3 files
read_file("server/orbe/aumbox/data_push_AUMBOX.py") → 847 lines
```

Thinking messages are retained verbatim — they are the reasoning chain and already compact.

**Cost:** one batch LLM call per iteration (input size: ~50 tokens × number of tool calls).

### Stage 2 — Reference file summarization

For each tool result classified `useful` by stage 1 that is a `read_file` exceeding a token threshold (default: 500 tokens), a **stage 2 sub-agent** produces an API surface summary. Same `excluded_from_context` + `compressed_summary` mechanism: original `content` untouched, summary stored in `compressed_summary`, flag set.

The summary covers:
- Module purpose (1–2 sentences)
- Public functions/classes: signature + one-line description
- Key constants and configuration values
- External dependencies (imports)

Target output: 400–800 tokens regardless of input file size.

**Sub-agent prompt** (fixed, not user-configurable):
```
Summarize this file for use as context in a coding agent. Output:
1. One-sentence module purpose
2. All public functions/classes: signature + one-line description
3. Key constants and config values
4. External imports

Be terse. No examples. No prose beyond descriptions.
```

**Cost:** one LLM call per large useful file per iteration.

### What is never compressed

- `write_file`, `edit_file` results — already tiny (success/error message only); classifier will naturally label them useful but compaction is a no-op
- The most recent `read_file` for the **active file** (currently being edited) — kept in context verbatim
- Thinking messages — already compact, provide reasoning continuity
- Tool confirmation messages — UI state, not context

### DB changes

| Column | Type | Purpose |
|--------|------|---------|
| `messages.compressed_summary` | Text (nullable) | Compact one-liner or file API summary shown in UI when `excluded_from_context = true` due to compression |

`excluded_from_context` already exists. No other schema changes.

The inference context builder (`prepare_messages` / context assembly) already skips `excluded_from_context = true` messages — no changes needed there.

## Alternatives considered

**Deterministic noise detection (no LLM for stage 1)** — classify noise by keyword matching on following_thinking or result metadata alone. Rejected: regex rules are brittle and ambiguous. An LLM reading a few sentences of thinking handles edge cases correctly at low cost.

**LLM reads full tool output for stage 1** — pass the complete tool result to the classifier. Rejected: the following thinking already encodes the agent's own assessment; re-reading the full output adds tokens with no classification benefit.

**Modify `content` in place** — overwrite the tool result content with the compact summary. Rejected: destroys the original, making re-inclusion impossible. The `excluded_from_context` + `compressed_summary` split keeps original content intact.

**Evict noise entirely** — remove noise tool calls from the message list. Rejected: losing the record of what was tried causes the agent to re-attempt the same failed searches. The compact one-liner preserves the negative signal.

**Compress only at context pressure** — trigger only when context > threshold. Rejected: compressing proactively after every iteration keeps the math simple and predictable.

## Implementation sketch

```python
# backend/agent/compress.py

ALWAYS_USEFUL_TOOLS = {"write_file", "edit_file"}  # results already tiny, skip classifier overhead

async def compress_iteration(
    iteration_messages: list[dict],
    db_message_ids: list[str],          # parallel list — message ID for each iteration message
    user_message: str,
    conversation_summary: str | None,   # None on first loop
    backend: LLMBackend,
    db: AsyncSession,
) -> str:
    """Classify and compress noise tool results. Returns updated conversation summary."""
    pairs = []
    for i, msg in enumerate(iteration_messages):
        if msg["role"] != "tool":
            continue
        result = json.loads(msg["content"])
        tool = result.get("tool")
        if tool in ALWAYS_USEFUL_TOOLS:
            continue
        following_thinking = _following_thinking(iteration_messages, i)
        pairs.append({
            "index": i,
            "msg_id": db_message_ids[i],
            "tool": tool,
            "args_summary": _key_args(result),
            "result_metadata": _metadata(result),
            "following_thinking": following_thinking,
        })

    if not pairs:
        return conversation_summary or ""

    # Single batch call — classify all tool calls + generate updated conversation summary
    labels, new_summary = await _classify_and_summarize(
        pairs, user_message, conversation_summary, backend
    )

    for p in pairs:
        if labels.get(str(p["index"])) == "noise":
            result = json.loads(iteration_messages[p["index"]]["content"])
            summary = _compact_summary(result)
            await db.execute(
                update(Message)
                .where(Message.id == p["msg_id"])
                .values(excluded_from_context=True, compressed_summary=summary)
            )
        elif labels.get(str(p["index"])) == "useful":
            result = json.loads(iteration_messages[p["index"]]["content"])
            if result.get("tool") == "read_file" and _token_count(result) > COMPRESS_THRESHOLD:
                summary = await _summarize_file(result, backend)  # Stage 2
                await db.execute(
                    update(Message)
                    .where(Message.id == p["msg_id"])
                    .values(excluded_from_context=True, compressed_summary=summary)
                )

    await db.commit()
    return new_summary
```

`_following_thinking`: finds the next assistant thinking content after position `i` in the slice.  
`_compact_summary`: string formatter, no LLM — `tool_name(key_arg) → outcome`.  
`_classify_and_summarize`: single batch LLM call returning `{labels, summary}`.  
`_summarize_file`: one LLM call returning the API surface summary (stage 2).  
`_metadata`: extracts `{status, file_count, entry_count, exit_code, line_count}` etc. from result dict without full content.

## Consequences

- Agentic runs that read large files remain usable across multiple turns.
- Exploration record preserved (negative signal kept) at ~1 token per noise call.
- Original tool result content never modified — re-inclusion possible in the future.
- Each iteration costs: one cheap batch classification call + one LLM call per large useful file.
- Token counts stored in DB reflect post-compression context size from the next iteration onward.
- Active file tracking must be implemented to avoid compressing content the agent still needs verbatim.
- Conversation title is updated as a side effect of the summary generation — no extra LLM call.
