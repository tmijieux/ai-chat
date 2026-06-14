import json
import asyncio
import logging
import os
import re
import uuid
import aiofiles
import aiohttp
from typing import Any, Callable, Awaitable

from .tools import TOOL_REGISTRY, get_ollama_tool_list
from .auto_safety import evaluate_tool_safety, _ALWAYS_SAFE_TOOLS, _FILE_WRITE_TOOLS, is_path_inside_workspace
from .compress import _summarize_shell_output, _summarize_search_results
from llm import backend
from llm.base import ToolCallStartEvent, ToolCallArgEvent

CTX_LIMIT = 2**15

logger = logging.getLogger(__name__)

_LARGE_OUTPUT_CHARS = 20_000  # preshrink threshold for run_shell / search_web outputs


async def _save_temp_output(content: str, prefix: str, working_directory: str) -> str:
    """Write content to .agent_tmp/ inside the workspace and return the relative path."""
    tmp_dir = os.path.join(working_directory, ".agent_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.txt"
    file_path = os.path.join(tmp_dir, filename)
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(content)
    return os.path.join(".agent_tmp", filename)


async def _maybe_preshrink_tool_output(
    tool_name: str,
    result_dict: dict,
    working_directory: str | None,
) -> dict:
    """If a tool output exceeds _LARGE_OUTPUT_CHARS, save raw to a temp file and replace with a summary."""
    if working_directory is None:
        return result_dict

    if tool_name == "run_shell":
        raw_output = result_dict.get("output")
        if raw_output is None or raw_output == "":
            error_dict = result_dict.get("error")
            if error_dict is None:
                error_dict = {}
            raw_output = error_dict.get("message")
            if raw_output is None:
                raw_output = ""
        if len(raw_output) <= _LARGE_OUTPUT_CHARS:
            return result_dict
        temp_path = await _save_temp_output(raw_output, "shell", working_directory)
        summary = await _summarize_shell_output(result_dict, backend)
        modified = dict(result_dict)
        modified["output"] = (
            f"[Output too large ({len(raw_output):,} chars) — auto-summarized. "
            f"Full output saved to {temp_path}; use grep_files or read_file_range on it for details.]\n\n"
            + summary
        )
        logger.info(
            "preshrink run_shell: %d chars → saved to %s, summary %d chars",
            len(raw_output), temp_path, len(summary),
        )
        return modified

    if tool_name == "search_web":
        results_list = result_dict.get("results")
        if results_list is None:
            results_list = []
        total_chars = sum(len(r.get("content") or "") for r in results_list)
        if total_chars <= _LARGE_OUTPUT_CHARS:
            return result_dict
        parts = [f"Query: {result_dict.get('query', '')}", ""]
        for i, r in enumerate(results_list):
            parts.append(f"=== Result {i + 1}: {r.get('url', '')} ===")
            parts.append(r.get("content") or "")
            parts.append("")
        temp_content = "\n".join(parts)
        temp_path = await _save_temp_output(temp_content, "search", working_directory)
        summary = await _summarize_search_results(result_dict, backend)
        modified = dict(result_dict)
        modified["results"] = [{
            "url": "[auto-summarized]",
            "content": (
                f"[Results too large ({total_chars:,} chars total) — auto-summarized. "
                f"Full results saved to {temp_path}; use grep_files or read_file_range on it.]\n\n"
                + summary
            ),
        }]
        logger.info(
            "preshrink search_web: %d total chars → saved to %s, summary %d chars",
            total_chars, temp_path, len(summary),
        )
        return modified

    return result_dict


class AgentSession:
    """Manages bidirectional communication between the agent loop and the WebSocket client."""

    def __init__(self):
        self.outbound: asyncio.Queue[dict] = asyncio.Queue()
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._pending_plan_confirms: dict[str, asyncio.Future] = {}
        self._pending_user_inputs: dict[str, asyncio.Future] = {}
        self._compression_event: asyncio.Event = asyncio.Event()
        self._compression_conv_id: str | None = None
        self.refresh_messages_callback: Callable[[str], Awaitable[list[dict]]] | None = None
        self.finish_result: dict | None = None
        self._search_result_ids: set[str] = set()
        self._sub_stage_counters: dict[str, int] = {}
        self.mode: str = "standard"
        self.working_directory: str | None = None
        self.last_user_message: str | None = None

    async def emit(self, event: dict) -> None:
        await self.outbound.put(event)

    async def request_confirm(
        self, tool_id: str, tool_name: str, arguments: dict, preview: str,
        diff_lines: list | None = None,
    ) -> tuple[bool, str | None]:
        """Emit a confirmation request and suspend until the client responds.

        In auto/yolo mode applies rule-based + LLM safety evaluation instead of
        prompting the user for every tool call.
        """
        if self.mode in ("auto", "yolo"):
            # Rule-based: always safe
            if tool_name in _ALWAYS_SAFE_TOOLS:
                return True, None
            # Rule-based: in-workspace file write
            if tool_name in _FILE_WRITE_TOOLS:
                path = arguments.get("file_path", "")
                if self.working_directory is not None and is_path_inside_workspace(path, self.working_directory):
                    return True, None
            # LLM evaluation for run_shell, search_web, out-of-workspace writes
            if self.mode == "auto":
                await self.emit({"type": "tool_evaluating", "tool_id": tool_id, "tool_name": tool_name})
            verdict, reason = await evaluate_tool_safety(
                tool_name, arguments, self.working_directory,
                self.last_user_message or "", backend,
            )
            if verdict == "safe":
                if self.mode == "auto":
                    await self.emit({"type": "tool_auto_approved", "tool_id": tool_id})
                return True, None
            # Dangerous: auto shows confirmation UI; yolo rejects and lets LLM handle it
            if self.mode == "yolo":
                return False, f"Safety evaluator blocked this action: {reason}"

        event: dict = {
            "type": "tool_confirm",
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "preview": preview,
        }
        if diff_lines is not None:
            event["diff_lines"] = diff_lines
        await self.emit(event)
        future: asyncio.Future[tuple[bool, str | None]] = asyncio.get_running_loop().create_future()
        self._pending_confirms[tool_id] = future
        return await future

    def resolve_confirm(self, tool_id: str, approved: bool, reason: str | None = None) -> None:
        future = self._pending_confirms.pop(tool_id, None)
        if future and not future.done():
            future.set_result((approved, reason))

    async def request_plan_confirm(self, plan_id: str, plan: str) -> dict:
        """Emit a plan_proposal event and suspend until the user responds with a payload dict."""
        await self.emit({"type": "plan_proposal", "plan_id": plan_id, "plan": plan})
        future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending_plan_confirms[plan_id] = future
        return await future

    def resolve_plan_confirm(self, plan_id: str, payload: dict) -> None:
        future = self._pending_plan_confirms.pop(plan_id, None)
        if future and not future.done():
            future.set_result(payload)

    async def request_user_input(self, question_id: str, question: str, options: list[str] | None = None) -> str:
        """Emit an agent_question event and suspend until the user replies."""
        event: dict = {"type": "agent_question", "question_id": question_id, "question": question}
        if options is not None:
            event["options"] = options
        await self.emit(event)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending_user_inputs[question_id] = future
        return await future

    def resolve_user_input(self, question_id: str, reply: str) -> None:
        future = self._pending_user_inputs.pop(question_id, None)
        if future and not future.done():
            future.set_result(reply)

    async def await_compression(self) -> str | None:
        """Suspend the agent loop until the frontend sends compression_done."""
        self._compression_event.clear()
        await self._compression_event.wait()
        return self._compression_conv_id

    def resume_after_compression(self, conv_id: str) -> None:
        self._compression_conv_id = conv_id
        self._compression_event.set()


_TOOL_CALL_BLOCK_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
_FUNCTION_RE = re.compile(r'<function=(\w+)>(.*?)</function>', re.DOTALL)
_PARAMETER_RE = re.compile(r'<parameter=(\w+)>\n?(.*?)\n?</parameter>', re.DOTALL)


def _parse_embedded_tool_call(block_content: str) -> dict | None:
    """Parse <function=NAME><parameter=K>V</parameter>…</function> into name + arguments dict."""
    func_match = _FUNCTION_RE.search(block_content)
    if func_match is None:
        return None
    name = func_match.group(1)
    func_body = func_match.group(2)
    arguments = {m.group(1): m.group(2) for m in _PARAMETER_RE.finditer(func_body)}
    return {"name": name, "arguments": arguments}


def _extract_tool_calls_from_thinking(thinking: str) -> list[dict]:
    """
    Recover tool calls that the model emitted inside the thinking block without closing </think>.
    Qwen3 uses <tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>.
    Returns tool call dicts in the same shape as the normal stream-assembled list,
    marked with _recovered=True so the caller can emit tool_call_start manually.
    """
    result = []
    for index, block_match in enumerate(_TOOL_CALL_BLOCK_RE.finditer(thinking)):
        parsed = _parse_embedded_tool_call(block_match.group(1))
        if parsed is None or parsed["name"] == "":
            continue
        result.append({
            "id": f"tc-recovered-{index}",
            "function": {"name": parsed["name"], "arguments": parsed["arguments"]},
            "_recovered": True,
        })
    return result


def _strip_tool_call_blocks(thinking: str) -> str:
    """Remove <tool_call>…</tool_call> blocks from thinking before storing it in LLM context."""
    return _TOOL_CALL_BLOCK_RE.sub("", thinking).strip()


def _find_superseded_read_file_indices(pairs: list[tuple[str, str]]) -> list[int]:
    """Given (role, content_json) pairs, return indices of superseded read_file results."""
    path_indices: dict[str, list[int]] = {}
    for i, (role, content_str) in enumerate(pairs):
        if role != "tool":
            continue
        try:
            content = json.loads(content_str or "")
        except (json.JSONDecodeError, ValueError):
            continue
        if content.get("tool") != "read_file" or content.get("status") != "success":
            continue
        path_indices.setdefault(content.get("path", ""), []).append(i)
    return [i for indices in path_indices.values() for i in indices[:-1]]


def _deduplicate_file_reads(messages: list[dict[str, Any]]) -> None:
    pairs = [(m.get("role", ""), m.get("content", "")) for m in messages]
    for i in _find_superseded_read_file_indices(pairs):
        try:
            path = json.loads(messages[i].get("content", "")).get("path", "")
        except (json.JSONDecodeError, ValueError):
            path = ""
        try:
            tool_call_id = json.loads(messages[i].get("content", "")).get("tool_call_id")
        except (json.JSONDecodeError, ValueError):
            tool_call_id = None
        messages[i]["content"] = json.dumps({
            "tool": "read_file",
            "status": "evicted",
            "path": path,
            "tool_call_id": tool_call_id,
            "reason": "file content removed — analysis was expressed in conversation above, superseded by later read",
        })




def _log_context(messages: list[dict[str, Any]]) -> None:
    print(f"\n=== CONTEXT ({len(messages)} messages) ===")
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if content is None:
            content = ""
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            img_count = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
            content = " ".join(text_parts) + (f" [+{img_count} image(s)]" if img_count else "")
        if role == "system":
            print(f"  [system] {content[:120].replace('\n', ' ')}")
        elif role == "user":
            print(f"  [user] {content[:200].replace('\n', ' ')}")
        elif role == "tool":
            try:
                j = json.loads(content)
                tool = j.get("tool", "?")
                status = j.get("status", "?")
                path = j.get("path", "")
                if tool == "read_file":
                    suffix = " [evicted]" if status == "evicted" else ""
                    print(f"  [tool] FILE {path}{suffix}")
                elif tool == "list_directory":
                    print(f"  [tool] DIRECTORY {path}")
                elif tool == "glob_files":
                    pattern = j.get("pattern", "")
                    print(f"  [tool] GLOB {pattern} in {path}")
                elif tool == "grep_files":
                    pattern = j.get("pattern", "")
                    glob_pat = j.get("glob_pattern", "")
                    suffix = f" [{glob_pat}]" if glob_pat else ""
                    print(f"  [tool] GREP '{pattern}' in {path}{suffix}")
                else:
                    print(f"  [tool] {tool}: {status}")
            except (json.JSONDecodeError, ValueError):
                print(f"  [tool] {content[:80].replace('\n', ' ')}")
        elif role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls is not None and len(tool_calls) > 0:
                print(f"  [thinking] {content.replace('\n', ' ')[:120]}")
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
                print(f"  [assistant] {len(tool_calls)} tool call(s): {names}")
            elif content.strip() != "":
                print(f"  [assistant] {content.replace('\n', ' ')[:120]}")
        else:
            print(f"  [{role}] {content[:80].replace('\n', ' ')}")
    print("=" * 40)


def _parse_tool_calls(tool_calls_acc: dict[int, dict], message: dict) -> list[dict]:
    """Build the tool_calls list from streamed fragments, with thinking-block recovery fallback."""
    tool_calls: list[dict] = []
    for acc in (tool_calls_acc[i] for i in sorted(tool_calls_acc)):
        try:
            arguments = json.loads(acc["arguments_str"]) if acc["arguments_str"] else {}
        except json.JSONDecodeError:
            logger.warning("Malformed tool call arguments for %s, skipping: %r", acc["name"], acc["arguments_str"])
            continue
        tool_calls.append({"id": acc["id"], "function": {"name": acc["name"], "arguments": arguments}})

    # Recover tool calls embedded in thinking when the model forgot to close </think> first.
    # Only attempt recovery when both regular content and stream-parsed tool calls are absent.
    if len(tool_calls) == 0 and len(message["content"]) == 0 and len(message["thinking"]) > 0:
        recovered = _extract_tool_calls_from_thinking(message["thinking"])
        if recovered:
            logger.warning("Recovering %d tool call(s) embedded in thinking block", len(recovered))
            tool_calls = recovered

    return tool_calls


async def _execute_tool_calls(
    tool_calls: list[dict],
    messages: list[dict[str, Any]],
    session: "AgentSession",
    tools: list[dict],
    working_directory: str | None,
    extra_tools: dict | None,
) -> bool:
    """Execute each tool call, append results to messages, and handle mid-run compression.

    Returns True when context overflows even after compression (caller should abort).
    Mutates messages in place.
    """
    ctx_before = await backend.count_tokens(backend.prepare_messages(messages), tools)
    print(f"[tokens] context before tool execution: {ctx_before}/{CTX_LIMIT}")
    await session.emit({"type": "ctx_update", "ctx_tokens": ctx_before})

    effective_registry = {**TOOL_REGISTRY, **(extra_tools if extra_tools is not None else {})}

    for tool_call in tool_calls:
        tool_name: str = tool_call.get("function", {}).get("name", "")
        tool_args: dict = tool_call.get("function", {}).get("arguments", {})
        call_id: str = tool_call.get("id", f"tc-{id(tool_call)}")

        # Recovered tool calls never got a tool_call_start during streaming — emit it now.
        if tool_call.get("_recovered"):
            await session.emit({"type": "tool_call_start", "tool_id": call_id, "tool_name": tool_name})

        await session.emit({"type": "tool_call", "tool_id": call_id, "tool_name": tool_name, "arguments": tool_args})

        if tool_name not in effective_registry:
            result_dict: dict = {"tool": tool_name, "status": "error", "error": {"message": f"Unknown tool: {tool_name}"}}
            log_msg = None
        else:
            tool_instance = effective_registry[tool_name]
            result_dict = await tool_instance.execute(tool_args, session, working_directory)
            result_dict = await _maybe_preshrink_tool_output(tool_name, result_dict, working_directory)
            log_msg = tool_instance.label(tool_args)

        result_dict["tool_call_id"] = call_id
        tool_output = json.dumps(result_dict)
        messages.append({"role": "tool", "name": tool_name, "content": tool_output})

        ctx_after = await backend.count_tokens(backend.prepare_messages(messages), tools)
        print(f"[tokens] context after tool result '{tool_name}': {ctx_after}/{CTX_LIMIT}")
        await session.emit({"type": "ctx_update", "ctx_tokens": ctx_after})

        if ctx_after > CTX_LIMIT:
            # Emit tool_result first so frontend saves it to DB before compressing.
            await session.emit({"type": "tool_result", "tool_id": call_id, "tool_name": tool_name, "content": tool_output, "log_message": log_msg})
            await session.emit({"type": "compressing", "ctx_tokens": ctx_after, "ctx_limit": CTX_LIMIT})
            conv_id = await session.await_compression()
            if conv_id is not None and session.refresh_messages_callback is not None:
                refreshed = await session.refresh_messages_callback(conv_id)
                messages[:] = refreshed
            ctx_after_compress = await backend.count_tokens(backend.prepare_messages(messages), tools)
            print(f"[tokens] context after compression: {ctx_after_compress}/{CTX_LIMIT}")
            await session.emit({"type": "ctx_update", "ctx_tokens": ctx_after_compress})
            if ctx_after_compress > CTX_LIMIT:
                await session.emit({"type": "error", "message": f"Context still exceeds limit after compression: {ctx_after_compress}/{CTX_LIMIT} tokens"})
                return True
            continue

        await session.emit({"type": "tool_result", "tool_id": call_id, "tool_name": tool_name, "content": tool_output, "log_message": log_msg})

    return False


async def chat_with_tools(
    messages: list[dict[str, Any]],
    session: AgentSession,
    tools: list[dict],
    working_directory: str | None,
    extra_tools: dict | None = None,
) -> tuple[bool,bool]:
    """
    One iteration of the LLM call + tool execution loop.
    Returns True when the agent is done (no tool calls were made).
    """
    prepared = backend.prepare_messages(messages)
    _log_context(prepared)

    ctx_before_generation = await backend.count_tokens(prepared, tools)
    max_tokens = CTX_LIMIT - ctx_before_generation
    print(f"[tokens] context before generation: {ctx_before_generation}/{CTX_LIMIT}, max_tokens={max_tokens}")
    await session.emit({"type": "ctx_update", "ctx_tokens": ctx_before_generation})

    message: dict[str, Any] = {"role": "assistant", "content": "", "thinking": ""}
    tool_calls_acc: dict[int, dict] = {}  # index → {id, name, arguments_str}
    prompt_eval_count: int = ctx_before_generation
    eval_count: int = 0
    done_reason: str = ""

    async for event in backend.stream_completion(prepared, tools, temperature=0.3, max_tokens=max_tokens):
        etype = event["type"]

        if etype == "thinking":
            message["thinking"] += event["content"]
            await session.emit({"type": "thinking", "content": event["content"]})

        elif etype == "content":
            message["content"] += event["content"]
            await session.emit({"type": "content", "content": event["content"]})

        elif etype == "tool_call_start":
            idx = event["index"]
            tool_calls_acc[idx] = {"id": event["id"], "name": event["name"], "arguments_str": ""}
            await session.emit({"type": "tool_call_start", "tool_id": event["id"], "tool_name": event["name"]})

        elif etype == "tool_call_arg":
            idx = event["index"]
            if idx in tool_calls_acc:
                tool_calls_acc[idx]["arguments_str"] += event["fragment"]
            await session.emit({"type": "tool_call_chunk", "tool_id": tool_calls_acc.get(idx, {}).get("id", ""), "chunk": event["fragment"]})

        elif etype == "done":
            prompt_eval_count = ctx_before_generation
            eval_count = event["completion_tokens"]
            done_reason = event["finish_reason"]
            print(f"[tokens] prompt_tokens={prompt_eval_count} completion_tokens={eval_count} finish_reason={done_reason}")

    if done_reason == "length":
        await session.emit({"type": "error", "message": f"Context limit reached during generation: {prompt_eval_count + eval_count}/{CTX_LIMIT} tokens. The response was cut off."})
        return True, False

    tool_calls = _parse_tool_calls(tool_calls_acc, message)

    finished_without_response = len(message["content"]) == 0 and len(tool_calls) == 0
    if finished_without_response:
        logger.warning("Agent finished without response: no content, no tool calls")

    if len(message["content"]) > 0:
        messages[:] = [m for m in messages if not m.get("_transient")]
        if len(tool_calls) > 0:
            message["tool_calls"] = tool_calls
        messages.append(message)
    elif len(message["thinking"]) > 0 or len(tool_calls) > 0:
        # Strip embedded <tool_call> blocks from thinking stored in context — the model already
        # sees them as tool_calls entries, keeping the raw XML would confuse it.
        thinking_for_context = _strip_tool_call_blocks(message["thinking"]) if tool_calls and tool_calls[0].get("_recovered") else message["thinking"]
        messages.append({
            "role": "assistant",
            "content": f"<think>{thinking_for_context}</think>",
            "tool_calls": tool_calls,
        })

    if len(tool_calls) > 0:
        overflow = await _execute_tool_calls(tool_calls, messages, session, tools, working_directory, extra_tools)
        if overflow:
            return True, False

    # Emit iteration_end after tool results so the frontend receives tool_result events
    # before iteration_end. The frontend rotation logic patches tool results from iteration N
    # with prompt_tokens from iteration N+1 — this ordering makes that work correctly.
    await session.emit({
        "type": "iteration_end",
        "prompt_tokens": prompt_eval_count,
        "response_tokens": eval_count,
    })

    _deduplicate_file_reads(messages)
    return len(tool_calls) == 0, finished_without_response


async def run_agent(
    session: AgentSession,
    messages: list[dict[str, Any]],
    tools: list[dict],
    working_directory: str | None,
    extra_tools: dict | None = None,
) -> None:
    """Run the full agent loop until done, emitting events via session."""
    try:
        finished = False
        finished_without_response = False
        while not finished:
            finished, finished_without_response = await chat_with_tools(messages, session, tools, working_directory, extra_tools)
        await session.emit({"type": "done", "finished_without_response": finished_without_response})
    except asyncio.CancelledError:
        await session.emit({"type": "error", "message": "Agent was aborted"})
    except aiohttp.ClientConnectorError as e:
        logger.error("LLM backend connection error: %s", e)
        await session.emit({"type": "error", "message": "LLM backend is not running"})
    except Exception as e:
        logger.exception("Unexpected error in agent loop")
        await session.emit({"type": "error", "message": str(e)})
