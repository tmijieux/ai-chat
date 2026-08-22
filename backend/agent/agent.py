import json
import asyncio
import logging
import os
import typing
import uuid
import aiofiles
import aiohttp
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Sequence, TypedDict, NotRequired, Literal, cast

from agent.tools.base import ToolDict
from conv_helpers import ToolSet, _find_superseded_read_file_indices

from .tools import TOOL_REGISTRY, get_ollama_tool_list
from .auto_safety import evaluate_tool_safety, _ALWAYS_SAFE_TOOLS, _FILE_WRITE_TOOLS, is_path_inside_workspace
from .compress import _summarize_shell_output, _summarize_search_results, WORKING_MEMORY_ITERATION_THRESHOLD
from llm import backend
from llm.base import ToolCallStartEvent, ToolCallArgEvent, TOOL_CALL_BLOCK_RE, parse_all_tool_calls
from message_types import LLMMessage, AssistantMessage, ToolCall, ToolCallFunction, PreparedMessages
from tool_result_types import ToolResult, RunShellResult, SearchWebResult, DiffLine

CTX_LIMIT = 2**15
CTX_COMPRESS_THRESHOLD = int(CTX_LIMIT * 0.55)  # ~18k tokens — compress early to leave headroom for compression LLM calls

logger = logging.getLogger(__name__)

_LARGE_OUTPUT_CHARS = 20_000  # preshrink threshold for run_shell / search_web outputs

# ANSI colors for the raw model-output stream printed to the console — kept visually distinct
# from the (separately colored) logging.* lines so the two don't blend together.
_ANSI_RESET = "\x1b[0m"
_ANSI_DIM = "\x1b[2m"       # thinking
_ANSI_GREEN = "\x1b[32m"    # content
_ANSI_MAGENTA = "\x1b[35m"  # tool calls
_ANSI_BLUE = "\x1b[34m"     # stream start/end markers


class ToolCallAccEntry(TypedDict):
    id: str
    name: str
    arguments_str: str


@dataclass
class GenerationResult:
    """Accumulated output from one LLM stream."""
    message: AssistantMessage
    tool_calls_acc: dict[int, ToolCallAccEntry]
    eval_count: int
    done_reason: str


@dataclass
class TurnResult:
    """Outcome of one chat_with_tools iteration."""
    is_done: bool
    finished_without_response: bool
    length_compressed: bool = False




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
    result_dict: ToolResult,
    working_directory: str | None,
) -> ToolResult:
    """If a tool output exceeds _LARGE_OUTPUT_CHARS, save raw to a temp file and replace with a summary."""
    if working_directory is None:
        return result_dict

    if tool_name == "run_shell":
        shell_result = cast(RunShellResult, result_dict)
        stdout = shell_result.get("output")
        if stdout is None:
            stdout = ""
        stderr = shell_result.get("stderr")
        if stderr is None:
            stderr = ""
        combined = stdout + stderr
        if len(combined) <= _LARGE_OUTPUT_CHARS:
            return shell_result
        temp_path = await _save_temp_output(combined, "shell", working_directory)
        summary = await _summarize_shell_output(shell_result, backend)
        modified_shell = cast(RunShellResult, dict(shell_result))
        modified_shell["output"] = (
            f"[Output too large ({len(combined):,} chars) — auto-summarized. "
            f"Full output saved to {temp_path}; use grep_files or read_file_range on it for details.]\n\n"
            + summary
        )
        modified_shell["stderr"] = ""
        logger.info(
            "preshrink run_shell: %d chars → saved to %s, summary %d chars",
            len(combined), temp_path, len(summary),
        )
        return modified_shell

    if tool_name == "search_web":
        search_result = cast(SearchWebResult, result_dict)
        results_list = search_result.get("results")
        if results_list is None:
            results_list = []
        total_chars = sum(len(r.get("content") or "") for r in results_list)
        if total_chars <= _LARGE_OUTPUT_CHARS:
            return search_result
        parts = [f"Query: {search_result.get('query', '')}", ""]
        for i, r in enumerate(results_list):
            parts.append(f"=== Result {i + 1}: {r.get('url', '')} ===")
            parts.append(r.get("content") or "")
            parts.append("")
        temp_content = "\n".join(parts)
        temp_path = await _save_temp_output(temp_content, "search", working_directory)
        summary = await _summarize_search_results(search_result, backend)
        modified_search = cast(SearchWebResult, dict(search_result))
        modified_search["results"] = [{
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
        return modified_search

    return result_dict


class OutboundEventTags(TypedDict):
    """Cross-cutting bookkeeping fields any outbound event can be stamped with while being
    forwarded — through a pipeline stage boundary or a subagent boundary — before reaching
    the frontend. Every OutboundEvent variant inherits these as optional fields."""
    _pipeline_stage: NotRequired[str]
    _workflow_execution: NotRequired[str | None]
    _subagent: NotRequired[bool]


class OutboundToolEvaluatingEvent(OutboundEventTags):
    type: Literal["tool_evaluating"]
    tool_id: str
    tool_name: str


class OutboundToolAutoApprovedEvent(OutboundEventTags):
    type: Literal["tool_auto_approved"]
    tool_id: str
    reason: str | None


class OutboundToolConfirmEvent(OutboundEventTags):
    type: Literal["tool_confirm"]
    tool_id: str
    tool_name: str
    arguments: dict[str, Any]
    preview: str
    diff_lines: NotRequired[list[DiffLine]]
    evaluator_reason: NotRequired[str]


class OutboundPlanProposalEvent(OutboundEventTags):
    type: Literal["plan_proposal"]
    plan_id: str
    plan: str


class OutboundAgentQuestionEvent(OutboundEventTags):
    type: Literal["agent_question"]
    question_id: str
    question: str
    options: NotRequired[list[str]]


class OutboundCtxUpdateEvent(OutboundEventTags):
    type: Literal["ctx_update"]
    ctx_tokens: int


class OutboundErrorEvent(OutboundEventTags):
    type: Literal["error"]
    message: str


class OutboundToolResultEvent(OutboundEventTags):
    type: Literal["tool_result"]
    tool_id: str
    tool_name: str
    content: str
    log_message: str | None
    ctx_tokens: int


class OutboundCompressingEvent(OutboundEventTags):
    type: Literal["compressing"]
    ctx_tokens: int
    ctx_limit: int
    reason: NotRequired[str]


class OutboundThinkingEvent(OutboundEventTags):
    type: Literal["thinking"]
    content: str


class OutboundContentEvent(OutboundEventTags):
    type: Literal["content"]
    content: str


class OutboundToolCallStartEvent(OutboundEventTags):
    type: Literal["tool_call_start"]
    tool_id: str
    tool_name: str


class OutboundToolCallChunkEvent(OutboundEventTags):
    type: Literal["tool_call_chunk"]
    tool_id: str
    chunk: str


class OutboundToolCallRawEvent(OutboundEventTags):
    type: Literal["tool_call_raw"]
    fragment: str


class OutboundGenerationEndEvent(OutboundEventTags):
    type: Literal["generation_end"]
    ctx_tokens: int


class OutboundContextEvent(OutboundEventTags):
    type: Literal["context"]
    ctx_tokens: int
    messages: PreparedMessages


class OutboundIterationEndEvent(OutboundEventTags):
    type: Literal["iteration_end"]
    prompt_tokens: int
    response_tokens: int


class OutboundDoneEvent(OutboundEventTags):
    type: Literal["done"]
    finished_without_response: bool


class OutboundModeChangedEvent(OutboundEventTags):
    type: Literal["mode_changed"]
    mode: str


class OutboundStageDoneEvent(OutboundEventTags):
    """Internal sentinel: a pipeline stage's own sub-session signals its run_stage loop that
    the stage finished. Consumed by run_stage itself, never forwarded to the frontend."""
    type: Literal["_stage_done"]


class OutboundPipelineSummaryEvent(OutboundEventTags):
    type: Literal["pipeline_summary"]
    label: str
    content: str
    notes: str | None


class OutboundWorkflowStartEvent(OutboundEventTags):
    type: Literal["workflow_start"]
    workflow_name: str
    run_id: str
    nodes: list[dict]


class OutboundStoppedEvent(OutboundEventTags):
    type: Literal["stopped"]


class OutboundStageEnterEvent(OutboundEventTags):
    type: Literal["stage_enter"]
    path: str
    execution_id: str
    stage_type: str
    invocation_number: int
    item_number: int | None
    item_total: int | None
    attempt_number: int | None
    attempt_total: int | None


class OutboundStageExitEvent(OutboundEventTags):
    type: Literal["stage_exit"]
    path: str
    execution_id: str
    status: str
    result: Any
    duration_ms: int


class OutboundLoopItemExitEvent(OutboundEventTags):
    type: Literal["loop_item_exit"]
    path: str
    item_number: int
    item_total: int
    success: bool
    status: str
    attempts_used: int
    item_result: Any


# Every shape session.emit() is called with anywhere in the backend (agent.py, pipeline.py,
# custom_workflow.py, tools/propose_plan.py, tools/subagent.py, ...) — this is the single
# outbound WebSocket event protocol shared across the whole agent loop.
OutboundEvent = (
    OutboundToolEvaluatingEvent
    | OutboundToolAutoApprovedEvent
    | OutboundToolConfirmEvent
    | OutboundPlanProposalEvent
    | OutboundAgentQuestionEvent
    | OutboundCtxUpdateEvent
    | OutboundErrorEvent
    | OutboundToolResultEvent
    | OutboundCompressingEvent
    | OutboundThinkingEvent
    | OutboundContentEvent
    | OutboundToolCallStartEvent
    | OutboundToolCallChunkEvent
    | OutboundToolCallRawEvent
    | OutboundGenerationEndEvent
    | OutboundContextEvent
    | OutboundIterationEndEvent
    | OutboundDoneEvent
    | OutboundModeChangedEvent
    | OutboundStageDoneEvent
    | OutboundPipelineSummaryEvent
    | OutboundWorkflowStartEvent
    | OutboundStoppedEvent
    | OutboundStageEnterEvent
    | OutboundStageExitEvent
    | OutboundLoopItemExitEvent
)


class AgentSession:
    """Manages bidirectional communication between the agent loop and the WebSocket client (or another client)."""

    def __init__(self):
        self.outbound: asyncio.Queue[OutboundEvent] = asyncio.Queue()
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._pending_plan_confirms: dict[str, asyncio.Future] = {}
        self._pending_user_inputs: dict[str, asyncio.Future] = {}
        self._compression_event: asyncio.Event = asyncio.Event()
        self._compression_conv_id: str | None = None
        self.apply_db_compressions_callback: Callable[[str], Awaitable[list[LLMMessage]]] | None = None
        self.finish_result: dict[str, Any] | None = None
        self._search_result_ids: set[str] = set()
        self._grepped_files: dict[str, int] = {}  # posix rel_path → count of grep calls that matched it
        self._sub_stage_counters: dict[str, int] = {}
        self.mode: str = "standard"
        self.working_directory: str | None = None
        self.last_user_message: str | None = None
        self.auto_safe_commands: list[str] = []
        self.compression_enabled: bool = True

    async def emit(self, event: OutboundEvent) -> None:
        await self.outbound.put(event)

    async def request_confirm(
        self, tool_id: str, tool_name: str, arguments: dict[str, Any], preview: str,
        diff_lines: list[DiffLine] | None = None,
    ) -> tuple[bool, str | None]:
        """Emit a confirmation request and suspend until the client responds.

        In auto/yolo mode applies rule-based + LLM safety evaluation instead of
        prompting the user for every tool call.
        """
        evaluator_reason: str | None = None
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
                safe_command_prefixes=self.auto_safe_commands if self.auto_safe_commands else None,
            )
            if verdict == "safe":
                if self.mode == "auto":
                    await self.emit({"type": "tool_auto_approved", "tool_id": tool_id, "reason": reason})
                return True, None
            # Dangerous: auto shows confirmation UI; yolo rejects and lets LLM handle it
            if self.mode == "yolo":
                return False, f"Safety evaluator blocked this action: {reason}"
            evaluator_reason = reason

        event: OutboundToolConfirmEvent = {
            "type": "tool_confirm",
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "preview": preview,
        }
        if diff_lines is not None:
            event["diff_lines"] = diff_lines
        if evaluator_reason is not None:
            event["evaluator_reason"] = evaluator_reason
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
        event: OutboundAgentQuestionEvent = {"type": "agent_question", "question_id": question_id, "question": question}
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


def _extract_tool_calls_from_thinking(thinking: str) -> list[ToolCall]:
    """
    Recover tool calls that the model emitted inside the thinking block without closing </think>.
    Qwen3 uses <tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>.
    Returns tool call dicts in the same shape as the normal stream-assembled list,
    marked with _recovered=True so the caller can emit streaming events before generation_end.
    """
    result: list[ToolCall] = []
    for index, parsed in enumerate(parse_all_tool_calls(thinking)):
        result.append({
            "id": f"tc-recovered-{index}",
            "function": {"name": parsed["name"], "arguments": parsed["arguments"]},
            "_recovered": True,
        })
    return result


def _strip_tool_call_blocks(thinking: str) -> str:
    """Remove <tool_call>…</tool_call> blocks from thinking before storing it in LLM context."""
    return TOOL_CALL_BLOCK_RE.sub("", thinking).strip()



def _content_as_str(content: str | list[dict]) -> str:
    """Coerce LLMMessage content to str, dropping multimodal list content (only ever present
    on user-role messages) since callers here only care about tool-role JSON-string content."""
    if isinstance(content, str):
        return content
    return ""


def _deduplicate_file_reads(messages: list[LLMMessage]) -> None:
    """modifies messages to remove file read that were superseded"""

    pairs = [(m.get("role", ""), _content_as_str(m.get("content", ""))) for m in messages]
    for i in _find_superseded_read_file_indices(pairs):  # only returns indices of role=="tool" messages
        tool_message_content = _content_as_str(messages[i].get("content", ""))
        try:
            parsed_content = json.loads(tool_message_content)
        except (json.JSONDecodeError, ValueError):
            parsed_content = {}
        path = parsed_content.get("path", "")
        tool_call_id = parsed_content.get("tool_call_id")
        messages[i]["content"] = json.dumps({
            "tool": "read_file",
            "status": "evicted",
            "path": path,
            "tool_call_id": tool_call_id,
            "reason": "file content removed — analysis was expressed in conversation above, superseded by later read",
        })




def _log_context(messages: PreparedMessages) -> None:
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
                j: ToolResult = json.loads(content)
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


async def _track_tokens(
    messages: list[LLMMessage],
    tools: list[ToolDict],
    session: "AgentSession",
    label: str,
    prepared: PreparedMessages | None = None,
) -> int:
    """Count context tokens, log the result, and emit ctx_update to the frontend."""
    if prepared is None:
        prepared = backend.prepare_messages(messages)
    tokens = await backend.count_tokens(prepared, tools)
    print(f"[tokens] {label}: {tokens}/{CTX_LIMIT}")
    await session.emit({"type": "ctx_update", "ctx_tokens": tokens})
    return tokens


def _parse_tool_calls(tool_calls_acc: dict[int, ToolCallAccEntry], message: AssistantMessage) -> list[ToolCall]:
    """Build the tool_calls list from streamed fragments, with thinking-block recovery fallback."""

    tool_calls: list[ToolCall] = []
    for acc in (tool_calls_acc[i] for i in sorted(tool_calls_acc)):
        try:
            arguments = json.loads(acc["arguments_str"]) if acc["arguments_str"] else {}
        except json.JSONDecodeError:
            logger.warning("Malformed tool call arguments for %s, skipping: %r", acc["name"], acc["arguments_str"])
            continue
        tool_calls.append(ToolCall(id=acc["id"], function=ToolCallFunction(name=acc["name"], arguments=arguments)))

    # Recover tool calls embedded in thinking when the model forgot to close </think> first.
    # Only attempt recovery when both regular content and stream-parsed tool calls are absent.
    if len(tool_calls) == 0 and message["content"] == "" and message["thinking"] != "":
        recovered = _extract_tool_calls_from_thinking(message["thinking"])
        if recovered:
            logger.warning("Recovering %d tool call(s) embedded in thinking block", len(recovered))
            tool_calls = recovered

    return tool_calls


async def _emit_compression_disabled_error(session: "AgentSession", detail: str) -> None:
    """Emit the standard error for a workflow stage session hitting a context limit with
    compression disabled (AgentSession.compression_enabled == False) — caller stops the turn."""
    await session.emit({
        "type": "error",
        "message": f"Context limit reached: {detail}. Compression is disabled for workflow stages — adjust the stage's inputs or definition and resume from here.",
    })


async def _execute_tool_calls(
    tool_calls: list[ToolCall],
    messages: list[LLMMessage],
    session: AgentSession,
    toolset: "ToolSet",
    working_directory: str | None,
) -> bool:
    """Execute each tool call, append results to messages, and handle mid-run compression.

    Returns True when context overflows even after compression (caller should abort).
    Mutates messages in place.
    """
    ctx_before = await _track_tokens(messages, toolset.tools, session, "context before tool execution")

    effective_registry = {**TOOL_REGISTRY, **(toolset.extra_tools if toolset.extra_tools is not None else {})}

    for tool_call in tool_calls:
        tool_name: str = tool_call.get("function", {}).get("name", "")
        tool_args: dict[str, Any] = tool_call.get("function", {}).get("arguments", {})
        call_id: str = tool_call.get("id", f"tc-{id(tool_call)}")


        if tool_name not in effective_registry:
            result_dict: ToolResult = {"tool": tool_name, "status": "error", "error": {"message": f"Unknown tool: {tool_name}"}}
            log_msg = None
        else:
            tool_instance = effective_registry[tool_name]
            result_dict = await tool_instance.execute(tool_args, session, working_directory)
            result_dict = await _maybe_preshrink_tool_output(tool_name, result_dict, working_directory)
            log_msg = tool_instance.label(tool_args)

        result_dict["tool_call_id"] = call_id
        tool_output = json.dumps(result_dict)
        messages.append({"role": "tool", "content": tool_output})

        ctx_after = await _track_tokens(messages, toolset.tools, session, f"context after tool result '{tool_name}'")

        if ctx_after > CTX_COMPRESS_THRESHOLD:
            # Emit tool_result first so frontend saves it to DB before compressing.
            await session.emit({
                "type": "tool_result",
                "tool_id": call_id,
                "tool_name": tool_name,
                "content": tool_output,
                "log_message": log_msg,
                "ctx_tokens": ctx_after,
            })
            if not session.compression_enabled:
                await _emit_compression_disabled_error(session, f"{ctx_after}/{CTX_LIMIT} tokens")
                return True
            await session.emit({"type": "compressing", "ctx_tokens": ctx_after, "ctx_limit": CTX_LIMIT})
            conv_id = await session.await_compression()
            if conv_id is not None and session.apply_db_compressions_callback is not None:
                refreshed = await session.apply_db_compressions_callback(conv_id)
                messages[:] = refreshed
            ctx_after_compress = await _track_tokens(messages, toolset.tools, session, "context after compression")
            if ctx_after_compress > CTX_LIMIT:
                await session.emit({
                   "type": "error",
                   "message": f"Context still exceeds limit after compression: {ctx_after_compress}/{CTX_LIMIT} tokens"
                })
                return True
            continue

        await session.emit({
            "type": "tool_result",
            "tool_id": call_id,
            "tool_name": tool_name,
            "content": tool_output,
            "log_message": log_msg,
            "ctx_tokens": ctx_after,
        })

    return False


async def _stream_llm(
    prepared: PreparedMessages,
    tools: list[ToolDict],
    max_tokens: int,
    session: AgentSession,
    tool_choice: dict[str, Any] | str | None = None,
) -> GenerationResult:
    """Stream one LLM completion, emit events to session, and return accumulated result."""
    message: AssistantMessage = {"role": "assistant", "content": "", "thinking": ""}
    tool_calls_acc: dict[int, ToolCallAccEntry] = {}
    eval_count: int = 0
    done_reason: str = ""

    logger.info("[llm] sending completion request to backend (%d messages, %d tools, max_tokens=%d, tool_choice=%r)", len(prepared), len(tools), max_tokens, tool_choice)
    first_event_received = False
    async for event in backend.stream_completion(prepared, tools, temperature=1.0, max_tokens=max_tokens, tool_choice=tool_choice):
        if not first_event_received:
            logger.info("[llm] backend started responding")
            print(f"{_ANSI_BLUE}\n--- LLM stream start ---{_ANSI_RESET}", flush=True)
            first_event_received = True
        if event["type"] == "thinking":
            message["thinking"] += event["content"]
            print(f"{_ANSI_DIM}{event['content']}{_ANSI_RESET}", end="", flush=True)
            await session.emit({"type": "thinking", "content": event["content"]})

        elif event["type"] == "content":
            message["content"] += event["content"]
            print(f"{_ANSI_GREEN}{event['content']}{_ANSI_RESET}", end="", flush=True)
            await session.emit({"type": "content", "content": event["content"]})

        elif event["type"] == "tool_call_start":
            idx = event["index"]
            tool_calls_acc[idx] = ToolCallAccEntry(id=event["id"], name=event["name"], arguments_str="")
            print(f"{_ANSI_MAGENTA}\n[tool_call] {event['name']}({_ANSI_RESET}", end="", flush=True)
            await session.emit({"type": "tool_call_start", "tool_id": event["id"], "tool_name": event["name"]})

        elif event["type"] == "tool_call_arg":
            idx = event["index"]
            if idx in tool_calls_acc:
                tool_calls_acc[idx]["arguments_str"] += event["fragment"]
            print(f"{_ANSI_MAGENTA}{event['fragment']}{_ANSI_RESET}", end="", flush=True)
            tool_call_entry = tool_calls_acc.get(idx)
            tool_id = tool_call_entry["id"] if tool_call_entry is not None else ""
            await session.emit({"type": "tool_call_chunk", "tool_id": tool_id, "chunk": event["fragment"]})

        elif event["type"] == "tool_call_raw":
            print(f"{_ANSI_MAGENTA}{event['fragment']}{_ANSI_RESET}", end="", flush=True)
            await session.emit({"type": "tool_call_raw", "fragment": event["fragment"]})

        elif event["type"] == "done":
            eval_count = event["completion_tokens"]
            done_reason = event["finish_reason"]

    if not first_event_received:
        logger.warning("[llm] backend stream ended with no events at all — check llama-server is reachable")
    else:
        print(f"{_ANSI_BLUE}\n--- LLM stream end ---{_ANSI_RESET}", flush=True)
    logger.info("[llm] completion done — finish_reason=%s completion_tokens=%d tool_calls=%d", done_reason, eval_count, len(tool_calls_acc))
    return GenerationResult(
        message=message,
        tool_calls_acc=tool_calls_acc,
        eval_count=eval_count,
        done_reason=done_reason,
    )


async def _append_assistant_message(
    message: AssistantMessage,
    tool_calls: list[ToolCall],
    messages: list[LLMMessage],
    tools: list[ToolDict],
    session: AgentSession,
) -> bool:
    """Append the assistant turn to messages. Returns True if a message was appended."""

    content = message["content"]
    thinking = message["thinking"]

    if content != "":
        messages[:] = [m for m in messages if not m.get("_transient")]
        if len(tool_calls) > 0:
            message["tool_calls"] = tool_calls
        # AssistantMessage.content is str-only, narrower than LLMMessage's str | list[dict];
        # TypedDict fields are invariant so the cast is needed even though this is a safe widening.
        messages.append(cast(LLMMessage, message))
        appended = True
    elif thinking != "" or len(tool_calls) > 0:
        thinking_for_context = (
            _strip_tool_call_blocks(thinking)
            if tool_calls and tool_calls[0].get("_recovered")
            else thinking
        )
        messages.append({
            "role": "assistant",
            "content": f"<think>{thinking_for_context}</think>",
            "tool_calls": tool_calls,
        })
        appended = True
    else:
        appended = False

    if appended:
        for tool_call in tool_calls:
            if tool_call.get("_recovered"):
                call_id = tool_call["id"]
                call_name = tool_call["function"]["name"]
                args_str = json.dumps(tool_call["function"]["arguments"])
                await session.emit({"type": "tool_call_start", "tool_id": call_id, "tool_name": call_name})
                await session.emit({"type": "tool_call_chunk", "tool_id": call_id, "chunk": args_str})
        ctx_after_gen = await _track_tokens(messages, tools, session, "context after generation")
        await session.emit({"type": "generation_end", "ctx_tokens": ctx_after_gen})

    return appended


async def chat_with_tools(
    messages: list[LLMMessage],
    session: AgentSession,
    toolset: "ToolSet",
    working_directory: str | None,

    allow_length_compression: bool = True,
    tool_choice: dict[str, Any] | str | None = None,
) -> TurnResult:
    """One iteration of the LLM call + tool execution loop.

    tool_choice forwards to the backend to mechanically constrain generation (grammar-based on
    llama.cpp) — pass a specific {"type": "function", "function": {"name": ...}} to guarantee
    that exact tool gets called instead of the model optionally choosing to answer in prose.
    """
    prepared = backend.prepare_messages(messages)
    _log_context(prepared)

    prompt_eval_count = await _track_tokens(messages, toolset.tools, session, "context before generation", prepared=prepared)
    max_tokens = CTX_LIMIT - prompt_eval_count

    await session.emit({"type": "context", "ctx_tokens": prompt_eval_count, "messages": prepared})

    generation = await _stream_llm(prepared, toolset.tools, max_tokens, session, tool_choice=tool_choice)

    print(f"[tokens] prompt_tokens={prompt_eval_count} completion_tokens={generation.eval_count} finish_reason={generation.done_reason}")

    if generation.done_reason == "length":
        if not allow_length_compression or not session.compression_enabled:
            await session.emit({
                "type": "error",
                "message": f"Context limit reached during generation: {prompt_eval_count + generation.eval_count}/{CTX_LIMIT} tokens. The response was cut off.",
            })
            return TurnResult(is_done=True, finished_without_response=False)
        await session.emit({
            "type": "compressing",
            "ctx_tokens": prompt_eval_count + generation.eval_count,
            "ctx_limit": CTX_LIMIT,
            "reason": "length",
        })
        conv_id = await session.await_compression()
        if conv_id is not None and session.apply_db_compressions_callback is not None:
            refreshed = await session.apply_db_compressions_callback(conv_id)
            messages[:] = refreshed
        return TurnResult(is_done=False, finished_without_response=False, length_compressed=True)

    tool_calls = _parse_tool_calls(generation.tool_calls_acc, generation.message)

    finished_without_response = generation.message["content"] == "" and len(tool_calls) == 0
    if finished_without_response:
        logger.warning("Agent finished without response: no content, no tool calls")

    await _append_assistant_message(generation.message, tool_calls, messages, toolset.tools, session)

    if len(tool_calls) > 0:
        overflow = await _execute_tool_calls(tool_calls, messages, session, toolset, working_directory)
        if overflow:
            return TurnResult(is_done=True, finished_without_response=False)

    # Emit iteration_end after tool results so the frontend receives tool_result events
    # before iteration_end. The frontend rotation logic patches tool results from iteration N
    # with prompt_tokens from iteration N+1 — this ordering makes that work correctly.
    await session.emit({
        "type": "iteration_end",
        "prompt_tokens": prompt_eval_count,
        "response_tokens": generation.eval_count,
    })

    _deduplicate_file_reads(messages)
    return TurnResult(
      is_done=len(tool_calls) == 0,
      finished_without_response=finished_without_response
    )


async def _maybe_compress_on_iteration_threshold(
    session: AgentSession,
    messages: list[LLMMessage],
    iteration_count: int,
    turn: TurnResult,
) -> bool:
    """Trigger working memory compression when the iteration threshold is reached.

    Returns True if compression was triggered (caller should reset iteration_count).
    Mutates messages in place with the refreshed context.
    """
    if turn.length_compressed or iteration_count < WORKING_MEMORY_ITERATION_THRESHOLD:
        return False
    if not session.compression_enabled:
        await _emit_compression_disabled_error(session, f"iteration threshold ({WORKING_MEMORY_ITERATION_THRESHOLD}) hit")
        turn.is_done = True
        return False
    logger.info("Iteration threshold reached (%d) — triggering compression", iteration_count)
    await session.emit({"type": "compressing", "ctx_tokens": 0, "ctx_limit": CTX_LIMIT, "reason": "iteration_threshold"})
    conv_id = await session.await_compression()
    if conv_id is not None and session.apply_db_compressions_callback is not None:
        refreshed = await session.apply_db_compressions_callback(conv_id)
        messages[:] = refreshed
    return True


async def run_agent(
    session: AgentSession,
    messages: list[LLMMessage],
    toolset: "ToolSet",
    working_directory: str | None,
) -> None:
    """Run the full agent loop until done, emitting events via session."""
    try:
        turn = TurnResult(is_done=False, finished_without_response=False)
        iteration_count = 0
        while not turn.is_done:
            if await _maybe_compress_on_iteration_threshold(session, messages, iteration_count, turn):
                iteration_count = 0
            # _maybe_compress_on_iteration_threshold mutates turn.is_done in place when it emits
            # the compression-disabled error above instead of compressing — the while condition
            # already passed for this pass, so without this check the loop would call
            # chat_with_tools one more time on a session that just errored out.
            if turn.is_done:
                break
            allow_length_compression = not turn.length_compressed
            turn = await chat_with_tools(
                messages, session, toolset, working_directory,
                allow_length_compression=allow_length_compression
            )
            iteration_count += 1
        await session.emit({"type": "done", "finished_without_response": turn.finished_without_response})
    except asyncio.CancelledError:
        await session.emit({"type": "error", "message": "Agent was aborted"})
    except aiohttp.ClientConnectorError as e:
        logger.error("LLM backend connection error: %s", e)
        await session.emit({"type": "error", "message": "LLM backend is not running"})
    except Exception as e:
        logger.exception("Unexpected error in agent loop")
        await session.emit({"type": "error", "message": str(e)})
