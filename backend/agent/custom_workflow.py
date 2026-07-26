import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import aiohttp

from agent.agent import AgentSession, run_agent
from agent.tools.base import ToolDict
from conv_helpers import ToolSet
from message_types import LLMMessage
from agent.finish_tools import BaseFinishTool
from agent.pipeline import run_stage
from agent.workflow_coordinator import run_coordinator_action
from agent.workflow_loader import (
    WorkflowDefinition,
    WorkflowStageDefinition,
    _BUILTIN_FINISH_TOOL_CLASSES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slot registry helpers
# ---------------------------------------------------------------------------

def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for dot-access in eval conditions."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _resolve_path(path: str, slots: dict[str, Any]) -> Any:
    """Resolve 'slot.field.subfield' dot-path from the slot registry. Returns None if missing."""
    parts = path.strip().split(".")
    value = slots.get(parts[0])
    for part in parts[1:]:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
    return value


def resolve_template(template: str, slots: dict[str, Any]) -> str:
    """Replace {{slot.field}} and {{slot.field | length}} references in a string."""
    def replace(match: re.Match) -> str:
        expr = match.group(1).strip()
        if expr.endswith(" | length"):
            path = expr[: -len(" | length")].strip()
            value = _resolve_path(path, slots)
            return str(len(value)) if value is not None else "0"
        value = _resolve_path(expr, slots)
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    return re.sub(r"\{\{([^}]+)\}\}", replace, template)


def resolve_value(expr: str, slots: dict[str, Any]) -> Any:
    """Resolve a single {{slot.field}} expression to its actual Python value (not a string)."""
    match = re.fullmatch(r"\{\{([^}]+)\}\}", expr.strip())
    if match is None:
        raise ValueError(f"Expected a single {{{{...}}}} expression, got: {expr!r}")
    return _resolve_path(match.group(1).strip(), slots)


_SAFE_EVAL_GLOBALS = {"__builtins__": {}, "len": len, "None": None, "True": True, "False": False}


def evaluate_condition(condition: str, slots: dict[str, Any]) -> bool:
    """Evaluate a Python-like condition expression against the slot registry.

    Slots are accessible by name with dot notation (e.g. plan.compile_command).
    Returns True when condition is empty (unconditional).
    """
    if condition is None or condition == "":
        return True
    context = {k: _to_namespace(v) for k, v in slots.items()}
    try:
        return bool(eval(condition, _SAFE_EVAL_GLOBALS, context))  # noqa: S307
    except Exception as exc:
        logger.warning("[workflow] condition eval failed %r: %s", condition, exc)
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_MAX_STAGE_RESULT_CHARS = 4000


def _truncate_stage_result(result: Any) -> Any:
    """Keep stage_exit payloads small enough to stream.

    Some stage products are inherently huge — chunk_file returns the entire source file, and a
    loop aggregate holds every item's output — and shipping those verbatim would duplicate the
    whole payload over the websocket. Oversized results are replaced by a description of their
    shape; the run view shows a summary line and the detail pane, never the raw bulk.
    """
    if result is None:
        return None
    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if len(serialized) <= _MAX_STAGE_RESULT_CHARS:
        return result
    summary: dict[str, Any] = {
        "_truncated": True,
        "_size_chars": len(serialized),
        "_type": type(result).__name__,
    }
    if isinstance(result, dict):
        summary["_keys"] = sorted(str(key) for key in result.keys())
    if isinstance(result, list):
        summary["_length"] = len(result)
    return summary


def _serialize_stage_nodes(
    stages: list[WorkflowStageDefinition], path_prefix: str = ""
) -> list[dict]:
    """Serialize the static stage structure for the workflow_start event.

    Emitted once up front so the run view can draw the whole plan before anything executes,
    then decorate it with runtime status. `path` is the stable dotted identity used by every
    later stage_enter / stage_exit event; loop inner stages are nested under `children`.
    """
    nodes: list[dict] = []
    for stage in stages:
        path = path_prefix + stage.name
        nodes.append({
            "path": path,
            "name": stage.name,
            "type": stage.type,
            "finish_tool": stage.finish_tool_name if stage.type == "llm" else None,
            "tools": stage.tools + stage.agents,
            "over": stage.over if stage.over != "" else None,
            "attempt_total": stage.max_retries + 1 if stage.type == "loop" else None,
            "children": _serialize_stage_nodes(stage.inner_stages, f"{path}."),
        })
    return nodes


@dataclass
class StageOutcome:
    """What a dispatched stage produced.

    jump_to is set only by branch stages, naming the stage to continue from.
    result is the stage's finish-tool args (llm/agent), coordinator output, or branch decision —
    whatever should be shown as that stage's product in the run view. None when there is none.
    """

    jump_to: str | None = None
    result: dict | None = None


@dataclass
class LoopContext:
    """Identity of the loop item and retry attempt an inner stage is currently running for.

    Deliberately separate from run_stage's #N counter: that counter counts every invocation of
    a stage name across the whole workflow, so a retry advances it just like a new item would.
    This carries the real item index so logs can show which chunk is actually being worked on.
    """

    item_number: int
    item_total: int
    attempt_number: int
    attempt_total: int

    def label(self) -> str:
        """Render as 'item 11/84 attempt 2/3' for log display."""
        return (
            f"item {self.item_number}/{self.item_total} "
            f"attempt {self.attempt_number}/{self.attempt_total}"
        )


class CustomWorkflowOrchestrator:
    """Drives a YAML-defined workflow through typed stages with a shared slot registry.

    Slot registry accumulates finish-tool results and coordinator outputs so later
    stages can reference earlier ones via {{slot.field}} in prompts and conditions.
    """

    def __init__(self, workflow: WorkflowDefinition, working_directory: str | None, tools: list[ToolDict] | None = None):
        self._workflow = workflow
        self._working_directory = working_directory
        self._tools = tools or []
        self._loop_context: LoopContext | None = None
        self._invocation_counters: dict[str, int] = {}

    async def run(self, session: AgentSession, user_message: str, messages: list[LLMMessage]) -> None:
        """Entry point — runs all workflow stages and emits events via session.

        Temporarily overrides session.mode and session.auto_safe_commands if the workflow
        declares them, then restores the previous values when the workflow finishes.
        """
        logger.info("[workflow:%s] starting — user_message=%r messages=%d", self._workflow.name, user_message, len(messages))
        saved_mode = session.mode
        saved_auto_safe_commands = session.auto_safe_commands
        if self._workflow.mode is not None:
            session.mode = self._workflow.mode
            logger.info("[workflow:%s] mode temporarily set to '%s'", self._workflow.name, self._workflow.mode)
        session.auto_safe_commands = list(self._workflow.auto_safe_commands)
        await session.emit({
            "type": "workflow_start",
            "workflow_name": self._workflow.name,
            "nodes": _serialize_stage_nodes(self._workflow.stages),
        })
        try:
            await self._run_workflow(session, user_message, messages)
        except asyncio.CancelledError:
            logger.info("[workflow:%s] aborted (CancelledError)", self._workflow.name)
            await session.emit({"type": "error", "message": f"Workflow '{self._workflow.name}' was aborted"})
        except aiohttp.ClientConnectorError as exc:
            logger.error("[workflow:%s] LLM backend connection error: %s", self._workflow.name, exc)
            await session.emit({"type": "error", "message": "LLM backend is not running"})
        except _WorkflowAbort as exc:
            logger.error("[workflow:%s] aborted: %s", self._workflow.name, exc)
            await session.emit({"type": "error", "message": str(exc)})
        except Exception as exc:
            import traceback
            full_tb = traceback.format_exc()
            logger.exception("[workflow:%s] unexpected error", self._workflow.name)
            await session.emit({"type": "error", "message": full_tb})
        finally:
            session.mode = saved_mode
            session.auto_safe_commands = saved_auto_safe_commands
            if self._workflow.mode is not None:
                logger.info("[workflow:%s] mode restored to '%s'", self._workflow.name, saved_mode)

    async def _run_workflow(
        self, session: AgentSession, user_message: str, messages: list[LLMMessage]
    ) -> None:
        effective_message = user_message if user_message.strip() != "" else f"/{self._workflow.name}"
        slots: dict[str, Any] = {"user_message": effective_message}
        stages = self._workflow.stages
        stage_index = {s.name: i for i, s in enumerate(stages)}
        logger.info("[workflow:%s] %d stages: %s", self._workflow.name, len(stages), [s.name for s in stages])
        current = 0

        while current < len(stages):
            stage = stages[current]
            logger.info("[workflow:%s] dispatching stage '%s' (type=%s)", self._workflow.name, stage.name, stage.type)
            outcome = await self._dispatch(stage, session, messages, slots)
            logger.info("[workflow:%s] stage '%s' done — slots keys: %s", self._workflow.name, stage.name, list(slots.keys()))
            if outcome.jump_to is not None:
                jump_to = outcome.jump_to
                if jump_to not in stage_index:
                    raise ValueError(f"Branch target '{jump_to}' not found in workflow stages")
                logger.info("[workflow:%s] jumping to stage '%s'", self._workflow.name, jump_to)
                current = stage_index[jump_to]
            else:
                current += 1

        logger.info("[workflow:%s] all stages complete", self._workflow.name)
        await session.emit({"type": "done", "finished_without_response": False})

    def _next_execution_id(self, path: str) -> str:
        """Allocate a unique id for one invocation of a stage path (e.g. 'translate_loop.translate_chunk#41')."""
        count = self._invocation_counters.get(path, 0) + 1
        self._invocation_counters[path] = count
        return f"{path}#{count}"

    async def _emit_stage_enter(
        self, session: AgentSession, stage: WorkflowStageDefinition, path: str, execution_id: str
    ) -> None:
        """Announce that a stage invocation is starting, carrying its loop item and attempt identity."""
        loop_context = self._loop_context
        await session.emit({
            "type": "stage_enter",
            "path": path,
            "execution_id": execution_id,
            "stage_type": stage.type,
            "invocation_number": self._invocation_counters[path],
            "item_number": None if loop_context is None else loop_context.item_number,
            "item_total": None if loop_context is None else loop_context.item_total,
            "attempt_number": None if loop_context is None else loop_context.attempt_number,
            "attempt_total": None if loop_context is None else loop_context.attempt_total,
        })

    async def _emit_stage_exit(
        self,
        session: AgentSession,
        path: str,
        execution_id: str,
        status: str,
        result: Any,
        started_at: float | None,
    ) -> None:
        """Announce that a stage invocation ended. started_at is None for stages that never ran."""
        duration_ms = 0 if started_at is None else int((time.monotonic() - started_at) * 1000)
        await session.emit({
            "type": "stage_exit",
            "path": path,
            "execution_id": execution_id,
            "status": status,
            "result": _truncate_stage_result(result),
            "duration_ms": duration_ms,
        })

    async def _dispatch(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[LLMMessage],
        slots: dict[str, Any],
        path_prefix: str = "",
    ) -> StageOutcome:
        """Run one stage, bracketed by stage_enter / stage_exit events for the run view.

        Single choke point for the skip condition, the invocation counter and the timing, so every
        stage type reports the same lifecycle no matter which runner handles it. A runner that
        raises still reports stage_exit(failed) before the exception propagates — _run_loop relies
        on catching it to retry the item.
        """
        path = path_prefix + stage.name
        execution_id = self._next_execution_id(path)

        # Loops gate on entry_condition, everything except branches on condition.
        skip_condition = stage.entry_condition if stage.type == "loop" else stage.condition
        if stage.type != "branch" and not evaluate_condition(skip_condition, slots):
            logger.info("[workflow] skipping %s stage '%s' (condition false)", stage.type, stage.name)
            await self._emit_stage_enter(session, stage, path, execution_id)
            await self._emit_stage_exit(session, path, execution_id, "skipped", None, None)
            return StageOutcome()

        await self._emit_stage_enter(session, stage, path, execution_id)
        started_at = time.monotonic()
        try:
            outcome = await self._run_by_type(stage, session, messages, slots, path, execution_id)
        except Exception:
            await self._emit_stage_exit(session, path, execution_id, "failed", None, started_at)
            raise
        await self._emit_stage_exit(session, path, execution_id, "done", outcome.result, started_at)
        return outcome

    async def _run_by_type(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[LLMMessage],
        slots: dict[str, Any],
        path: str,
        execution_id: str,
    ) -> StageOutcome:
        """Route a stage to the runner for its type and return what it produced."""
        if stage.type == "llm":
            return StageOutcome(result=await self._run_llm(stage, session, slots, execution_id))
        if stage.type == "coordinator":
            return StageOutcome(result=await self._run_coordinator(stage, session, slots))
        if stage.type == "branch":
            return self._resolve_branch(stage, slots)
        if stage.type == "loop":
            return StageOutcome(result=await self._run_loop(stage, session, messages, slots, path))
        if stage.type == "respond":
            await self._run_respond(stage, session, messages, slots)
            return StageOutcome()
        if stage.type == "agent":
            return StageOutcome(result=await self._run_isolated_agent(stage, session, slots, execution_id))
        raise ValueError(f"Unknown stage type '{stage.type}'")

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_llm(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        slots: dict[str, Any],
        execution_id: str,
    ) -> dict:
        """Run an LLM stage: resolve prompts, run the loop, store and return the finish result."""
        system_prompt = resolve_template(stage.system_prompt, slots)
        user_prompt = resolve_template(stage.user_prompt, slots)
        logger.info("[workflow] llm stage '%s' — tools=%s finish_tool=%s user_prompt=%r",
                    stage.name, stage.tools + stage.agents, stage.finish_tool_name, user_prompt[:120])
        stage_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tool_names = stage.tools + stage.agents
        finish_tool = self._workflow.make_finish_tool(stage.finish_tool_name)

        result = await run_stage(
            stage.name,
            stage_messages,
            tool_names,
            finish_tool,
            session,
            self._working_directory,
            max_iterations=stage.max_iterations,
            inject_turn_reminders=stage.inject_turn_reminders,
            stage_label=None if self._loop_context is None else self._loop_context.label(),
            execution_id=execution_id,
        )

        logger.info("[workflow] llm stage '%s' result keys: %s", stage.name, list(result.keys()) if result else "None")
        slots[stage.name] = result
        return result

    async def _run_coordinator(
        self, stage: WorkflowStageDefinition, session: AgentSession, slots: dict[str, Any]
    ) -> Any:
        """Run a coordinator action: resolve inputs, execute action, store and return its output."""
        logger.info("[workflow] coordinator '%s' action='%s' inputs=%s", stage.name, stage.action, list(stage.action_input.keys()))
        resolved_inputs = {k: resolve_value(v, slots) for k, v in stage.action_input.items()}
        output = await run_coordinator_action(stage.action, resolved_inputs, session, self._working_directory)

        if stage.action_output != "":
            slots[stage.action_output] = output
        logger.info("[workflow] coordinator '%s' done — output_slot='%s'", stage.name, stage.action_output)
        return output

    def _resolve_branch(self, stage: WorkflowStageDefinition, slots: dict[str, Any]) -> StageOutcome:
        """Evaluate a branch condition and return the target stage name as the jump target.

        The decision rides along as the stage result so the run view can show which way it went
        without needing a dedicated event.
        """
        condition_result = evaluate_condition(stage.condition, slots)
        target = stage.if_true if condition_result else stage.if_false
        logger.info("[workflow] branch '%s' → %s (%s)", stage.name, target, condition_result)
        return StageOutcome(
            jump_to=target,
            result={"condition_result": condition_result, "target": target},
        )

    async def _run_loop(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[LLMMessage],
        slots: dict[str, Any],
        path: str,
    ) -> dict:
        """Run a loop stage: iterate over a list or repeat until exit_condition.

        Returns a compact tally rather than the aggregated per-item results: those are already
        stored in the loop_output slot, and streaming them again as a stage result would repeat
        every item's full output over the event stream.
        """
        items: list[Any]
        if stage.over != "":
            items = resolve_value(stage.over, slots) or []
        else:
            items = [None]  # single-item sentinel for non-list loops

        aggregated: list[dict] = []
        item_total = len(items)
        attempt_total = stage.max_retries + 1
        succeeded_count = 0
        failed_count = 0
        logger.info("[workflow] loop '%s' — %d item(s), up to %d attempt(s) each", stage.name, item_total, attempt_total)

        for item_index, item in enumerate(items):
            item_number = item_index + 1
            if stage.over != "":
                slots[stage.item_var] = item
            # Reset on_retry slots to their default (empty string) before first attempt
            for slot_name in stage.on_retry:
                slots[slot_name] = ""

            item_success = False
            for attempt in range(attempt_total):
                self._loop_context = LoopContext(
                    item_number=item_number,
                    item_total=item_total,
                    attempt_number=attempt + 1,
                    attempt_total=attempt_total,
                )
                if attempt > 0:
                    logger.info("[workflow] loop '%s' item %d/%d — retry attempt %d/%d", stage.name, item_number, item_total, attempt + 1, attempt_total)
                    for slot_name, template in stage.on_retry.items():
                        slots[slot_name] = resolve_template(template, slots)

                try:
                    for inner in stage.inner_stages:
                        await self._dispatch(inner, session, messages, slots, path_prefix=f"{path}.")
                except Exception as exc:
                    # An inner stage (e.g. the model failing to call its required finish tool)
                    # raises instead of returning — treat that as a failed attempt so it goes
                    # through the same retry path as a failed exit_condition, instead of
                    # escaping the loop and aborting the whole workflow after one try.
                    logger.warning("[workflow] loop '%s' item %d/%d attempt %d/%d raised: %s", stage.name, item_number, item_total, attempt + 1, attempt_total, exc)
                    continue

                if evaluate_condition(stage.exit_condition, slots):
                    item_success = True
                    break

            attempts_used = attempt + 1
            self._loop_context = None

            await session.emit({
                "type": "loop_item_exit",
                "path": path,
                "item_number": item_number,
                "item_total": item_total,
                "success": item_success,
                "attempts_used": attempts_used,
            })

            if not item_success:
                if stage.on_max_retries == "abort_workflow":
                    raise _WorkflowAbort(f"Loop '{stage.name}' item {item_number}/{item_total} exhausted {attempt_total} attempts — aborting workflow")
                logger.warning("[workflow] loop '%s' item %d/%d failed all %d attempt(s) — skipping it and continuing", stage.name, item_number, item_total, attempt_total)
                failed_count += 1
            else:
                logger.info("[workflow] loop '%s' item %d/%d done", stage.name, item_number, item_total)
                succeeded_count += 1

            if stage.over != "":
                aggregated.append(_collect_inner_results(stage, slots, item, item_success))

        if stage.loop_output != "" and stage.over != "":
            task_summary = _format_task_summary(aggregated)
            slots[stage.loop_output] = {"items": aggregated, "task_summary": task_summary}

        return {
            "item_total": item_total,
            "succeeded": succeeded_count,
            "failed": failed_count,
        }

    async def _run_respond(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[LLMMessage],
        slots: dict[str, Any],
    ) -> None:
        """Run the plain agent loop on the full conversation history.

        Unlike isolated stages, this uses the original DB conversation so the model
        can reference everything the user said before the workflow ran. Workflow results
        are injected as a synthetic tool result so the model treats them as internal
        data rather than user-visible content.
        """
        user_prompt = slots.get("user_message", "")
        invocation = f'The user invoked the workflow "{self._workflow.name}"'
        if user_prompt is not None and user_prompt.strip() != "":
            invocation += f" with the following prompt: {user_prompt}"

        working_messages = list(messages)
        last_role = working_messages[-1].get("role") if working_messages else None
        last_content = working_messages[-1].get("content", "") if working_messages else ""
        logger.info("[workflow] respond stage — %d base messages, last role=%r content=%r", len(messages), last_role, (last_content or "")[:80])

        if working_messages and last_role == "user" and not (last_content or "").strip():
            logger.info("[workflow] respond: patching empty last user message with invocation header")
            working_messages[-1] = {**working_messages[-1], "content": invocation}
        else:
            logger.info("[workflow] respond: appending new user message with invocation header")
            working_messages.append({"role": "user", "content": invocation})

        if stage.message_suffix != "":
            suffix = resolve_template(stage.message_suffix, slots)
            logger.info("[workflow] respond: injecting tool result (%d chars)", len(suffix))
            tool_call_id = f"wf_{uuid.uuid4().hex[:8]}"
            working_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": "workflow_result", "arguments": "{}"}}],
            })
            working_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": suffix,
            })

        logger.info("[workflow] respond: calling run_agent with %d messages", len(working_messages))
        await run_agent(session, working_messages, ToolSet(tools=self._tools, extra_tools=None), self._working_directory)
        logger.info("[workflow] respond: run_agent returned")

    async def _run_isolated_agent(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        slots: dict[str, Any],
        execution_id: str,
    ) -> dict:
        """Run a named agent from agents/ as an isolated stage. Result stored in slots and returned.

        Resolves agent YAML relative to the workflow's own directory first (agents/
        subdirectory), then falls back to the global backend/agents/ directory.
        """
        from agent.workflow_loader import load_agent
        from pathlib import Path

        ref_filename = f"{stage.workflow_ref}.yaml"
        local_agent_path = self._workflow.directory / "agents" / ref_filename
        global_agent_path = Path(__file__).parent.parent / "agents" / ref_filename
        agent_path = local_agent_path if local_agent_path.exists() else global_agent_path
        agent_def = load_agent(agent_path)
        finish_tool_classes = dict(_BUILTIN_FINISH_TOOL_CLASSES)
        finish_tool = _make_finish_tool(agent_def.finish_tool_name, finish_tool_classes)

        user_prompt = resolve_template(stage.user_prompt, slots)
        stage_messages = [
            {"role": "system", "content": agent_def.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = await run_stage(
            stage.name,
            stage_messages,
            agent_def.tools,
            finish_tool,
            session,
            self._working_directory,
            max_iterations=agent_def.max_iterations,
            inject_turn_reminders=agent_def.inject_turn_reminders,
            execution_id=execution_id,
        )

        slots[stage.name] = result
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finish_tool(name: str, finish_tool_classes: dict[str, type[BaseFinishTool]]) -> BaseFinishTool:
    """Instantiate a finish tool by name from the given registry."""
    cls = finish_tool_classes.get(name)
    if cls is None:
        raise ValueError(f"Unknown finish tool '{name}'")
    return cls()


class _WorkflowAbort(Exception):
    """Raised to abort the entire workflow (e.g. compile fix loop exhausted)."""


def _collect_inner_results(
    stage: WorkflowStageDefinition, slots: dict[str, Any], item: Any, success: bool
) -> dict:
    """Snapshot inner stage finish results for one loop iteration."""
    result: dict = {"item": item, "success": success}
    for inner in stage.inner_stages:
        if inner.name in slots:
            result[inner.name] = slots[inner.name]
    return result


def _format_task_summary(aggregated: list[dict]) -> str:
    """Produce a human-readable task summary from loop aggregated results."""
    lines = []
    for entry in aggregated:
        item = entry.get("item") or {}
        task_id = item.get("id", "?") if isinstance(item, dict) else str(item)
        description = item.get("description", "") if isinstance(item, dict) else ""
        status = "done" if entry.get("success") else "failed"
        execute = entry.get("execute_task") or {}
        result_text = execute.get("result", "") if isinstance(execute, dict) else ""
        lines.append(f"- {task_id} [{status}]: {description} → {result_text or 'no result'}")
    return "\n".join(lines)
