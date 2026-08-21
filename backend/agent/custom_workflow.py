import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
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
    load_workflow,
)
from agent.workflow_resume import reconstruct_slots
from agent.workflow_run_recorder import WorkflowRunRecorder

logger = logging.getLogger(__name__)

_WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"
_sub_workflow_cache: dict[str, WorkflowDefinition] = {}


def _load_referenced_workflow(ref: str) -> WorkflowDefinition:
    """Load a workflow by name (flat `<ref>.yaml` or `<ref>/workflow.yaml`), cached by name.

    Same resolution rule ws.py uses to dispatch a top-level workflow, so `type: workflow` stages
    can reference any workflow the slash command palette can. Workflow definitions are immutable
    once parsed, so caching them process-wide is safe.
    """
    if ref not in _sub_workflow_cache:
        flat_path = _WORKFLOWS_DIR / f"{ref}.yaml"
        dir_path = _WORKFLOWS_DIR / ref
        workflow_path = flat_path if flat_path.exists() else dir_path
        _sub_workflow_cache[ref] = load_workflow(workflow_path)
    return _sub_workflow_cache[ref]

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


def resolve_action_input_value(value: Any, slots: dict[str, Any]) -> Any:
    """Resolve one coordinator action_input value, permissively.

    A plain `{{slot.field}}` string resolves to its actual typed value (list/dict/str/...), same
    as resolve_value. A string mixing literal text with `{{...}}` resolves via string
    interpolation. A string with no `{{` at all passes through unchanged as a literal (e.g. a
    fixed script path or CLI flag). Lists and dicts resolve element-wise, so an action input like
    `args: ["--apply", "{{gaps.path}}"]` — needed by run_script to build an argv — works.
    """
    if isinstance(value, list):
        return [resolve_action_input_value(item, slots) for item in value]
    if isinstance(value, dict):
        return {key: resolve_action_input_value(item, slots) for key, item in value.items()}
    if isinstance(value, str):
        match = re.fullmatch(r"\{\{([^}]+)\}\}", value.strip())
        if match is not None:
            return _resolve_path(match.group(1).strip(), slots)
        if "{{" in value:
            return resolve_template(value, slots)
        return value
    return value


_SAFE_EVAL_GLOBALS = {"__builtins__": {}, "len": len, "None": None, "True": True, "False": False}


def evaluate_condition(condition: str, slots: dict[str, Any]) -> bool:
    """Evaluate a Python-like condition expression against the slot registry.

    Slots are accessible by name with dot notation (e.g. plan.compile_command).
    Returns True when condition is empty (unconditional).

    A condition that fails to evaluate at all (undefined slot name, bad syntax, wrong type) is a
    workflow-definition bug, not a runtime maybe-true-maybe-false outcome — it will fail exactly
    the same way on every retry, so silently treating it as False previously meant a loop would
    burn every retry attempt against a check that could never pass. It now raises _WorkflowAbort
    instead, so the whole run stops immediately with a clear error rather than retrying pointlessly.
    """
    if condition is None or condition == "":
        return True
    context = {k: _to_namespace(v) for k, v in slots.items()}
    try:
        return bool(eval(condition, _SAFE_EVAL_GLOBALS, context))  # noqa: S307
    except Exception as exc:
        raise _WorkflowAbort(f"Condition {condition!r} failed to evaluate: {exc}") from exc


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


def _display_stage_type(stage: WorkflowStageDefinition) -> str:
    """The stage type as shown in the run view: 'coordinator' is the engine's real dispatch
    category for every deterministic action, but a run_script stage reads as its own kind of
    thing to a workflow author, not generically "coordinator" alongside enumerate_files or
    chunk_file — so it's relabeled 'script' wherever a stage's type is emitted or persisted. The
    underlying WorkflowStageDefinition.type is untouched; this only affects what's displayed.
    """
    if stage.type == "coordinator" and stage.action == "run_script":
        return "script"
    return stage.type


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
        if stage.type == "workflow" and stage.workflow_ref != "":
            sub_def = _load_referenced_workflow(stage.workflow_ref)
            visible_sub_stages = [s for s in sub_def.stages if s.name not in stage.skip_stages]
            children = _serialize_stage_nodes(visible_sub_stages, f"{path}.")
        else:
            children = _serialize_stage_nodes(stage.inner_stages, f"{path}.")
        nodes.append({
            "path": path,
            "name": stage.name,
            "type": _display_stage_type(stage),
            "finish_tool": stage.finish_tool_name if stage.type == "llm" else None,
            "tools": stage.tools + stage.agents,
            "over": stage.over if stage.over != "" else None,
            "attempt_total": stage.max_retries + 1 if stage.type == "loop" else None,
            "children": children,
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
        # Ancestry of currently-active loop items, outermost first — (loop_path, item_number).
        # Distinct from _loop_context (a single "innermost" slot for live event labels): this is a
        # real stack, shared by reference into any sub-workflow orchestrator, because persisting a
        # stage's on-disk address needs EVERY ancestor loop's item number at once, not just the
        # innermost — see WorkflowRunRecorder.address_for.
        self._active_item_stack: list[tuple[str, int]] = []
        self._recorder: WorkflowRunRecorder | None = None

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

        run_id = uuid.uuid4().hex[:8]
        # Same fallback _run_workflow applies to slots["user_message"] below — persisted here too
        # (rather than the raw user_message) so a later resume reseeds the identical value.
        effective_message = user_message if user_message.strip() != "" else f"/{self._workflow.name}"
        nodes = _serialize_stage_nodes(self._workflow.stages)
        self._recorder = WorkflowRunRecorder(self._workflow.name, run_id, user_message=effective_message, nodes=nodes)
        recording_session = _RecordingSession(session, self._recorder, self._active_item_stack)

        await session.emit({
            "type": "workflow_start",
            "workflow_name": self._workflow.name,
            "run_id": run_id,
            "nodes": nodes,
        })
        try:
            await self._run_workflow(recording_session, user_message, messages)
        except asyncio.CancelledError:
            logger.info("[workflow:%s] stopped by user (CancelledError)", self._workflow.name)
            self._recorder.mark_stopped()
            await session.emit({"type": "stopped"})
        except aiohttp.ClientConnectorError as exc:
            logger.error("[workflow:%s] LLM backend connection error: %s", self._workflow.name, exc)
            self._recorder.mark_failed()
            await session.emit({"type": "error", "message": "LLM backend is not running"})
        except _WorkflowAbort as exc:
            logger.error("[workflow:%s] aborted: %s", self._workflow.name, exc)
            self._recorder.mark_failed()
            await session.emit({"type": "error", "message": str(exc)})
        except Exception as exc:
            import traceback
            full_tb = traceback.format_exc()
            logger.exception("[workflow:%s] unexpected error", self._workflow.name)
            self._recorder.mark_failed()
            await session.emit({"type": "error", "message": full_tb})
        else:
            self._recorder.mark_done()
        finally:
            session.mode = saved_mode
            session.auto_safe_commands = saved_auto_safe_commands
            if self._workflow.mode is not None:
                logger.info("[workflow:%s] mode restored to '%s'", self._workflow.name, saved_mode)

    async def resume(
        self,
        session: AgentSession,
        run_id: str,
        resume_address: list[str],
        messages: list[LLMMessage],
    ) -> None:
        """Resume a previously persisted run in place (same run_id), continuing from
        resume_address — see ADR-0011's "Deferred: resumability".

        Mirrors .run() but reopens the existing run directory (WorkflowRunRecorder.load) instead
        of creating one, and reconstructs the slot registry from disk (workflow_resume.
        reconstruct_slots, which also recovers the original user_message from run.json) instead
        of seeding it empty. Kept as its own method rather than merged into .run()/_run_workflow:
        the two entry points differ in how the recorder and slots get built, and duplicating the
        small mode-save/try-except-finally scaffold around that is simpler than threading a
        conditional through the existing, working .run() path.
        """
        logger.info("[workflow:%s] resuming run %s from %s", self._workflow.name, run_id, resume_address)
        saved_mode = session.mode
        saved_auto_safe_commands = session.auto_safe_commands
        if self._workflow.mode is not None:
            session.mode = self._workflow.mode
            logger.info("[workflow:%s] mode temporarily set to '%s'", self._workflow.name, self._workflow.mode)
        session.auto_safe_commands = list(self._workflow.auto_safe_commands)

        self._recorder = WorkflowRunRecorder.load(self._workflow.name, run_id)
        recording_session = _RecordingSession(session, self._recorder, self._active_item_stack)
        slots = reconstruct_slots(self._recorder._root, self._workflow, resume_address)

        await session.emit({
            "type": "workflow_start",
            "workflow_name": self._workflow.name,
            "run_id": run_id,
            "nodes": _serialize_stage_nodes(self._workflow.stages),
        })
        try:
            await self._run_stage_sequence(recording_session, messages, slots, resume_cursor=resume_address)
            logger.info("[workflow:%s] all stages complete (resumed)", self._workflow.name)
            await session.emit({"type": "done", "finished_without_response": False})
        except asyncio.CancelledError:
            logger.info("[workflow:%s] stopped by user (CancelledError)", self._workflow.name)
            self._recorder.mark_stopped()
            await session.emit({"type": "stopped"})
        except aiohttp.ClientConnectorError as exc:
            logger.error("[workflow:%s] LLM backend connection error: %s", self._workflow.name, exc)
            self._recorder.mark_failed()
            await session.emit({"type": "error", "message": "LLM backend is not running"})
        except _WorkflowAbort as exc:
            logger.error("[workflow:%s] aborted: %s", self._workflow.name, exc)
            self._recorder.mark_failed()
            await session.emit({"type": "error", "message": str(exc)})
        except Exception as exc:
            import traceback
            full_tb = traceback.format_exc()
            logger.exception("[workflow:%s] unexpected error", self._workflow.name)
            self._recorder.mark_failed()
            await session.emit({"type": "error", "message": full_tb})
        else:
            self._recorder.mark_done()
        finally:
            session.mode = saved_mode
            session.auto_safe_commands = saved_auto_safe_commands
            if self._workflow.mode is not None:
                logger.info("[workflow:%s] mode restored to '%s'", self._workflow.name, saved_mode)

    async def _run_workflow(
        self, session: AgentSession, user_message: str, messages: list[LLMMessage]
    ) -> None:
        """Top-level entry: seed the slot registry from the user's message and run every stage."""
        effective_message = user_message if user_message.strip() != "" else f"/{self._workflow.name}"
        slots: dict[str, Any] = {"user_message": effective_message}
        await self._run_stage_sequence(session, messages, slots)
        logger.info("[workflow:%s] all stages complete", self._workflow.name)
        await session.emit({"type": "done", "finished_without_response": False})

    async def _run_stage_sequence(
        self,
        session: AgentSession,
        messages: list[LLMMessage],
        slots: dict[str, Any],
        skip_stage_names: frozenset[str] = frozenset(),
        path_prefix: str = "",
        outcome_sink: dict[str, Any] | None = None,
        resume_cursor: list[str] | None = None,
    ) -> None:
        """Drive this orchestrator's own stages (`self._workflow.stages`) to completion.

        Shared by the top-level run and by a `type: workflow` sub-invocation (which constructs a
        fresh orchestrator over the referenced definition and calls this on it directly, bypassing
        `.run()`'s workflow_start/done events since those belong to the top-level run only).
        `skip_stage_names` drops stages the caller already seeded via sub_workflow_input.

        `outcome_sink`, when given, is filled with `{stage.name: outcome.result}` for every stage
        dispatched here — each stage's own compact, display-ready result, as opposed to the full
        slot registry. Used by `_run_sub_workflow` so a sub-invocation can report a small summary
        of what it did instead of exposing its entire internal state to the caller.

        `resume_cursor`, when given, is an on-disk address (see WorkflowRunRecorder.address_for)
        naming where to start: everything before it in this stage list is assumed already
        reconstructed into `slots` by workflow_resume.reconstruct_slots, so dispatch starts at the
        named stage instead of index 0. Only that one stage receives the remainder of the cursor
        (for a loop, which segment of its own item/inner-stage tree to resume into) — every stage
        dispatched after it runs as an ordinary fresh dispatch.
        """
        stages = [s for s in self._workflow.stages if s.name not in skip_stage_names]
        stage_index = {s.name: i for i, s in enumerate(stages)}
        logger.info("[workflow:%s] %d stages: %s", self._workflow.name, len(stages), [s.name for s in stages])

        if resume_cursor is not None:
            current = stage_index[resume_cursor[0]]
            pending_cursor: list[str] | None = resume_cursor[1:]
        else:
            current = 0
            pending_cursor = None

        while current < len(stages):
            stage = stages[current]
            dispatch_cursor = pending_cursor
            pending_cursor = None  # only the first stage dispatched here (the resume target) gets a cursor
            logger.info("[workflow:%s] dispatching stage '%s' (type=%s)", self._workflow.name, stage.name, stage.type)
            outcome = await self._dispatch(stage, session, messages, slots, path_prefix=path_prefix, resume_cursor=dispatch_cursor)
            if outcome_sink is not None:
                outcome_sink[stage.name] = outcome.result
            logger.info("[workflow:%s] stage '%s' done — slots keys: %s", self._workflow.name, stage.name, list(slots.keys()))
            if outcome.jump_to is not None:
                jump_to = outcome.jump_to
                if jump_to not in stage_index:
                    raise ValueError(f"Branch target '{jump_to}' not found in workflow stages")
                logger.info("[workflow:%s] jumping to stage '%s'", self._workflow.name, jump_to)
                current = stage_index[jump_to]
            else:
                current += 1

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
            "stage_type": _display_stage_type(stage),
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
        """Announce that a stage invocation ended. started_at is None for stages that never ran.

        `result` is emitted in full, untruncated — the point of persisting it (ADR-0011) is to
        keep what the live view can't afford to. Truncation for the websocket happens downstream,
        in _RecordingSession.emit, on the copy actually forwarded to the real session — never on
        what the recorder writes to disk.
        """
        duration_ms = 0 if started_at is None else int((time.monotonic() - started_at) * 1000)
        await session.emit({
            "type": "stage_exit",
            "path": path,
            "execution_id": execution_id,
            "status": status,
            "result": result,
            "duration_ms": duration_ms,
        })

    async def _dispatch(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[LLMMessage],
        slots: dict[str, Any],
        path_prefix: str = "",
        resume_cursor: list[str] | None = None,
    ) -> StageOutcome:
        """Run one stage, bracketed by stage_enter / stage_exit events for the run view.

        Single choke point for the skip condition, the invocation counter and the timing, so every
        stage type reports the same lifecycle no matter which runner handles it. A runner that
        raises still reports stage_exit(failed) before the exception propagates — _run_loop relies
        on catching it to retry the item.

        `resume_cursor` is only meaningful for a loop stage (see _run_loop) — every other stage
        type is atomic from a resume's point of view and ignores it.
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
            outcome = await self._run_by_type(stage, session, messages, slots, path, execution_id, resume_cursor)
        except asyncio.CancelledError:
            # CancelledError is a BaseException, not an Exception — without this branch it skips
            # the exit-emit entirely, leaving this invocation's meta.json/live row stuck at
            # "running" forever (the run view keeps it spinning even though nothing is happening).
            await self._emit_stage_exit(session, path, execution_id, "stopped", None, started_at)
            raise
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
        resume_cursor: list[str] | None = None,
    ) -> StageOutcome:
        """Route a stage to the runner for its type and return what it produced."""
        if stage.type == "llm":
            return StageOutcome(result=await self._run_llm(stage, session, slots, execution_id))
        if stage.type == "coordinator":
            return StageOutcome(result=await self._run_coordinator(stage, session, slots))
        if stage.type == "branch":
            return self._resolve_branch(stage, slots)
        if stage.type == "loop":
            return StageOutcome(result=await self._run_loop(stage, session, messages, slots, path, resume_cursor))
        if stage.type == "respond":
            await self._run_respond(stage, session, messages, slots, execution_id)
            return StageOutcome()
        if stage.type == "agent":
            return StageOutcome(result=await self._run_isolated_agent(stage, session, slots, execution_id))
        if stage.type == "workflow":
            return StageOutcome(result=await self._run_sub_workflow(stage, session, messages, slots, path))
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
        resolved_inputs = {k: resolve_action_input_value(v, slots) for k, v in stage.action_input.items()}
        output = await run_coordinator_action(
            stage.action, resolved_inputs, session, self._working_directory, self._workflow.directory
        )

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
        resume_cursor: list[str] | None = None,
    ) -> dict:
        """Run a loop stage: iterate over a list or repeat until exit_condition.

        Returns a compact tally rather than the aggregated per-item results: those are already
        stored in the loop_output slot, and streaming them again as a stage result would repeat
        every item's full output over the event stream.

        `resume_cursor` is non-None only when THIS loop is on the resume path (it was threaded
        down because this loop is the stage `resume_cursor` in _run_stage_sequence/an enclosing
        loop's own resume target). An empty list means "resume at this loop itself" (start fresh
        at item 1); a longer list's first segment is the item number to resume into, with any
        remainder addressing an inner stage within that item. Items strictly before the resume
        item are read back from their persisted item_result.json rather than redispatched — their
        slot writes were already replayed by workflow_resume.reconstruct_slots before this run
        started. Items after it are also read from disk when stage.assume_independent and a
        result exists there; otherwise every item dispatches live, same as an ordinary run.
        `resume_cursor is None` (this loop simply isn't on the resume path — either an ordinary
        run, or this loop comes after the resume point in its own stage sequence) always
        dispatches every item fresh, with no disk reuse at all: "everything after the resume
        point" must be redone in full, never partially reused.
        """
        # self._loop_context is a single shared attribute, not a stack — when this loop is nested
        # inside another loop's per-item dispatch, entering here sees the OUTER loop's current
        # context. Restoring it (rather than hardcoding None) after each of THIS loop's items is
        # what lets a stage dispatched after a nested loop still carry the outer loop's item
        # number, and stops this loop's own item numbers (e.g. a chunk index) from leaking into
        # sibling stages once this loop has finished.
        enclosing_loop_context = self._loop_context

        items: list[Any]
        if stage.over != "":
            items = resolve_value(stage.over, slots) or []
        else:
            items = [None]  # single-item sentinel for non-list loops

        if resume_cursor is None:
            resume_item_number = None
            loop_address = None
        elif len(resume_cursor) == 0:
            resume_item_number = 1
            loop_address = None
        else:
            resume_item_number = int(resume_cursor[0])
            loop_address = session._recorder.address_for(path, self._active_item_stack)

        aggregated: list[dict] = []
        item_total = len(items)
        attempt_total = stage.max_retries + 1
        succeeded_count = 0
        failed_count = 0
        logger.info("[workflow] loop '%s' — %d item(s), up to %d attempt(s) each", stage.name, item_total, attempt_total)

        for item_index, item in enumerate(items):
            item_number = item_index + 1

            if loop_address is not None and item_number != resume_item_number:
                is_before_resume = item_number < resume_item_number
                if is_before_resume or stage.assume_independent:
                    item_payload = session._recorder.read_item_result(loop_address, item_number)
                    if item_payload is not None:
                        aggregated.append(item_payload["item_result"])
                        if item_payload["success"]:
                            succeeded_count += 1
                        else:
                            failed_count += 1
                        continue
                    # Nothing persisted for this item — either the original run never reached it
                    # (item strictly before the resume point but somehow not on disk) or nothing
                    # to reuse for an "after" item — fall through and dispatch it fresh below.

            item_inner_cursor: list[str] | None = None
            if resume_cursor is not None and item_number == resume_item_number and len(resume_cursor) > 1:
                item_inner_cursor = resume_cursor[1:]

            if stage.over != "":
                slots[stage.item_var] = item
            # Reset on_retry slots to their default (empty string) before first attempt
            for slot_name in stage.on_retry:
                slots[slot_name] = ""
            # Clear each inner stage's own output slot before this item's first attempt. Slots
            # are a single shared dict keyed by stage name across the whole run — if an inner
            # stage never successfully completes for this item (it raised, or the item exhausted
            # its retries before reaching that stage), its slot would otherwise still hold
            # whatever the PREVIOUS item last wrote there, and _collect_inner_results would
            # silently report that stale value as if it belonged to this item.
            #
            # Skipped when resuming partway into this item's own inner stages: those slots were
            # just populated from disk by workflow_resume.reconstruct_slots, and this item's own
            # dispatch below starts at item_inner_cursor, not at its first inner stage.
            if item_inner_cursor is None:
                for inner in stage.inner_stages:
                    slots.pop(inner.name, None)

            item_success = False
            # Set when the item's dispatch is cancelled (stop button) rather than merely failing —
            # the item must still be closed out below (its dot stops spinning) but must not be
            # retried or treated as an on_max_retries failure; the original CancelledError is
            # re-raised once the item is closed out, so it keeps propagating up to abort the loop
            # and the whole workflow, same as it would have without this loop catching it at all.
            cancelled_exc: asyncio.CancelledError | None = None
            # Pushed for the whole item (every attempt), not per-attempt — a retry keeps the same
            # item_number, and this stack exists so WorkflowRunRecorder can address a nested stage
            # on disk by its full item ancestry (see _active_item_stack's docstring in __init__).
            # try/finally guarantees the pop happens even when _WorkflowAbort escapes the item.
            self._active_item_stack.append((path, item_number))
            try:
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

                    # A resumed item's inner_cursor only applies to its very first (re-entered)
                    # attempt — a subsequent retry of the same item redoes every inner stage from
                    # the start, same as an ordinary (non-resumed) retry always has.
                    if attempt == 0 and item_inner_cursor is not None:
                        start_index = next(
                            index for index, inner in enumerate(stage.inner_stages)
                            if inner.name == item_inner_cursor[0]
                        )
                        inner_stages_this_attempt = stage.inner_stages[start_index:]
                        first_inner_cursor = item_inner_cursor[1:]
                    else:
                        inner_stages_this_attempt = stage.inner_stages
                        first_inner_cursor = None

                    try:
                        for position, inner in enumerate(inner_stages_this_attempt):
                            cursor = first_inner_cursor if position == 0 else None
                            await self._dispatch(inner, session, messages, slots, path_prefix=f"{path}.", resume_cursor=cursor)
                    except _WorkflowAbort:
                        # A condition (skip-condition or exit_condition) that failed to evaluate at
                        # all is a workflow-definition bug, not a per-attempt failure — it will fail
                        # the exact same way on every retry, so let it escape and abort the whole
                        # run instead of burning every remaining attempt against a check that can
                        # never pass.
                        raise
                    except asyncio.CancelledError as exc:
                        # The inner _dispatch already emitted its own stage_exit(stopped) for
                        # whichever inner stage was actually in flight — this item as a whole still
                        # needs its own loop_item_exit below so its dot stops spinning too. Stop
                        # attempting (not a retry-then-continue case) and fall out of the attempt
                        # loop to close the item out.
                        cancelled_exc = exc
                        break
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
            finally:
                self._active_item_stack.pop()

            attempts_used = attempt + 1
            self._loop_context = enclosing_loop_context

            # Snapshot exactly what this item produced — every inner stage's result, keyed by
            # inner stage name — the same data _collect_inner_results feeds into the loop's own
            # aggregated output slot. Sent here too so the run view can show a specific item's
            # real result unambiguously (keyed by loop path + item_number, no guessing which
            # execution_id "belongs" to this item), instead of trying to reconstruct it from the
            # separate stage_enter/stage_exit stream after the fact.
            item_result = _collect_inner_results(stage, slots, item, item_success) if stage.over != "" else None

            # "status" is the item's own real terminal state — distinct from "success" (a plain
            # bool the aggregate/resume logic still keys off) because a stopped item must render
            # differently from a genuinely failed one, same as stage_exit's status string.
            item_status = "stopped" if cancelled_exc is not None else ("done" if item_success else "failed")

            # item_result is emitted in full here too, same reasoning as _emit_stage_exit's result —
            # truncated only for the copy _RecordingSession forwards to the real session.
            await session.emit({
                "type": "loop_item_exit",
                "path": path,
                "item_number": item_number,
                "item_total": item_total,
                "success": item_success,
                "status": item_status,
                "attempts_used": attempts_used,
                "item_result": item_result,
            })

            if cancelled_exc is not None:
                # The item is closed out above; the loop (and the whole workflow) is being torn
                # down, so — same as any other uncaught exception escaping this loop — the
                # remaining items and the aggregate write below are skipped.
                raise cancelled_exc

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
        execution_id: str,
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
        respond_session = _RespondStageSession(session, execution_id)
        await run_agent(respond_session, working_messages, ToolSet(tools=self._tools, extra_tools=None), self._working_directory)
        if respond_session.error_message is not None:
            logger.warning("[workflow] respond: run_agent errored: %s", respond_session.error_message)
        elif respond_session.finished_without_response:
            logger.warning("[workflow] respond: run_agent finished without a response")
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

    async def _run_sub_workflow(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[LLMMessage],
        slots: dict[str, Any],
        path: str,
    ) -> dict:
        """Run another workflow definition as an isolated sub-stage.

        The sub-workflow gets its own fresh slot registry — seeded only from
        sub_workflow_input — rather than sharing the caller's, so its internal slot names
        (e.g. `chunks`, `parse_request`) can never collide with the caller's. `skip_stages`
        bypasses stages whose slot was already seeded this way (e.g. an extraction stage the
        caller has no use for because it already knows the values).

        _invocation_counters is shared by reference with the caller: without that, a fresh
        orchestrator's counters would restart at 0 on every call (e.g. every item of an outer
        loop), so the same nested stage path would get execution_id '...#1' every time and the
        run view would treat separate invocations as the same one.
        """
        sub_def = _load_referenced_workflow(stage.workflow_ref)
        sub_slots = _seed_sub_slots(stage.sub_workflow_input, slots)
        sub_orchestrator = CustomWorkflowOrchestrator(sub_def, self._working_directory, self._tools)
        sub_orchestrator._invocation_counters = self._invocation_counters
        sub_orchestrator._active_item_stack = self._active_item_stack

        saved_mode = session.mode
        saved_auto_safe_commands = session.auto_safe_commands
        if sub_def.mode is not None:
            session.mode = sub_def.mode
        session.auto_safe_commands = list(sub_def.auto_safe_commands)
        outcomes: dict[str, Any] = {}
        try:
            await sub_orchestrator._run_stage_sequence(
                session, messages, sub_slots,
                skip_stage_names=frozenset(stage.skip_stages),
                path_prefix=f"{path}.",
                outcome_sink=outcomes,
            )
        finally:
            session.mode = saved_mode
            session.auto_safe_commands = saved_auto_safe_commands

        # Report only each dispatched stage's own compact loop tally (item_total/succeeded/failed),
        # never the sub-workflow's full slot registry. A sub-workflow like translate-locale carries
        # every chunk's raw source and translated text in its internal slots — exposing all of that
        # to the caller (e.g. into a directory-sync loop summary, then a synthesis prompt) can
        # trivially overflow the model's context. The caller only needs to know what happened, not
        # replay the sub-workflow's own working state.
        summary = {
            name: result for name, result in outcomes.items()
            if isinstance(result, dict) and {"item_total", "succeeded", "failed"} <= result.keys()
        }
        slots[stage.name] = summary
        return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_sub_slots(input_map: dict[str, str], parent_slots: dict[str, Any]) -> dict[str, Any]:
    """Build a fresh isolated slot registry for a sub-workflow from dotted paths -> resolved values.

    E.g. {"parse_request.output_path": "{{gaps.translated_path}}"} produces
    {"parse_request": {"output_path": <value>}} so the sub-workflow's own templates
    ({{parse_request.output_path}}) resolve exactly as if that stage had produced it itself.
    """
    seeded: dict[str, Any] = {}
    for dotted_path, expr in input_map.items():
        value = resolve_action_input_value(expr, parent_slots)
        parts = dotted_path.split(".")
        target = seeded
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return seeded

def _make_finish_tool(name: str, finish_tool_classes: dict[str, type[BaseFinishTool]]) -> BaseFinishTool:
    """Instantiate a finish tool by name from the given registry."""
    cls = finish_tool_classes.get(name)
    if cls is None:
        raise ValueError(f"Unknown finish tool '{name}'")
    return cls()


class _WorkflowAbort(Exception):
    """Raised to abort the entire workflow (e.g. compile fix loop exhausted)."""


class _RecordingSession:
    """Session proxy that persists stage lifecycle and activity events to WorkflowRunRecorder as
    they pass through, then forwards to the real session — see ADR-0011.

    Constructed once in CustomWorkflowOrchestrator.run() and used as the `session` argument passed
    into _run_workflow, so it sits on the single funnel every event already flows through for the
    whole call tree: run_stage's per-iteration forwarding (pipeline.py), a sub-workflow's own
    dispatch (the same session object is threaded through unchanged in _run_sub_workflow), and
    _RespondStageSession's forwarding all call .emit() on whatever session they were handed, which
    is this proxy (or something wrapping it) for the entire life of one run.

    stage_exit/loop_item_exit reach the recorder with their full, untruncated result/item_result
    — that completeness is the entire point of persisting them (ADR-0011) — but the copy actually
    forwarded to the real session (and from there, the websocket) still gets _truncate_stage_result
    applied, same as before this recorder existed (ADR-0009): the live view was never meant to
    carry a huge payload verbatim, only the persisted copy needed to stop being lossy.
    """

    def __init__(
        self,
        real_session: AgentSession,
        recorder: WorkflowRunRecorder,
        active_item_stack: list[tuple[str, int]],
    ) -> None:
        self._real = real_session
        self._recorder = recorder
        self._active_item_stack = active_item_stack

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def emit(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "stage_enter":
            self._recorder.on_stage_enter(event, self._active_item_stack)
        elif event_type == "stage_exit":
            self._recorder.on_stage_exit(event, self._active_item_stack)
            event = {**event, "result": _truncate_stage_result(event["result"])}
        elif event_type == "loop_item_exit":
            self._recorder.on_loop_item_exit(event, self._active_item_stack)
            event = {**event, "item_result": _truncate_stage_result(event["item_result"])}
        elif event.get("_workflow_execution") is not None:
            self._recorder.on_activity(event["_workflow_execution"], event)
        await self._real.emit(event)


class _RespondStageSession:
    """Session proxy wrapped around a respond stage's run_agent call.

    run_agent (agent.py) always emits its own {"type": "done"/"error"} event when it finishes.
    For a respond stage that call is just one stage inside a larger workflow, not the whole run —
    but nothing distinguishes that event from the workflow's real completion. Both the websocket
    relay (ws.py) and the run view (workflow-run.service.ts) treat any untagged done/error as
    terminal for the entire run: left unhandled, they stop and cancel the orchestrator right there,
    before this stage's own stage_exit — and the workflow's actual final done — ever get sent.
    Swallowing it here lets the orchestrator report the real outcome once every stage has finished.
    Everything else (content, thinking, tool calls...) is forwarded untouched, since that output is
    meant to stream straight into the conversation.
    """

    def __init__(self, real_session: AgentSession, execution_id: str) -> None:
        self._real = real_session
        self._execution_id = execution_id
        self.finished_without_response = False
        self.error_message: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def emit(self, event: dict) -> None:
        if event.get("type") == "done":
            self.finished_without_response = event.get("finished_without_response", False)
            return
        if event.get("type") == "error":
            self.error_message = event.get("message")
            return
        # A respond stage's own thinking/content/tool-call events come from run_agent directly,
        # not run_stage — so unlike an llm/agent stage, nothing has stamped _workflow_execution on
        # them yet. Stamp it here so WorkflowRunRecorder (via _RecordingSession, further down this
        # session chain) can persist this stage's transcript the same as any other stage type.
        if event.get("_workflow_execution") is None:
            event = {**event, "_workflow_execution": self._execution_id}
        await self._real.emit(event)


def _collect_inner_results(
    stage: WorkflowStageDefinition, slots: dict[str, Any], item: Any, success: bool
) -> dict:
    """Snapshot inner stage finish results for one loop iteration.

    Skips any inner stage with include_in_item_result=False — scratch data meant only for other
    inner stages in the same iteration (e.g. a chunking pass consumed by a per-chunk summarize
    loop), not for "what this item produced" as seen by the loop's own aggregated output slot or
    the loop_item_exit event the run view reads.
    """
    result: dict = {"item": item, "success": success}
    for inner in stage.inner_stages:
        if inner.include_in_item_result and inner.name in slots:
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
