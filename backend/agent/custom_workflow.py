from __future__ import annotations

import asyncio
import json
import logging
import re
from types import SimpleNamespace
from typing import Any

import aiohttp

from agent.agent import AgentSession, run_agent
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

class CustomWorkflowOrchestrator:
    """Drives a YAML-defined workflow through typed stages with a shared slot registry.

    Slot registry accumulates finish-tool results and coordinator outputs so later
    stages can reference earlier ones via {{slot.field}} in prompts and conditions.
    """

    def __init__(self, workflow: WorkflowDefinition, working_directory: str | None, tools: list[dict] | None = None):
        self._workflow = workflow
        self._working_directory = working_directory
        self._tools = tools or []

    async def run(self, session: AgentSession, user_message: str, messages: list[dict]) -> None:
        """Entry point — runs all workflow stages and emits events via session."""
        try:
            await self._run_workflow(session, user_message, messages)
        except asyncio.CancelledError:
            await session.emit({"type": "error", "message": f"Workflow '{self._workflow.name}' was aborted"})
        except aiohttp.ClientConnectorError as exc:
            logger.error("[workflow:%s] LLM backend connection error: %s", self._workflow.name, exc)
            await session.emit({"type": "error", "message": "LLM backend is not running"})
        except _WorkflowAbort as exc:
            await session.emit({"type": "error", "message": str(exc)})
        except Exception as exc:
            logger.exception("[workflow:%s] unexpected error", self._workflow.name)
            await session.emit({"type": "error", "message": str(exc)})

    async def _run_workflow(
        self, session: AgentSession, user_message: str, messages: list[dict]
    ) -> None:
        slots: dict[str, Any] = {"user_message": user_message}
        stages = self._workflow.stages
        stage_index = {s.name: i for i, s in enumerate(stages)}
        current = 0

        while current < len(stages):
            stage = stages[current]
            jump_to = await self._dispatch(stage, session, messages, slots)
            if jump_to is not None:
                if jump_to not in stage_index:
                    raise ValueError(f"Branch target '{jump_to}' not found in workflow stages")
                current = stage_index[jump_to]
            else:
                current += 1

        await session.emit({"type": "done", "finished_without_response": False})

    async def _dispatch(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[dict],
        slots: dict[str, Any],
    ) -> str | None:
        """Run one stage. Returns a jump target name for branch stages, None otherwise."""
        if stage.type == "llm":
            await self._run_llm(stage, session, slots)
        elif stage.type == "coordinator":
            await self._run_coordinator(stage, session, slots)
        elif stage.type == "branch":
            return self._resolve_branch(stage, slots)
        elif stage.type == "loop":
            await self._run_loop(stage, session, messages, slots)
        elif stage.type == "respond":
            await self._run_respond(stage, session, messages, slots)
        elif stage.type == "agent":
            await self._run_isolated_agent(stage, session, slots)
        else:
            raise ValueError(f"Unknown stage type '{stage.type}'")
        return None

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_llm(
        self, stage: WorkflowStageDefinition, session: AgentSession, slots: dict[str, Any]
    ) -> None:
        """Run an LLM stage: resolve prompts, run the loop, store finish result."""
        if not evaluate_condition(stage.condition, slots):
            logger.info("[workflow] skipping llm stage '%s' (condition false)", stage.name)
            return

        system_prompt = resolve_template(stage.system_prompt, slots)
        user_prompt = resolve_template(stage.user_prompt, slots)
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
        )

        slots[stage.name] = result
        await session.emit({
            "type": "pipeline_summary",
            "label": f"stage: {stage.name}",
            "content": json.dumps(result, ensure_ascii=False, indent=2),
            "notes": None,
        })

    async def _run_coordinator(
        self, stage: WorkflowStageDefinition, session: AgentSession, slots: dict[str, Any]
    ) -> None:
        """Run a coordinator action: resolve inputs, execute action, store output."""
        if not evaluate_condition(stage.condition, slots):
            logger.info("[workflow] skipping coordinator stage '%s' (condition false)", stage.name)
            return

        resolved_inputs = {k: resolve_value(v, slots) for k, v in stage.action_input.items()}
        output = await run_coordinator_action(stage.action, resolved_inputs, session, self._working_directory)

        if stage.action_output != "":
            slots[stage.action_output] = output
        logger.info("[workflow] coordinator '%s' action='%s' output_slot='%s'", stage.name, stage.action, stage.action_output)

    def _resolve_branch(self, stage: WorkflowStageDefinition, slots: dict[str, Any]) -> str:
        """Evaluate a branch condition and return the target stage name."""
        if evaluate_condition(stage.condition, slots):
            logger.info("[workflow] branch '%s' → %s (true)", stage.name, stage.if_true)
            return stage.if_true
        logger.info("[workflow] branch '%s' → %s (false)", stage.name, stage.if_false)
        return stage.if_false

    async def _run_loop(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[dict],
        slots: dict[str, Any],
    ) -> None:
        """Run a loop stage: iterate over a list or repeat until exit_condition."""
        if not evaluate_condition(stage.entry_condition, slots):
            logger.info("[workflow] skipping loop '%s' (entry_condition false)", stage.name)
            return

        items: list[Any]
        if stage.over != "":
            items = resolve_value(stage.over, slots) or []
        else:
            items = [None]  # single-item sentinel for non-list loops

        aggregated: list[dict] = []

        for item in items:
            if stage.over != "":
                slots[stage.item_var] = item
            # Reset on_retry slots to their default (empty string) before first attempt
            for slot_name in stage.on_retry:
                slots[slot_name] = ""

            item_success = False
            for attempt in range(stage.max_retries + 1):
                if attempt > 0:
                    logger.info("[workflow] loop '%s' retry %d/%d", stage.name, attempt, stage.max_retries)
                    for slot_name, template in stage.on_retry.items():
                        slots[slot_name] = resolve_template(template, slots)

                for inner in stage.inner_stages:
                    await self._dispatch(inner, session, messages, slots)

                if evaluate_condition(stage.exit_condition, slots):
                    item_success = True
                    break

            if not item_success:
                if stage.on_max_retries == "abort_workflow":
                    raise _WorkflowAbort(f"Loop '{stage.name}' exhausted {stage.max_retries} retries — aborting workflow")
                logger.warning("[workflow] loop '%s' max retries exhausted for item, continuing", stage.name)

            if stage.over != "":
                aggregated.append(_collect_inner_results(stage, slots, item, item_success))

        if stage.loop_output != "" and stage.over != "":
            task_summary = _format_task_summary(aggregated)
            slots[stage.loop_output] = {"items": aggregated, "task_summary": task_summary}

    async def _run_respond(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        messages: list[dict],
        slots: dict[str, Any],
    ) -> None:
        """Run the plain agent loop on the full conversation history.

        Unlike isolated stages, this uses the original DB conversation so the model
        can reference everything the user said before the workflow ran. The suffix
        injects workflow results into the last user message before responding.
        """
        working_messages = list(messages)
        if stage.message_suffix != "" and working_messages and working_messages[-1].get("role") == "user":
            suffix = resolve_template(stage.message_suffix, slots)
            last = working_messages[-1]
            working_messages[-1] = {**last, "content": last["content"] + "\n\n" + suffix}

        await run_agent(session, working_messages, self._tools, self._working_directory)

    async def _run_isolated_agent(
        self,
        stage: WorkflowStageDefinition,
        session: AgentSession,
        slots: dict[str, Any],
    ) -> None:
        """Run a named agent from agents/ as an isolated stage. Result stored in slots."""
        from agent.workflow_loader import load_agent
        from pathlib import Path

        agents_dir = Path(__file__).parent.parent / "agents"
        agent_def = load_agent(agents_dir / f"{stage.workflow_ref}.yaml")
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
        )

        slots[stage.name] = result
        await session.emit({
            "type": "pipeline_summary",
            "label": f"agent: {stage.workflow_ref}",
            "content": json.dumps(result, ensure_ascii=False, indent=2),
            "notes": None,
        })


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
