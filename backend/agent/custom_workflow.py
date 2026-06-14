from __future__ import annotations

import asyncio
import json
import logging

import aiohttp

from agent.agent import AgentSession
from agent.pipeline import run_stage
from agent.workflow_loader import WorkflowDefinition

logger = logging.getLogger(__name__)


class CustomWorkflowOrchestrator:
    """Runs a YAML-defined workflow by executing each stage in sequence.

    Accumulated finish results from each stage are injected as context into
    subsequent stages' user messages, allowing later stages to reference
    what earlier stages produced.
    """

    def __init__(self, workflow: WorkflowDefinition, working_directory: str | None):
        self._workflow = workflow
        self._working_directory = working_directory

    async def run(self, session: AgentSession, user_message: str, messages: list[dict]) -> None:
        """Entry point — runs all workflow stages and emits events via session."""
        try:
            await self._run_workflow(session, user_message)
        except asyncio.CancelledError:
            await session.emit({"type": "error", "message": f"Workflow '{self._workflow.name}' was aborted"})
        except aiohttp.ClientConnectorError as e:
            logger.error("[workflow:%s] LLM backend connection error: %s", self._workflow.name, e)
            await session.emit({"type": "error", "message": "LLM backend is not running"})
        except Exception as e:
            logger.exception("[workflow:%s] unexpected error", self._workflow.name)
            await session.emit({"type": "error", "message": str(e)})

    async def _run_workflow(self, session: AgentSession, user_message: str) -> None:
        accumulated_results: dict[str, dict] = {}

        for stage in self._workflow.stages:
            stage_user_message = _build_stage_user_message(user_message, accumulated_results)
            stage_messages = [
                {"role": "system", "content": stage.prompt},
                {"role": "user", "content": stage_user_message},
            ]
            finish_tool = self._workflow.make_finish_tool(stage.finish_tool_name)

            result = await run_stage(
                stage.name,
                stage_messages,
                stage.tools,
                finish_tool,
                session,
                self._working_directory,
                max_iterations=stage.max_iterations,
                inject_turn_reminders=stage.inject_turn_reminders,
            )

            accumulated_results[stage.name] = result
            await session.emit({
                "type": "pipeline_summary",
                "label": f"stage: {stage.name}",
                "content": json.dumps(result, ensure_ascii=False, indent=2),
                "notes": None,
            })

        await session.emit({"type": "done", "finished_without_response": False})


def _build_stage_user_message(user_message: str, accumulated_results: dict[str, dict]) -> str:
    """Append prior stage results as a JSON context block to the user message."""
    if not accumulated_results:
        return user_message
    context_parts = ["[Prior stage results]"]
    for stage_name, result in accumulated_results.items():
        context_parts.append(f"\n--- {stage_name} ---")
        context_parts.append(json.dumps(result, ensure_ascii=False, indent=2))
    return user_message + "\n\n" + "\n".join(context_parts)
