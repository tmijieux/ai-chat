import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import aiohttp

from agent.agent import AgentSession, chat_with_tools, run_agent
from agent.finish_tools import (
    FinishAugmentation,
    FinishClassify,
    FinishCritique,
    FinishPlan,
    FinishTask,
    FinishVerify,
)
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list

logger = logging.getLogger(__name__)

MAX_TASK_RETRIES = 2
MAX_STAGE_ITERATIONS = 12


@dataclass
class PipelineTask:
    """A single task produced by the plan stage."""

    id: str
    description: str
    verification_method: str
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"
    result: str | None = None
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Stage system prompts
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
You are a request classifier. Classify the user request as simple or complex.
Simple: greetings, factual questions, explanations that need no file access.
Complex: tasks requiring reading/writing files, multi-step code changes, or research across a codebase.
You MUST call finish_classify to submit your answer. Do not write a text response — the only valid output is the tool call.\
"""

_AUGMENT_SYSTEM = """\
You are a prompt augmentation agent. Enrich the user's request with concrete context from the codebase.
Steps:
1. Use read_file, glob_files, grep_files, list_directory to locate relevant files and understand the code structure.
2. Once you have enough context, call finish_augmentation with the enriched prompt.
You MUST call finish_augmentation to complete your task. Do not write a text summary — call the tool instead.\
"""

_CRITIQUE_SYSTEM = """\
You are a critical reviewer. You receive an original request and an augmented version.
Check: does the augmented prompt correctly capture the intent? Do the file paths and function names actually exist?
You MUST call finish_critique to submit your verdict. Do not write a text response — call the tool instead.\
"""

_PLAN_SYSTEM = """\
You are a task planner. Break the request into a minimal ordered list of atomic tasks.
Each task must be small enough to complete in one agent run and have a concrete verification method.
You MUST call finish_plan to submit the task list. Do not write a text response — call the tool instead.\
"""

_VERIFY_SYSTEM = """\
You are a verification agent. Check whether a task was completed correctly using the provided verification method.
Use read_file, glob_files, and run_shell to inspect the actual state of the code or filesystem.
You MUST call finish_verify to submit your verdict. Do not write a text response — call the tool instead.\
"""


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _finish_tool_schema(tool) -> dict:
    """Convert a finish tool instance to the Ollama function schema dict."""
    return {"type": "function", "function": tool.to_ollama_schema()}


async def _run_stage_loop(
    sub_session: AgentSession,
    stage_messages: list[dict],
    tool_schemas: list[dict],
    extra_tools: dict,
    working_directory: str | None,
    stage_name: str,
) -> None:
    """Inner loop for a pipeline stage. Stops when the finish tool is called or no more tool calls."""
    for iteration in range(MAX_STAGE_ITERATIONS):
        done, _ = await chat_with_tools(
            stage_messages, sub_session, tool_schemas, working_directory, extra_tools=extra_tools
        )
        if sub_session.finish_result is not None:
            break
        if done:
            logger.warning(
                "[pipeline:%s] model stopped at iteration %d without calling finish tool",
                stage_name,
                iteration,
            )
            break
    await sub_session.outbound.put({"type": "_stage_done"})


async def run_stage(
    stage_name: str,
    messages: list[dict],
    regular_tool_names: list[str],
    finish_tool,
    parent_session: AgentSession,
    working_directory: str | None,
) -> dict:
    """
    Run a single pipeline stage. Loops until the finish tool is called.
    Forwards all events to parent_session tagged with _pipeline_stage.
    Returns the finish tool's args dict, or empty dict if it was never called.
    """
    sub_session = AgentSession()
    stage_messages = list(messages)
    regular_schemas = get_ollama_tool_list(regular_tool_names)
    all_schemas = regular_schemas + [_finish_tool_schema(finish_tool)]
    extra_tools = {finish_tool.name: finish_tool}

    logger.info("[pipeline] starting stage: %s", stage_name)

    loop_task = asyncio.create_task(
        _run_stage_loop(sub_session, stage_messages, all_schemas, extra_tools, working_directory, stage_name)
    )

    while True:
        event = await sub_session.outbound.get()
        if event["type"] == "_stage_done":
            break
        await parent_session.emit({**event, "_pipeline_stage": stage_name})

    await loop_task

    result = sub_session.finish_result or {}
    logger.info("[pipeline] stage %s done — result keys: %s", stage_name, list(result.keys()))
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """
    Python-orchestrated multi-stage pipeline.
    Parallel alternative to run_agent — the existing agent loop is unchanged.
    """

    def __init__(
        self,
        system_messages: list[dict],
        working_directory: str | None,
        regular_tools: list[dict],
    ):
        self._system_messages = system_messages
        self._working_directory = working_directory
        self._regular_tools = regular_tools

    async def run(
        self,
        session: AgentSession,
        user_message: str,
        messages: list[dict],
    ) -> None:
        """Entry point. Runs the full pipeline and emits events via session."""
        try:
            await self._run_pipeline(session, user_message, messages)
        except asyncio.CancelledError:
            await session.emit({"type": "error", "message": "Pipeline was aborted"})
        except aiohttp.ClientConnectorError as e:
            logger.error("[pipeline] LLM backend connection error: %s", e)
            await session.emit({"type": "error", "message": "LLM backend is not running"})
        except Exception as e:
            logger.exception("[pipeline] unexpected error")
            await session.emit({"type": "error", "message": str(e)})

    async def _run_pipeline(
        self,
        session: AgentSession,
        user_message: str,
        messages: list[dict],
    ) -> None:
        wd = self._working_directory

        # Stage 1: Classify
        classify_result = await run_stage(
            "classify",
            [{"role": "system", "content": _CLASSIFY_SYSTEM}, {"role": "user", "content": user_message}],
            [],
            FinishClassify(),
            session,
            wd,
        )

        if classify_result.get("complexity") == "simple":
            logger.info("[pipeline] classified as simple — running direct agent loop")
            await run_agent(session, messages, self._regular_tools, wd)
            return

        # Stage 2: Augment
        augment_result = await run_stage(
            "augment",
            [{"role": "system", "content": _AUGMENT_SYSTEM}, {"role": "user", "content": user_message}],
            ["read_file", "glob_files", "grep_files", "list_directory"],
            FinishAugmentation(),
            session,
            wd,
        )
        augmented_prompt = augment_result.get("augmented_prompt") or user_message
        context_notes = augment_result.get("context_notes") or ""

        # Stage 3: Critique
        critique_input = (
            f"Original request: {user_message}\n\n"
            f"Augmented prompt:\n{augmented_prompt}\n\n"
            f"Context notes:\n{context_notes}"
        )
        critique_result = await run_stage(
            "critique",
            [{"role": "system", "content": _CRITIQUE_SYSTEM}, {"role": "user", "content": critique_input}],
            [],
            FinishCritique(),
            session,
            wd,
        )
        final_prompt = critique_result.get("final_prompt") or augmented_prompt

        # Stage 4: Plan
        plan_result = await run_stage(
            "plan",
            [{"role": "system", "content": _PLAN_SYSTEM}, {"role": "user", "content": final_prompt}],
            [],
            FinishPlan(),
            session,
            wd,
        )
        tasks = [
            PipelineTask(
                id=t.get("id") or str(i),
                description=t.get("description") or "",
                verification_method=t.get("verification_method") or "",
            )
            for i, t in enumerate(plan_result.get("tasks") or [])
            if t.get("description")
        ]

        if not tasks:
            logger.warning("[pipeline] plan produced no tasks — falling back to direct agent loop")
            await run_agent(session, messages, self._regular_tools, wd)
            return

        # Stages 5+6: Execute + Verify each task
        task_results: list[PipelineTask] = []
        for task in tasks:
            task.status = "in_progress"
            success = False

            for attempt in range(MAX_TASK_RETRIES + 1):
                if attempt > 0:
                    logger.info("[pipeline] retrying task %s (attempt %d)", task.id, attempt + 1)

                retry_note = (
                    f"\n\nPrevious attempt failed: {task.result}"
                    if attempt > 0 and task.result is not None
                    else ""
                )
                execute_messages = [
                    *self._system_messages,
                    {
                        "role": "user",
                        "content": (
                            f"Execute the following task. When done, call finish_task.\n\n"
                            f"Task: {task.description}\n\n"
                            f"Full context:\n{final_prompt}"
                            f"{retry_note}"
                        ),
                    },
                ]
                execute_result = await run_stage(
                    "execute",
                    execute_messages,
                    list(TOOL_REGISTRY.keys()),
                    FinishTask(),
                    session,
                    wd,
                )
                task.result = execute_result.get("result") or ""

                if execute_result.get("status") == "failure":
                    task.retry_count += 1
                    continue

                # Verify
                verify_result = await run_stage(
                    "verify",
                    [
                        {"role": "system", "content": _VERIFY_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"Task: {task.description}\n\n"
                                f"Verification method: {task.verification_method}\n\n"
                                f"Result reported by executor: {task.result}"
                            ),
                        },
                    ],
                    ["read_file", "glob_files", "run_shell"],
                    FinishVerify(),
                    session,
                    wd,
                )

                if verify_result.get("passed", False):
                    task.status = "done"
                    success = True
                    break

                task.retry_count += 1
                logger.info(
                    "[pipeline] task %s verify failed: %s", task.id, verify_result.get("issues", "")
                )

            if not success:
                task.status = "failed"

            task_results.append(task)

        # Synthesize: inject pipeline context into the final agent run
        task_summary = "\n".join(
            f"- {t.id} [{t.status}]: {t.description} → {t.result or 'no result'}"
            for t in task_results
        )
        synth_messages = list(messages)
        if synth_messages and synth_messages[-1].get("role") == "user":
            original_content = synth_messages[-1]["content"]
            synth_messages[-1] = {
                "role": "user",
                "content": (
                    f"{original_content}\n\n"
                    f"[Pipeline pre-execution complete. Task results:\n{task_summary}]\n\n"
                    f"Summarize what was accomplished and flag any failed tasks."
                ),
            }

        await run_agent(session, synth_messages, self._regular_tools, wd)
