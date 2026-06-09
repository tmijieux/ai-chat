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
from agent.tools.explore_codebase import _read_lines

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
    verify_issues: str | None = None
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
You are a prompt augmentation agent. Your only tool is explore_codebase(query).
You have at most 3 turns. Each turn you either explore or submit — never both in the same turn.
- Turn 1: explore the main subject.
- Turn 2: if something critical is still missing, do ONE hyper-specific explore — name the exact file or symbol needed. Otherwise call finish_augmentation.
- Turn 3: you MUST call finish_augmentation with whatever you have. No more explores.
In finish_augmentation, the snippets field is REQUIRED. List the file locations (file_path, start_line, end_line) needed to plan and implement the changes — the system will read the actual code from those coordinates.
Do not write a text response — call the tool instead.\
"""

_CRITIQUE_SYSTEM = """\
You are a critical reviewer. You receive an original request, an augmented prompt, and code snippets already read from the codebase.
Check: does the augmented prompt correctly capture the intent? Do the file paths and function names match the provided code?
In finish_critique, the snippets field is REQUIRED. Return only the snippet coordinates actually needed to plan and implement the changes — prune irrelevant ones, keep relevant ones unchanged (same file_path, start_line, end_line).
You MUST call finish_critique to submit your verdict. Do not write a text response — call the tool instead.\
"""

_PLAN_SYSTEM = """\
You are a task planner working with a model that needs extremely precise instructions.
The user message contains the exact code snippets needed for this task — file paths, line numbers, and actual code content. Use them directly. You have no search tools — call finish_plan immediately after reasoning through the tasks.

!! ONE FILE PER TASK — this is non-negotiable. If a change touches two files, it must be two separate tasks. A task that edits both a .ts and a .html file is INVALID. !!

Rules for each task:
- ONE file per task. No exceptions.
- The description must be fully self-contained: exact file path, exact insertion point (e.g. "after the closing </button> tag on line 68"), exact attribute names, method names, class names, and signal names. A developer with zero context must be able to execute it from the description alone.
- Keep each task under 15 lines of code change.
- Generate at minimum 5 tasks. If you have fewer than 5, you are being too coarse — split further.
- Each task must describe exactly ONE action. If the description contains "and", "also", "then", or "but" connecting two actions, it must be split into separate tasks. Read each description and ask: does this do more than one thing? If yes, split it.
- Verification must name a specific string to grep and a specific file — never "verify it works".

BAD task (two files — invalid):
  "Add togglePipelineMode() to chat-input.component.ts and add a toggle button in chat-input.component.html"

GOOD tasks (one file each — required):
  Task A: "In chat-client/src/components/chat-input/chat-input.component.ts, after the closing brace of the constructor on line 61, add: togglePipelineMode(): void \{ this.chatSvc.togglePipelineMode() \}"
  Task B: "In chat-client/src/components/chat-input/chat-input.component.html, after the closing </button> of the Send button on line 70, add a sibling <button> with (click)=\"togglePipelineMode()\", [class.active]=\"chatSvc.pipelineMode()\", class=\"pipeline-toggle-btn\", inner text 'Pipeline'."

You MUST call finish_plan to submit the task list.\
"""

_EXECUTE_SYSTEM = """\
You are a precise code executor. Your task is always small and exactly scoped.
Rules:
- Do EXACTLY what the task description says — nothing more, nothing less.
- Do NOT refactor, rename, reformat, or touch anything outside the task scope.
- Do NOT add extra features, comments, or improvements not asked for.
- If the task says add one line, add one line. If it says edit one attribute, edit one attribute.
When done, call finish_task.\
"""

_VERIFY_SYSTEM = """\
You are a verification agent. Check two things:
1. Was the task completed correctly according to the verification method?
2. Did the executor do MORE than the task asked? (scope creep — extra refactors, renames, reformats, unrelated changes)
Use read_file, glob_files, and run_shell to inspect the actual state of the code or filesystem.
If the task passed but scope creep was detected, set passed=false and describe the extra changes in issues.
You MUST call finish_verify to submit your verdict. Do not write a text response — call the tool instead.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_code_context(snippets: list[dict], working_directory: str) -> str:
    """Read snippet coordinates and return formatted code blocks for prompt injection."""
    parts = []
    for snippet in snippets:
        file_path = snippet.get("file_path", "")
        start_line = snippet.get("start_line", 1)
        end_line = snippet.get("end_line", start_line)
        code = _read_lines(file_path, start_line, end_line, working_directory)
        if code is not None:
            parts.append(f"[{file_path} lines {start_line}-{end_line}]\n{code}")
    return "\n\n".join(parts)


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
    max_iterations: int = MAX_STAGE_ITERATIONS,
    numbered_name: str | None = None,
    inject_turn_reminders: bool = False,
    finish_tool_schema: dict | None = None,
) -> None:
    """Inner loop for a pipeline stage. Stops when the finish tool is called or no more tool calls."""
    display_name = numbered_name or stage_name
    for iteration in range(max_iterations):
        is_last_turn = (iteration == max_iterations - 1)
        active_schemas = [finish_tool_schema] if (is_last_turn and finish_tool_schema is not None) else tool_schemas
        done, _ = await chat_with_tools(
            stage_messages, sub_session, active_schemas, working_directory, extra_tools=extra_tools
        )
        if sub_session.finish_result is not None:
            break
        if done:
            logger.warning(
                "[pipeline:%s] model stopped at iteration %d without calling finish tool",
                display_name,
                iteration,
            )
            break
        if inject_turn_reminders:
            turns_used = iteration + 1
            turns_left = max_iterations - turns_used
            if turns_left == 0:
                reminder = f"[Turn {turns_used}/{max_iterations} complete. No turns remaining — you will be terminated. Call the finish tool NOW with whatever you found.]"
            elif turns_left == 1:
                reminder = f"[Turn {turns_used}/{max_iterations} complete. NEXT TURN IS YOUR LAST — call ONLY the finish tool, no other tool calls. Submit what you have found so far, mark it inconclusive if needed.]"
            elif turns_left == 2:
                reminder = f"[Turn {turns_used}/{max_iterations} complete. {turns_left} turns remaining. Do your last search NOW — the turn after must be finish_explore only.]"
            else:
                reminder = f"[Turn {turns_used}/{max_iterations} complete. {turns_left} turns remaining.]"
            stage_messages.append({"role": "user", "content": reminder})
    else:
        logger.warning("[pipeline:%s] max iterations (%d) reached without calling finish tool", display_name, max_iterations)
        await sub_session.emit({"type": "error", "message": f"[pipeline:{display_name}] max iterations ({max_iterations}) reached without calling finish tool"})
    await sub_session.outbound.put({"type": "_stage_done"})


async def run_stage(
    stage_name: str,
    messages: list[dict],
    regular_tool_names: list[str],
    finish_tool,
    parent_session: AgentSession,
    working_directory: str | None,
    max_iterations: int = MAX_STAGE_ITERATIONS,
    inject_turn_reminders: bool = False,
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

    count = parent_session._sub_stage_counters.get(stage_name, 0) + 1
    parent_session._sub_stage_counters[stage_name] = count
    numbered_name = f"{stage_name}#{count}"

    logger.info("[pipeline] starting stage: %s (call #%d)", stage_name, count)

    finish_schema = _finish_tool_schema(finish_tool)
    loop_task = asyncio.create_task(
        _run_stage_loop(sub_session, stage_messages, all_schemas, extra_tools, working_directory, stage_name, max_iterations, numbered_name, inject_turn_reminders, finish_schema)
    )

    while True:
        event = await sub_session.outbound.get()
        if event["type"] == "_stage_done":
            break
        existing = event.get("_pipeline_stage")
        tag = f"{numbered_name}.{existing}" if existing else numbered_name
        await parent_session.emit({**event, "_pipeline_stage": tag})

    await loop_task

    if sub_session.finish_result is None:
        msg = f"[pipeline:{numbered_name}] stage did not call its finish tool — aborting"
        logger.error(msg)
        await parent_session.emit({"type": "error", "message": msg, "_pipeline_stage": numbered_name})
        raise RuntimeError(msg)

    result = sub_session.finish_result
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
            ["explore_codebase"],
            FinishAugmentation(),
            session,
            wd,
            max_iterations=3,
            inject_turn_reminders=True,
        )
        augmented_prompt = augment_result.get("augmented_prompt") or user_message
        augment_snippets = augment_result.get("snippets") or []
        augment_code_context = _build_code_context(augment_snippets, wd) if wd else ""
        await session.emit({"type": "pipeline_summary", "label": "augmented prompt", "content": augmented_prompt, "notes": None})

        # Stage 3: Critique
        critique_input = (
            f"Original request: {user_message}\n\n"
            f"Augmented prompt:\n{augmented_prompt}\n\n"
            f"Code snippets:\n{augment_code_context}"
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
        issues = critique_result.get("issues") or []
        critique_snippets = critique_result.get("snippets") or augment_snippets
        critique_code_context = _build_code_context(critique_snippets, wd) if wd else ""
        await session.emit({"type": "pipeline_summary", "label": "final prompt", "content": final_prompt, "notes": ("issues: " + "; ".join(issues)) if issues else None})

        # Stage 4: Plan
        plan_input = final_prompt
        if critique_code_context:
            plan_input += f"\n\n--- Code context ---\n{critique_code_context}"
        plan_result = await run_stage(
            "plan",
            [{"role": "system", "content": _PLAN_SYSTEM}, {"role": "user", "content": plan_input}],
            [],
            FinishPlan(),
            session,
            wd,
            max_iterations=5,
            inject_turn_reminders=True,
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

                if attempt > 0 and task.verify_issues is not None:
                    retry_note = f"\n\nPrevious attempt failed verification: {task.verify_issues}"
                else:
                    retry_note = ""
                execute_messages = [
                    {"role": "system", "content": _EXECUTE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
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

                task.verify_issues = verify_result.get("issues") or ""
                task.retry_count += 1
                logger.info(
                    "[pipeline] task %s verify failed: %s", task.id, task.verify_issues
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
