import asyncio
import functools
import json
import logging

import aiohttp
from fastapi import APIRouter, Depends, Form, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from agent.agent import AgentSession, run_agent
from agent.pipeline import PipelineOrchestrator
from agent.workflow_loader import load_workflow
from agent.custom_workflow import CustomWorkflowOrchestrator
from agent.tools import TOOL_REGISTRY, PLAN_MODE_TOOLS, CONVERSATIONAL_TOOLS, get_ollama_tool_list
from conv_helpers import (
    ConvBranch,
    ToolSet,
    _build_active_branch_path,
    _build_inference_context,
    _deduplicate_branch_file_reads,
    _parse_conv_settings,
)
from database import get_db_session, AsyncSession
import loaders as ld
import tables as db
from message_types import LLMMessage
import whisper_pipeline
from pathlib import Path

router = APIRouter()

logger = logging.getLogger(__name__)

_PLAN_EXCLUDED_TOOLS = frozenset({"write_file", "edit_file", "run_shell"})
_VALID_MODES = frozenset({"standard", "auto", "plan", "yolo"})


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

_STT_CORRECTION_SYSTEM_FR = (
    "You are a speech-to-text correction assistant for a French-speaking developer using an agentic coding assistant.\n"
    "The assistant has tools to operate on files and projects: read_file, list_directory, glob_files, grep_files, edit_file, write_file, run_shell.\n"
    "The user dictates commands and questions in French, heavily mixed with English technical terms: "
    "filenames, function names, variable names, class names, CLI commands, package names, git terms, framework names.\n"
    "Common speech patterns: 'lis le fichier X', 'lance la commande X', 'modifie la fonction X dans Y', "
    "'liste les fichiers de Z', 'fais un commit', 'installe le package X'.\n"
    "The STT model often mishears English technical words as phonetically similar French words or nonsense. "
    "Use semantic coherence and typical developer vocabulary to infer the intended word — "
    "a phrase like 'lis le fichier et demi' makes no sense but 'lis le fichier README' does.\n"
    "Correct only obvious STT errors. Do not rephrase, translate, or add anything. "
    "If the input is a question or a command, output the corrected question or command — never answer it. "
    "Do NOT sanitize, soften, or replace crude language — if the user said 'merde', keep 'merde'. Your job is dictation correction, not content moderation. "
    "Return only the corrected text, nothing else."
)

_STT_EXAMPLES_FR: list[tuple[str, str]] = [
    ("Lis le fichier Redmi et explique-moi ce qu'il fait.",
     "Lis le fichier README et explique-moi ce qu'il fait."),
    ("Lance la commande nope install dans le terminal.",
     "Lance la commande npm install dans le terminal."),
    ("Ouvre le fichier rythmique point p y.",
     "Ouvre le fichier readme.py."),
    ("Je veux modifier la fonction render dès composant.",
     "Je veux modifier la fonction render du composant."),
    ("Listons les fichiers à la racine de StripoGit.",
     "Listons les fichiers à la racine du dépôt git."),
    ("Fais un commit dans le tripoGit.",
     "Fais un commit dans le dépôt git."),
    ("Qu'est-ce que fait la fonction rend deux ?",
     "Qu'est-ce que fait la fonction render ?"),
]

_STT_CORRECTION_SYSTEM_EN = (
    "You are a speech-to-text correction assistant for an English-speaking developer using an agentic coding assistant.\n"
    "The assistant has tools to operate on files and projects: read_file, list_directory, glob_files, grep_files, edit_file, write_file, run_shell.\n"
    "The user dictates commands and questions in English, with technical terms: "
    "filenames, function names, variable names, class names, CLI commands, package names, git terms, framework names.\n"
    "The STT model sometimes mishears technical terms as phonetically similar words or nonsense. "
    "Use semantic coherence and typical developer vocabulary to infer the intended word.\n"
    "Correct only obvious STT errors. Do not rephrase or add anything. "
    "If the input is a question or a command, output the corrected question or command — never answer it. "
    "Do NOT sanitize or soften language. Your job is dictation correction, not content moderation. "
    "Return only the corrected text, nothing else."
)

_STT_EXAMPLES_EN: list[tuple[str, str]] = [
    ("Read the read me file and explain what it does.",
     "Read the README file and explain what it does."),
    ("Run in pee em install in the terminal.",
     "Run npm install in the terminal."),
    ("Edit the function render component dot T S.",
     "Edit the function renderComponent.ts."),
    ("Make a get commit on the main branch.",
     "Make a git commit on the main branch."),
    ("What does the greet user function do?",
     "What does the greetUser function do?"),
]


async def _correct_stt(text: str, language: str | None) -> str:
    """Send raw STT text to the LLM for correction and return the corrected string."""
    from llm.llama_server import LLAMA_CHAT_URL, MODEL_NAME
    if language == "en":
        system = _STT_CORRECTION_SYSTEM_EN
        examples = _STT_EXAMPLES_EN
    else:
        system = _STT_CORRECTION_SYSTEM_FR
        examples = _STT_EXAMPLES_FR
    messages = [{"role": "system", "content": system}]
    for user_ex, assistant_ex in examples:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})
    messages.append({"role": "user", "content": text})
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with aiohttp.ClientSession() as http:
        async with http.post(LLAMA_CHAT_URL, json=body) as resp:
            data = await resp.json()
            logger.info("STT correction LLM response: %s", data)
            return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Agent WebSocket helpers
# ---------------------------------------------------------------------------

async def _ws_receive_messages_from_frontend(websocket: WebSocket, session: AgentSession, agent_task: asyncio.Task) -> None:
    """Forward inbound WebSocket control messages to the agent session until disconnect or abort."""
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "confirm":
                session.resolve_confirm(data["tool_id"], data["approved"], data.get("reason"))
            elif data.get("type") == "plan_accept":
                payload = {k: v for k, v in data.items() if k not in ("type", "plan_id")}
                session.resolve_plan_confirm(data["plan_id"], payload)
            elif data.get("type") == "user_question_reply":
                session.resolve_user_input(data["question_id"], data["reply"])
            elif data.get("type") == "compression_done":
                session.resume_after_compression(data.get("conversation_id", ""))
            elif data.get("type") == "set_mode":
                new_mode = data.get("mode")
                if new_mode in _VALID_MODES:
                    session.mode = new_mode
                    logger.info("session mode updated mid-run to '%s'", new_mode)
            elif data.get("type") == "abort":
                agent_task.cancel()
                return
    except (WebSocketDisconnect, Exception):
        agent_task.cancel()


async def _load_conversation_branch(
    sess: AsyncSession, conversation_id: str | None
) -> ConvBranch:
    """Load the conversation and its active message branch from the database.
    Deduplicates superseded file reads in place before returning."""
    if conversation_id is None:
        return ConvBranch(conv=None, branch=[])
    conv = (await sess.execute(
        select(db.Conversation).where(db.Conversation.id == conversation_id)
    )).scalars().first()
    if conv is None:
        return ConvBranch(conv=None, branch=[])
    all_msgs = list((await sess.execute(
        select(db.Message).where(db.Message.conversation_id == conversation_id)
    )).scalars().all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)
    _deduplicate_branch_file_reads(branch)
    await sess.flush()
    return ConvBranch(conv=conv, branch=branch)


def _build_tool_set(mode: str, active_tool_names: list[str]) -> ToolSet:
    """Assemble the tool list for the given conversation mode.
    Plan strips destructive tools and injects propose_plan + ask_user_question.
    Standard injects ask_user_question only. Auto/Yolo use the raw tool list."""
    if mode == "plan":
        filtered_names = [n for n in active_tool_names if n not in _PLAN_EXCLUDED_TOOLS]
        tools = get_ollama_tool_list(filtered_names)
        injected = {**CONVERSATIONAL_TOOLS, **PLAN_MODE_TOOLS}
        for injected_tool in injected.values():
            tools.append({"type": "function", "function": injected_tool.to_ollama_schema()})
        return ToolSet(tools=tools, extra_tools=injected)
    elif mode == "standard":
        tools = get_ollama_tool_list(active_tool_names)
        if len(tools) > 0:
            for injected_tool in CONVERSATIONAL_TOOLS.values():
                tools.append({"type": "function", "function": injected_tool.to_ollama_schema()})
            return ToolSet(tools=tools, extra_tools=dict(CONVERSATIONAL_TOOLS))
        return ToolSet(tools=tools, extra_tools=None)
    else:
        tools = get_ollama_tool_list(active_tool_names)
        return ToolSet(tools=tools, extra_tools=None)


async def _apply_db_compressions(
    sess: AsyncSession, messages: list[LLMMessage], conv_id: str
) -> list[LLMMessage]:
    """Patch in-memory tool messages with their compressed summaries after mid-run compression.
    Matches by tool_call_id so assistant messages (not persisted to DB) are preserved."""
    from tool_result_types import ToolResult
    compressed_rows = (await sess.execute(
        select(db.Message)
        .where(db.Message.conversation_id == conv_id)
        .where(db.Message.context_excluded == True)
        .where(db.Message.compressed_summary.isnot(None))
    )).scalars().all()

    call_id_to_summary: dict[str, str] = {}
    for m in compressed_rows:
        try:
            content: ToolResult = json.loads(m.content)
            call_id = content.get("tool_call_id")
            if call_id is not None and m.compressed_summary is not None:
                call_id_to_summary[call_id] = m.compressed_summary
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("continue despite error")

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        raw_content = msg.get("content")
        if not isinstance(raw_content, str):
            continue
        try:
            content_dict: ToolResult = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("continue despite error")
            continue
        call_id = content_dict.get("tool_call_id")
        if call_id is not None and call_id in call_id_to_summary:
            msg["content"] = json.dumps({
                "tool": content_dict.get("tool", "tool"),
                "status": "compressed",
                "summary": call_id_to_summary[call_id],
                "tool_call_id": call_id,
            })

    return messages


def _create_agent_task(
    session: AgentSession,
    workflow_name: str | None,
    working_directory: str | None,
    user_message: str,
    messages: list[LLMMessage],
    toolset: ToolSet,
) -> asyncio.Task:
    """Dispatch either a named workflow or the standard agentic loop as an asyncio task."""
    if workflow_name is not None:
        workflows_dir = Path(__file__).parent.parent / "workflows"
        flat_path = workflows_dir / f"{workflow_name}.yaml"
        dir_path = workflows_dir / workflow_name
        workflow_path = flat_path if flat_path.exists() else dir_path
        workflow_def = load_workflow(workflow_path)
        orchestrator = CustomWorkflowOrchestrator(workflow_def, working_directory, toolset.tools)
        return asyncio.create_task(orchestrator.run(session, user_message, messages))
    else:
        return asyncio.create_task(run_agent(session, messages, toolset, working_directory))


async def _run_agent_event_loop(
    websocket: WebSocket, session: AgentSession, agent_task: asyncio.Task
) -> None:
    """Drive the agent session over WebSocket until the agent emits done or error."""
    async def _send_events_from_agent_to_frontend() -> None:
        while True:
            event = await session.outbound.get()
            await websocket.send_json(event)
            if event["type"] in ("done", "error"):
                return

    send_task = asyncio.create_task(_send_events_from_agent_to_frontend())
    recv_task = asyncio.create_task(_ws_receive_messages_from_frontend(websocket, session, agent_task))
    await send_task
    recv_task.cancel()
    agent_task.cancel()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile,
    language: str | None = Form(default=None),
):
    import main as _main
    from fastapi import HTTPException
    if _main._whisper is None:
        raise HTTPException(503, "Whisper pipeline is still loading, try again in a moment")
    data = await audio.read()
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, whisper_pipeline.transcribe, _main._whisper, data, language
    )
    logger.info("STT raw transcript: %r", text)
    return {"text": text}


@router.post("/api/correct")
async def correct_stt(req: ld.CorrectRequest):
    if not req.text:
        return {"text": req.text}
    corrected = await _correct_stt(req.text, req.language)
    logger.info("STT corrected: %r", corrected)
    return {"text": corrected}


@router.websocket("/api/agent/ws")
async def agent_websocket(websocket: WebSocket, sess: AsyncSession = Depends(get_db_session)):
    """WebSocket endpoint for the main agentic loop."""
    await websocket.accept()
    try:
        init_data = await websocket.receive_json()
        user_message: str = init_data.get("message", "")
        conversation_id: str | None = init_data.get("conversation_id")
        user_message_id: str | None = init_data.get("user_message_id")
        workflow_name: str | None = init_data.get("workflow_name") or None

        conv_branch = await _load_conversation_branch(sess, conversation_id)
        settings = _parse_conv_settings(conv_branch.conv) if conv_branch.conv is not None else ld.ConversationSettings()
        active_tool_names = (
            settings.active_tool_names
            if (conv_branch.conv is not None and conv_branch.conv.settings is not None)
            else list(TOOL_REGISTRY.keys())
        )
        tool_set = _build_tool_set(settings.mode, active_tool_names)

        messages = await _build_inference_context(conv_branch.branch, settings.active_prompt_id, sess)
        if user_message_id is None:
            messages.append({"role": "user", "content": user_message})

        session = AgentSession()
        session.mode = settings.mode
        session.working_directory = settings.working_directory
        session.last_user_message = user_message
        if conversation_id is not None:
            session.apply_db_compressions_callback = functools.partial(_apply_db_compressions, sess, messages)

        agent_task = _create_agent_task(
            session, workflow_name, settings.working_directory,
            user_message, messages, tool_set
        )
        await _run_agent_event_loop(websocket, session, agent_task)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("error in main agent websocket handling")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            logger.exception("error when sending error message in websocket")


@router.websocket("/api/agent/pipeline/ws")
async def pipeline_websocket(websocket: WebSocket, sess: AsyncSession = Depends(get_db_session)):
    """WebSocket endpoint for the pipeline orchestrator."""
    await websocket.accept()
    try:
        init_data = await websocket.receive_json()
        user_message: str = init_data.get("message", "")
        conversation_id: str | None = init_data.get("conversation_id")
        user_message_id: str | None = init_data.get("user_message_id")

        conv: db.Conversation | None = None
        branch: list[db.Message] = []

        if conversation_id:
            result = await sess.execute(
                select(db.Conversation).where(db.Conversation.id == conversation_id)
            )
            conv = result.scalars().first()
            if conv:
                all_msgs_result = await sess.execute(
                    select(db.Message).where(db.Message.conversation_id == conversation_id)
                )
                all_msgs = list(all_msgs_result.scalars().all())
                branch = _build_active_branch_path(all_msgs, conv.active_message_id)
                _deduplicate_branch_file_reads(branch)
                await sess.flush()

        settings = _parse_conv_settings(conv) if conv else ld.ConversationSettings()
        working_directory = settings.working_directory

        tools = get_ollama_tool_list(list(TOOL_REGISTRY.keys()))
        messages = await _build_inference_context(branch, settings.active_prompt_id, sess)
        if user_message_id is None:
            messages.append({"role": "user", "content": user_message})

        system_messages = [m for m in messages if m.get("role") == "system"]

        session = AgentSession()
        orchestrator = PipelineOrchestrator(
            system_messages=system_messages,
            working_directory=working_directory,
            regular_tools=tools,
        )
        agent_task = asyncio.create_task(orchestrator.run(session, user_message, messages))

        async def send_pipeline_events_to_websocket() -> None:
            while True:
                event = await session.outbound.get()
                await websocket.send_json(event)
                if event["type"] in ("done", "error"):
                    return

        send_task = asyncio.create_task(send_pipeline_events_to_websocket())
        recv_task = asyncio.create_task(_ws_receive_messages_from_frontend(websocket, session, agent_task))

        await send_task
        recv_task.cancel()
        agent_task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("error in pipeline websocket handling")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
