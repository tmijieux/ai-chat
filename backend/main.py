import datetime
import uuid
import logging
import json
import asyncio
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from fastapi.responses import StreamingResponse
from sqlalchemy import select, delete
from database import init_db, get_db_session, AsyncSession
import loaders as ld
import tables as db
import aiohttp
import asyncio
from agent.agent import AgentSession, run_agent, _find_superseded_read_file_indices
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from agent.count_token import count_token

MODEL_NAME = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------

async def _ensure_ollama_running() -> None:
    async with aiohttp.ClientSession() as http:
        try:
            async with http.get(f"{OLLAMA_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    logger.info("Ollama already running.")
                    return
        except Exception:
            pass

    logger.info("Ollama not detected — launching 'ollama serve' ...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    async with aiohttp.ClientSession() as http:
        for _ in range(60):  # 30 s total
            await asyncio.sleep(0.5)
            try:
                async with http.get(f"{OLLAMA_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=1)) as r:
                    if r.status == 200:
                        logger.info("Ollama started successfully.")
                        return
            except Exception:
                pass

    logger.warning("Ollama did not respond within 30 s — continuing without it.")


async def check_ollama() -> None:
    async with aiohttp.ClientSession() as http:
        try:
            async with http.get(f"{OLLAMA_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
    raise HTTPException(status_code=503, detail="Ollama is not running")


def disable_sqlalchemy_logging():
    for name in ["sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine", "aiosqlite"]:
        log = logging.getLogger(name)
        log.disabled = True
        log.propagate = False


disable_sqlalchemy_logging()
logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Prompt seeding
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from a markdown file. No external dependency."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict = {}
    for line in parts[1].splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.lower() == "true":
            meta[key] = True
        elif value.lower() == "false":
            meta[key] = False
        elif value.startswith('"') and value.endswith('"'):
            meta[key] = value[1:-1]
        else:
            meta[key] = value
    return meta, parts[2].strip()


async def _seed_prompts(sess: AsyncSession) -> None:
    prompts_dir = Path(__file__).parent / "prompts"
    if not prompts_dir.exists():
        return
    for md_file in sorted(prompts_dir.glob("*.md")):
        meta, content = _parse_frontmatter(md_file.read_text(encoding="utf-8"))
        if not meta.get("is_current"):
            continue
        name = meta.get("name", md_file.stem)
        existing = (
            await sess.scalars(
                select(db.SystemPromptTemplate).where(db.SystemPromptTemplate.name == name)
            )
        ).first()
        if existing:
            continue
        token_count_value = None
        try:
            token_count_value = await count_token([{"role": "system", "content": content}], tool_names=[])
        except Exception:
            pass
        sess.add(
            db.SystemPromptTemplate(
                id=str(uuid.uuid4()),
                name=name,
                category=meta.get("category", "general"),
                content=content,
                is_default=True,
                token_count=token_count_value,
                created_at=_now(),
            )
        )
    await sess.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _ensure_ollama_running()
    await init_db()
    print("Database initialized successfully.")
    async for sess in get_db_session():
        await _seed_prompts(sess)
    yield


app = FastAPI(title="LLM Chat Backend", lifespan=lifespan)




# ---------------------------------------------------------------------------
# Branch / tree helpers
# ---------------------------------------------------------------------------

def _build_active_branch_path(
    messages: list[db.Message], active_message_id: str | None
) -> list[db.Message]:
    """Walk parent_id chain from active leaf up to root, return in root→leaf order."""
    if not active_message_id:
        return []
    msg_map = {m.id: m for m in messages}
    path: list[db.Message] = []
    current_id: str | None = active_message_id
    while current_id is not None:
        msg = msg_map.get(current_id)
        if msg is None:
            break
        path.append(msg)
        current_id = msg.parent_id
    return list(reversed(path))


def _find_deepest_leaf(messages: list[db.Message], start_id: str) -> str:
    """Follow the single-child path from start_id until a fork or true leaf."""
    children_map: dict[str, list[str]] = {}
    for m in messages:
        if m.parent_id:
            children_map.setdefault(m.parent_id, []).append(m.id)
    current = start_id
    while True:
        children = children_map.get(current, [])
        if len(children) == 1:
            current = children[0]
        else:
            break
    return current


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


async def _build_inference_context(
    branch: list[db.Message],
    prompt_id: str | None,
    sess: AsyncSession,
) -> list[dict]:
    """Build the message list sent to Ollama. Prepends system prompt from DB if prompt_id given."""
    messages: list[dict] = []
    if prompt_id is not None:
        p = (
            await sess.scalars(
                select(db.SystemPromptTemplate).where(db.SystemPromptTemplate.id == prompt_id)
            )
        ).first()
        if p is not None:
            messages.append({"role": "system", "content": p.content})
    for m in branch:
        if m.context_excluded:
            continue
        if m.content is None or m.content.strip() == "":
            continue
        messages.append({"role": m.role, "content": m.content})
    return messages


def _deduplicate_branch_file_reads(branch: list[db.Message]) -> None:
    pairs = [(m.role, m.content or "") for m in branch]
    for i in _find_superseded_read_file_indices(pairs):
        branch[i].context_excluded = True
        branch[i].exclusion_reason = "file_superseded"


def _parse_conv_settings(conv: db.Conversation) -> ld.ConversationSettings:
    try:
        return ld.ConversationSettings.model_validate_json(conv.settings or "{}")
    except Exception:
        return ld.ConversationSettings()


def _msg_dict(m: db.Message) -> dict:
    return {
        "id": m.id,
        "conversation_id": m.conversation_id,
        "parent_id": m.parent_id,
        "role": m.role,
        "content": m.content,
        "thinking": m.thinking,
        "created_at": m.created_at,
        "token_count": m.token_count,
        "token_delta": m.token_delta,
        "context_excluded": m.context_excluded,
        "exclusion_reason": m.exclusion_reason,
    }


def _enrich_branch(path: list[db.Message], all_msgs: list[db.Message]) -> list[dict]:
    """Return _msg_dict for each message in path, enriched with sibling navigation metadata."""
    children_by_parent: dict[str | None, list[db.Message]] = {}
    for m in all_msgs:
        children_by_parent.setdefault(m.parent_id, []).append(m)
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda m: m.created_at)

    result = []
    for m in path:
        siblings = children_by_parent.get(m.parent_id, [m])
        idx = next((i for i, s in enumerate(siblings) if s.id == m.id), 0)
        count = len(siblings)
        result.append({
            **_msg_dict(m),
            "sibling_count": count,
            "sibling_index": idx + 1,
            "prev_sibling_id": siblings[idx - 1].id if idx > 0 else None,
            "next_sibling_id": siblings[idx + 1].id if idx < count - 1 else None,
            "has_children": len(children_by_parent.get(m.id, [])) > 0,
        })
    return result


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations(sess: AsyncSession = Depends(get_db_session)):
    result = await sess.execute(select(db.Conversation))
    convs = result.scalars().all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "settings": c.settings,
            "created_at": c.created_at,
            "active_message_id": c.active_message_id,
        }
        for c in convs
    ]


@app.post("/api/conversations")
async def create_conversation(
    input: ld.NewConversation, sess: AsyncSession = Depends(get_db_session)
):
    default_settings = ld.ConversationSettings(
        active_tool_names=list(TOOL_REGISTRY.keys())
    ).model_dump_json()
    new_conv = db.Conversation(
        id=str(uuid.uuid4()),
        title=input.title,
        settings=default_settings,
        created_at=_now(),
    )
    sess.add(new_conv)
    await sess.flush()
    return {
        "id": new_conv.id,
        "title": new_conv.title,
        "settings": new_conv.settings,
        "created_at": new_conv.created_at,
        "active_message_id": new_conv.active_message_id,
    }


@app.delete("/api/conversations/{id}")
async def delete_conversation(id: str, sess: AsyncSession = Depends(get_db_session)):
    await sess.execute(delete(db.Message).where(db.Message.conversation_id == id))
    await sess.execute(delete(db.Conversation).where(db.Conversation.id == id))
    return ""


@app.put("/api/conversations/{id}/settings")
async def update_conversation_settings(
    id: str, body: ld.ConversationSettings, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))
    ).first()
    if conv is None:
        raise HTTPException(404)
    conv.settings = json.dumps(body.model_dump())
    return body


@app.put("/api/conversations/{id}/active-branch")
async def set_active_branch(
    id: str, body: ld.BranchNavigation, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))
    ).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list(
        (
            await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))
        ).all()
    )
    leaf_id = _find_deepest_leaf(all_msgs, body.message_id)
    conv.active_message_id = leaf_id
    path = _build_active_branch_path(all_msgs, leaf_id)
    return {
        "active_message_id": leaf_id,
        "messages": _enrich_branch(path, all_msgs),
    }


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@app.get("/api/conversations/{id}/messages")
async def get_conversation_messages(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))
    ).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list(
        (
            await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))
        ).all()
    )
    path = _build_active_branch_path(all_msgs, conv.active_message_id)
    return _enrich_branch(path, all_msgs)


@app.get("/api/conversations/{id}/tree")
async def get_conversation_tree(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))
    ).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list(
        (
            await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))
        ).all()
    )
    sibling_count: dict[str | None, int] = {}
    for m in all_msgs:
        sibling_count[m.parent_id] = sibling_count.get(m.parent_id, 0) + 1

    return {
        "active_message_id": conv.active_message_id,
        "nodes": [
            {
                "id": m.id,
                "parent_id": m.parent_id,
                "role": m.role,
                "content_preview": (m.content or "")[:60],
                "created_at": m.created_at,
                "sibling_count": sibling_count.get(m.parent_id, 1),
            }
            for m in all_msgs
        ],
    }


@app.post("/api/messages")
async def add_message(
    message: ld.NewMessage,
    conversationId: str,
    sess: AsyncSession = Depends(get_db_session),
):
    conv = (
        await sess.scalars(
            select(db.Conversation).where(db.Conversation.id == conversationId)
        )
    ).first()
    if conv is None:
        raise HTTPException(404)

    parent_id = message.parent_id if message.parent_id is not None else conv.active_message_id

    msg_id = message.id
    sess.add(
        db.Message(
            id=msg_id,
            conversation_id=conversationId,
            parent_id=parent_id,
            content=message.content,
            thinking=message.thinking,
            created_at=_now(),
            role=message.role,
            token_count=message.token_count,
            token_delta=message.token_delta,
        )
    )
    conv.active_message_id = msg_id
    await sess.flush()
    return {"id": msg_id, "parent_id": parent_id}


@app.patch("/api/messages/{id}/token-count")
async def update_message_token_count(
    id: str, body: ld.UpdateTokenCount, sess: AsyncSession = Depends(get_db_session)
):
    msg = (await sess.scalars(select(db.Message).where(db.Message.id == id))).first()
    if msg is None:
        raise HTTPException(404)
    if body.token_count is not None:
        msg.token_count = body.token_count
    if body.token_delta is not None:
        msg.token_delta = body.token_delta
    await sess.flush()
    return {"ok": True}


@app.put("/api/messages/{id}/branch")
async def branch_message(
    id: str, body: ld.EditMessageContent, sess: AsyncSession = Depends(get_db_session)
):
    original = (
        await sess.scalars(select(db.Message).where(db.Message.id == id))
    ).first()
    if original is None:
        raise HTTPException(404)

    new_id = str(uuid.uuid4())
    sess.add(
        db.Message(
            id=new_id,
            conversation_id=original.conversation_id,
            parent_id=original.parent_id,
            content=body.content,
            thinking=None,
            created_at=_now(),
            role=original.role,
        )
    )
    conv = (
        await sess.scalars(
            select(db.Conversation).where(
                db.Conversation.id == original.conversation_id
            )
        )
    ).first()
    if conv:
        conv.active_message_id = new_id
    await sess.flush()
    return {"id": new_id, "parent_id": original.parent_id}


@app.delete("/api/conversations/{conv_id}/messages/{msg_id}")
async def delete_message_branch(
    conv_id: str, msg_id: str, subtree: bool = True, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == conv_id))
    ).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list(
        (
            await sess.scalars(
                select(db.Message).where(db.Message.conversation_id == conv_id)
            )
        ).all()
    )
    msg_map = {m.id: m for m in all_msgs}
    children_map: dict[str, list[str]] = {}
    for m in all_msgs:
        if m.parent_id:
            children_map.setdefault(m.parent_id, []).append(m.id)

    target = msg_map.get(msg_id)
    if target is None:
        raise HTTPException(404)

    if subtree:
        to_delete: set[str] = set()
        queue = [msg_id]
        while queue:
            current = queue.pop()
            to_delete.add(current)
            queue.extend(children_map.get(current, []))

        if conv.active_message_id in to_delete:
            remaining = [m for m in all_msgs if m.id not in to_delete]
            new_active: str | None = None
            if target.parent_id:
                siblings = [m.id for m in all_msgs if m.parent_id == target.parent_id and m.id not in to_delete]
                if siblings:
                    new_active = _find_deepest_leaf(remaining, siblings[0])
                else:
                    new_active = target.parent_id
            else:
                other_roots = [m.id for m in remaining if m.parent_id is None]
                if other_roots:
                    new_active = _find_deepest_leaf(remaining, other_roots[0])
            conv.active_message_id = new_active

        await sess.execute(delete(db.Message).where(db.Message.id.in_(list(to_delete))))
        return {"deleted": list(to_delete)}
    else:
        # Single-message delete: re-parent direct children to target's parent
        direct_children = children_map.get(msg_id, [])
        for child_id in direct_children:
            child = msg_map.get(child_id)
            if child:
                child.parent_id = target.parent_id
        await sess.flush()

        if conv.active_message_id == msg_id:
            remaining = [m for m in all_msgs if m.id != msg_id]
            if direct_children:
                new_active = _find_deepest_leaf(remaining, direct_children[0])
            elif target.parent_id:
                new_active = target.parent_id
            else:
                new_active = next((m.id for m in remaining if m.parent_id is None), None)
            conv.active_message_id = new_active

        await sess.execute(delete(db.Message).where(db.Message.id == msg_id))
        return {"deleted": [msg_id]}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

async def _compute_prompt_token_count(content: str) -> int | None:
    try:
        return await count_token([{"role": "system", "content": content}], tool_names=[])
    except Exception as e:
        logger.exception("unexpected error in _compute_prompt_token_count()", exc_info=e)
        return None


@app.get("/api/system-prompts")
async def list_system_prompts(sess: AsyncSession = Depends(get_db_session)):
    result = await sess.scalars(select(db.SystemPromptTemplate))
    prompts = result.all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "category": p.category,
            "content": p.content,
            "is_default": p.is_default,
            "token_count": p.token_count,
            "created_at": p.created_at,
        }
        for p in prompts
    ]


@app.post("/api/system-prompts")
async def create_system_prompt(
    body: ld.NewSystemPrompt, sess: AsyncSession = Depends(get_db_session)
):
    token_count_value = await _compute_prompt_token_count(body.content)
    p = db.SystemPromptTemplate(
        id=str(uuid.uuid4()),
        name=body.name,
        category=body.category,
        content=body.content,
        is_default=body.is_default,
        token_count=token_count_value,
        created_at=_now(),
    )
    sess.add(p)
    await sess.flush()
    return {
        "id": p.id,
        "name": p.name,
        "category": p.category,
        "content": p.content,
        "is_default": p.is_default,
        "token_count": p.token_count,
        "created_at": p.created_at,
    }


@app.put("/api/system-prompts/{id}")
async def update_system_prompt(
    id: str, body: ld.UpdateSystemPrompt, sess: AsyncSession = Depends(get_db_session)
):
    p = (
        await sess.scalars(
            select(db.SystemPromptTemplate).where(db.SystemPromptTemplate.id == id)
        )
    ).first()
    if p is None:
        raise HTTPException(404)
    if body.name is not None:
        p.name = body.name
    if body.category is not None:
        p.category = body.category
    if body.content is not None:
        p.content = body.content
        p.token_count = await _compute_prompt_token_count(body.content)
    if body.is_default is not None:
        p.is_default = body.is_default
    return {
        "id": p.id,
        "name": p.name,
        "category": p.category,
        "content": p.content,
        "is_default": p.is_default,
        "token_count": p.token_count,
        "created_at": p.created_at,
    }


@app.delete("/api/system-prompts/{id}")
async def delete_system_prompt(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    await sess.execute(
        delete(db.SystemPromptTemplate).where(db.SystemPromptTemplate.id == id)
    )
    return ""


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

@app.get("/api/utils/browse-directory")
async def browse_directory(path: str | None = None):
    current = Path(path).resolve() if path else Path.cwd()
    try:
        entries = sorted(
            [
                {"name": e.name, "path": str(e)}
                for e in current.iterdir()
                if e.is_dir() and not e.name.startswith(".")
            ],
            key=lambda e: e["name"].lower(),
        )
    except PermissionError:
        raise HTTPException(403, detail=f"Permission denied: {current}")
    parent = str(current.parent) if current != current.parent else None
    return {"path": str(current), "parent": parent, "entries": entries}


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

@app.get("/api/agent/tools")
async def list_agent_tools():
    from agent.tools.base import TOOL_FRAMEWORK_OVERHEAD, STACKING_OVERHEAD_PER_ADDITIONAL_TOOL
    return {
        "framework_overhead": TOOL_FRAMEWORK_OVERHEAD,
        "stacking_overhead_per_additional_tool": STACKING_OVERHEAD_PER_ADDITIONAL_TOOL,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "requires_confirmation": t.requires_confirmation,
                "token_count": t.token_count,
            }
            for t in TOOL_REGISTRY.values()
        ],
    }


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------

async def get_http_session():
    async with aiohttp.ClientSession(conn_timeout=10) as httpsess:
        yield httpsess


OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"


async def chat_endpoint_generator(
    http_sess: aiohttp.ClientSession, conversation: ld.Conversation
):
    try:
        query_body = {
            "model": MODEL_NAME,
            "messages": [m.model_dump() for m in conversation.messages],
            "stream": True,
        }
        print("query_body=", query_body)
        async with http_sess.post(OLLAMA_URL, json=query_body) as response:
            if response.status != 200:
                logger.error("ollama response code %s", response.status)
                yield '{"error":true,"done":true}\n'
                return
            async for chunk_line in response.content:
                if chunk_line:
                    data = json.loads(chunk_line)
                    print(".", end="", flush=True)
                    if "done" in data and data["done"]:
                        print(json.dumps(data, indent=2))
                    yield json.dumps(data) + "\n"
    except Exception:
        logger.exception("chat stream error")
        yield '{"error":true,"done":true}\n'


@app.post("/api/chat")
async def chat_endpoint(
    conversation: ld.Conversation,
    http_sess: aiohttp.ClientSession = Depends(get_http_session),
    _: None = Depends(check_ollama),
):
    print(f"--- Received chat request for {len(conversation.messages)} messages ---")
    return StreamingResponse(
        chat_endpoint_generator(http_sess, conversation),
        media_type="application/x-ndjson",
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

@app.post("/api/conversations/{id}/count-tokens")
async def count_conversation_tokens(id: str, sess: AsyncSession = Depends(get_db_session), _: None = Depends(check_ollama)):
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)

    settings = _parse_conv_settings(conv)
    messages = await _build_inference_context(branch, settings.active_prompt_id, sess)
    tool_names = settings.active_tool_names if conv.settings is not None else list(TOOL_REGISTRY.keys())
    token_count_value = await count_token(messages, tool_names=tool_names)

    last_id: str | None = None
    if branch:
        branch[-1].token_count = token_count_value
        last_id = branch[-1].id
        await sess.flush()

    return {"token_count": token_count_value, "message_id": last_id}


# ---------------------------------------------------------------------------
# Agent WebSocket
# ---------------------------------------------------------------------------

async def _ws_receive_messages(websocket: WebSocket, session: AgentSession, agent_task: asyncio.Task) -> None:
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "confirm":
                session.resolve_confirm(data["tool_id"], data["approved"], data.get("reason"))
            elif data.get("type") == "abort":
                agent_task.cancel()
                return
    except (WebSocketDisconnect, Exception):
        agent_task.cancel()


@app.websocket("/api/agent/ws")
async def agent_websocket(websocket: WebSocket, sess: AsyncSession = Depends(get_db_session)):
    """
    WebSocket endpoint for the agentic loop.
    First client message: {"message": "...", "conversation_id": "optional"}
    """
    await websocket.accept()
    try:
        init_data = await websocket.receive_json()
        user_message: str = init_data.get("message", "")
        conversation_id: str | None = init_data.get("conversation_id")
        # When set, the user message is already in DB (edit flow) — don't append it again.
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
        active_tool_names = settings.active_tool_names if (conv and conv.settings is not None) else list(TOOL_REGISTRY.keys())
        working_directory = settings.working_directory

        tools = get_ollama_tool_list(active_tool_names)
        messages = await _build_inference_context(branch, settings.active_prompt_id, sess)
        if user_message_id is None:
            messages.append({"role": "user", "content": user_message})

        session = AgentSession()
        agent_task = asyncio.create_task(run_agent(session, messages, tools, working_directory))

        async def send_events_to_websocket() -> None:
            while True:
                event = await session.outbound.get()
                await websocket.send_json(event)
                if event["type"] in ("done", "error"):
                    return

        send_task = asyncio.create_task(send_events_to_websocket())
        recv_task = asyncio.create_task(_ws_receive_messages(websocket, session, agent_task))

        await send_task
        recv_task.cancel()
        agent_task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("error in main agent websocket handling")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            logger.exception("error when sending error message in websocket")
            pass
