import datetime
import uuid
import logging
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select, delete
from database import init_db, get_db_session, AsyncSession
import loaders as ld
import tables as db
import aiohttp
from agent.count_token import count_token
from agent.system_prompt import SYSTEM_PROMPT

MODEL_NAME = "qwen3.5:9b"

logger = logging.getLogger(__name__)


def disable_sqlalchemy_logging():
    for name in ["sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine"]:
        log = logging.getLogger(name)
        log.disabled = True
        log.propagate = False


disable_sqlalchemy_logging()
logging.basicConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database on startup."""
    await init_db()
    print("Database initialized successfully.")
    yield


app = FastAPI(title="LLM Chat Backend", lifespan=lifespan)

# Configure CORS to allow Angular (Angular default port is 4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:4200/*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    }


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations(sess: AsyncSession = Depends(get_db_session)):
    """Retrieves a list of all saved conversations."""
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
    """Creates and returns a new empty conversation object."""
    new_conv = db.Conversation(
        id=str(uuid.uuid4()),
        title=input.title,
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
    """Deletes a conversation and all its messages."""
    await sess.execute(delete(db.Message).where(db.Message.conversation_id == id))
    await sess.execute(delete(db.Conversation).where(db.Conversation.id == id))
    return ""


@app.put("/api/conversations/{id}/settings")
async def update_conversation_settings(
    id: str, body: ld.ConversationSettings, sess: AsyncSession = Depends(get_db_session)
):
    """Update per-conversation settings (active prompts, tools, agentic mode)."""
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
    """Switch the active branch. Auto-advances to the deepest single-child leaf from the given message."""
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
        "messages": [_msg_dict(m) for m in path],
    }


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@app.get("/api/conversations/{id}/messages")
async def get_conversation_messages(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    """Retrieves messages for the active branch of a conversation (root→leaf order)."""
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
    return [_msg_dict(m) for m in path]


@app.get("/api/conversations/{id}/tree")
async def get_conversation_tree(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    """Returns the full message tree with sibling counts for branch navigation."""
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
    """Append a message to a conversation. Uses conversation.active_message_id as parent when parent_id is omitted."""
    conv = (
        await sess.scalars(
            select(db.Conversation).where(db.Conversation.id == conversationId)
        )
    ).first()
    if conv is None:
        raise HTTPException(404)

    parent_id = message.parent_id if message.parent_id is not None else conv.active_message_id

    msg_id = str(uuid.uuid4())
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
        )
    )
    conv.active_message_id = msg_id
    await sess.flush()
    return {"id": msg_id, "parent_id": parent_id}


@app.put("/api/messages/{id}/branch")
async def branch_message(
    id: str, body: ld.EditMessageContent, sess: AsyncSession = Depends(get_db_session)
):
    """Create a sibling of message `id` with new content, forming a new branch. Updates conversation active branch."""
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
    conv_id: str, msg_id: str, sess: AsyncSession = Depends(get_db_session)
):
    """Delete a message and all its descendants. Adjusts active_message_id if it was inside the deleted subtree."""
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
    children_map: dict[str, list[str]] = {}
    for m in all_msgs:
        if m.parent_id:
            children_map.setdefault(m.parent_id, []).append(m.id)

    to_delete: set[str] = set()
    queue = [msg_id]
    while queue:
        current = queue.pop()
        to_delete.add(current)
        queue.extend(children_map.get(current, []))

    if conv.active_message_id in to_delete:
        msg_map = {m.id: m for m in all_msgs}
        target = msg_map.get(msg_id)
        new_active: str | None = None
        if target and target.parent_id:
            siblings = [
                m.id
                for m in all_msgs
                if m.parent_id == target.parent_id and m.id not in to_delete
            ]
            remaining = [m for m in all_msgs if m.id not in to_delete]
            if siblings:
                new_active = _find_deepest_leaf(remaining, siblings[0])
            else:
                new_active = target.parent_id
        conv.active_message_id = new_active

    await sess.execute(delete(db.Message).where(db.Message.id.in_(list(to_delete))))
    return {"deleted": list(to_delete)}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

@app.get("/api/system-prompts")
async def list_system_prompts(sess: AsyncSession = Depends(get_db_session)):
    """List all system prompt templates."""
    result = await sess.scalars(select(db.SystemPromptTemplate))
    return result.all()


@app.post("/api/system-prompts")
async def create_system_prompt(
    body: ld.NewSystemPrompt, sess: AsyncSession = Depends(get_db_session)
):
    """Create a new system prompt template."""
    p = db.SystemPromptTemplate(
        id=str(uuid.uuid4()),
        name=body.name,
        category=body.category,
        content=body.content,
        is_global=1 if body.is_global else 0,
        created_at=_now(),
    )
    sess.add(p)
    await sess.flush()
    return p


@app.put("/api/system-prompts/{id}")
async def update_system_prompt(
    id: str, body: ld.UpdateSystemPrompt, sess: AsyncSession = Depends(get_db_session)
):
    """Update fields on an existing system prompt template."""
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
    if body.is_global is not None:
        p.is_global = 1 if body.is_global else 0
    return p


@app.delete("/api/system-prompts/{id}")
async def delete_system_prompt(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    """Delete a system prompt template."""
    await sess.execute(
        delete(db.SystemPromptTemplate).where(db.SystemPromptTemplate.id == id)
    )
    return ""


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------

async def get_http_session():
    async with aiohttp.ClientSession(conn_timeout=10) as httpsess:
        yield httpsess


OLLAMA_URL = "http://localhost:11434/api/chat"


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
):
    """Streaming chat endpoint. Proxies to Ollama and streams NDJSON chunks back."""
    print(f"--- Received chat request for {len(conversation.messages)} messages ---")

    return StreamingResponse(
        chat_endpoint_generator(http_sess, conversation),
        media_type="application/x-ndjson",
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

@app.post("/api/conversations/{id}/count-tokens")
async def count_conversation_tokens(id: str, sess: AsyncSession = Depends(get_db_session)):
    """Count tokens for the active branch via a 1-token Ollama generate and persist on the last message."""
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in branch:
        if m.role == "tool" or not (m.content or "").strip():
            continue
        messages.append({"role": m.role, "content": m.content})

    token_count_value = await count_token(messages)

    last_id: str | None = None
    if branch:
        branch[-1].token_count = token_count_value
        last_id = branch[-1].id
        await sess.flush()

    return {"token_count": token_count_value, "message_id": last_id}


# ---------------------------------------------------------------------------
# Agent WebSocket
# ---------------------------------------------------------------------------

async def _ws_send_events(websocket: WebSocket, session) -> None:
    """Forward agent events to the WebSocket client until done/error."""
    while True:
        event = await session.outbound.get()
        await websocket.send_json(event)
        if event["type"] in ("done", "error"):
            return


async def _ws_receive_messages(websocket: WebSocket, session, agent_task) -> None:
    """Forward client confirm/abort messages to the agent session."""
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
    import asyncio
    from agent.agent import AgentSession, run_agent

    await websocket.accept()
    try:
        init_data = await websocket.receive_json()
        user_message: str = init_data.get("message", "")
        conversation_id: str | None = init_data.get("conversation_id")

        # Build initial message history
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
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
                for m in branch:
                    if len(m.content) == 0:
                        # skip messages that only contains thinking
                        continue
                    messages.append({"role": m.role, "content": m.content})

        messages.append({"role": "user", "content": user_message})

        # Find the last user message in DB branch so we can store its token_count after iter 1
        last_user_msg_id: str | None = None
        if conversation_id and conv and branch:
            for m in reversed(branch):
                if m.role == "user":
                    last_user_msg_id = m.id
                    break

        session = AgentSession()
        agent_task = asyncio.create_task(run_agent(session, messages))

        async def send_events_with_token_update() -> None:
            """Forward agent events to WebSocket, updating user message token_count after iteration 1."""
            iteration = 0
            while True:
                event = await session.outbound.get()
                await websocket.send_json(event)
                if event["type"] == "iteration_end":
                    iteration += 1
                    if iteration == 1 and last_user_msg_id:
                        user_msg = (await sess.scalars(select(db.Message).where(db.Message.id == last_user_msg_id))).first()
                        if user_msg:
                            user_msg.token_count = event.get("prompt_tokens", 0)
                            await sess.flush()
                if event["type"] in ("done", "error"):
                    return

        send_task = asyncio.create_task(send_events_with_token_update())
        recv_task = asyncio.create_task(_ws_receive_messages(websocket, session, agent_task))

        # Wait for send_task to finish (it ends when agent emits done/error)
        await send_task
        recv_task.cancel()
        agent_task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Ollama proxy (catch-all — must stay last to not shadow API routes)
# ---------------------------------------------------------------------------

def print_content(data: bytes, point_in_code: str):
    try:
        print(f"{point_in_code=} data=", json.dumps(json.loads(data.decode()), indent=2))
    except Exception:
        data_s = [d.strip() for d in data.decode().split("\n") if d.strip() != ""]
        for d in data_s:
            print(d)


@app.get("/{full_path:path}")
async def catch_all_get(
    full_path: str,
    response: Response,
    http_sess: aiohttp.ClientSession = Depends(get_http_session),
):
    async with http_sess.get("http://localhost:11434/" + full_path) as r:
        data = await r.content.read()
        print_content(data, "RESPONSE catch_all_get")
        response.status_code = r.status
        return data


@app.post("/{full_path:path}")
async def catch_all_post(
    full_path: str,
    request: Request,
    response: Response,
    http_sess: aiohttp.ClientSession = Depends(get_http_session),
):
    body = await request.body()
    print_content(body, "QUERY catch_all_post")
    async with http_sess.post("http://localhost:11434/" + full_path, data=body) as r:
        data = await r.content.read()
        print_content(data, "RESPONSE catch_all_post")
        response.status_code = r.status
        return data


@app.put("/{full_path:path}")
async def catch_all_put(
    full_path: str,
    request: Request,
    response: Response,
    http_sess: aiohttp.ClientSession = Depends(get_http_session),
):
    body = await request.body()
    print_content(body, "QUERY catch_all_put")
    async with http_sess.put("http://localhost:11434/" + full_path, data=body) as r:
        data = await r.content.read()
        response.status_code = r.status
        print_content(data, "RESPONSE catch_all_put")
        return data
