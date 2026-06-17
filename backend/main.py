import asyncio
import base64
import datetime
import functools
import io
import math
import re
import uuid
import logging
import json
import threading
from dataclasses import dataclass
from typing import Any
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Response, WebSocket, WebSocketDisconnect, UploadFile, Form
from sqlalchemy import select, delete
from database import init_db, get_db_session, AsyncSession
import loaders as ld
import tables as db
from agent.agent import AgentSession, run_agent, _find_superseded_read_file_indices
from agent.pipeline import PipelineOrchestrator
from agent.workflow_loader import load_workflow
from agent.custom_workflow import CustomWorkflowOrchestrator
from agent.compress import compress_messages, CompressionResult, Compression
from message_types import LLMMessage, TrackedMessage
from tool_result_types import ToolResult
from agent.tools import TOOL_REGISTRY, PLAN_MODE_TOOLS, CONVERSATIONAL_TOOLS, get_ollama_tool_list
from llm import backend
import whisper_pipeline

logger = logging.getLogger(__name__)


async def check_llm() -> None:
    await backend.check_or_raise()


def disable_sqlalchemy_logging():
    for name in ["sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine", "aiosqlite"]:
        log = logging.getLogger(name)
        log.disabled = True
        log.propagate = False


disable_sqlalchemy_logging()
logging.basicConfig(level=logging.DEBUG)






_whisper: whisper_pipeline.WhisperPipeline | None = None
_llm_ready: bool = False


def _load_whisper_bg() -> None:
    global _whisper
    try:
        _whisper = whisper_pipeline.load_pipeline()
    except Exception:
        logger.exception("Whisper pipeline failed to load — /api/transcribe will be unavailable")


def _load_llm_bg() -> None:
    global _llm_ready
    try:
        asyncio.run(backend.ensure_running())
        _llm_ready = True
        logger.info("LLM backend ready.")
    except Exception:
        logger.exception("LLM backend failed to start.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("Database initialized successfully.")
    threading.Thread(target=_load_llm_bg, daemon=True).start()
    threading.Thread(target=_load_whisper_bg, daemon=True).start()
    yield


app = FastAPI(title="LLM Chat Backend", lifespan=lifespan)


@app.get("/api/status")
async def get_status():
    return {"llm": _llm_ready, "whisper": _whisper is not None}



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
) -> list[LLMMessage]:
    """Build the message list sent to the LLM. Prepends system prompt from file if prompt_id (slug) given."""
    messages: list[LLMMessage] = []
    if prompt_id is not None:
        prompt_path = _PROMPTS_DIR / f"{prompt_id}.yaml"
        if prompt_path.exists():
            prompt_data = _load_prompt_file(prompt_path)
            today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
            content = f"Today's date: {today}\n\n{prompt_data['content']}"
            messages.append({"role": "system", "content": content})

    # Batch-fetch image attachments for all branch messages
    branch_ids = [m.id for m in branch]
    img_rows = (await sess.execute(
        select(db.MessageImageAttachment, db.Image)
        .join(db.Image, db.MessageImageAttachment.image_id == db.Image.id)
        .where(db.MessageImageAttachment.message_id.in_(branch_ids))
        .order_by(db.MessageImageAttachment.position)
    )).all() if branch_ids else []
    images_by_msg: dict[str, list] = {}
    for att, img in img_rows:
        images_by_msg.setdefault(att.message_id, []).append(img)

    for m in branch:
        if m.context_excluded:
            if m.compressed_summary:
                try:
                    original: ToolResult = json.loads(m.content)
                except (json.JSONDecodeError, ValueError, TypeError):
                    original = {"tool": "tool", "status": "unknown"}
                messages.append({"role": "tool", "content": json.dumps({
                    "tool": original.get("tool", "tool"),
                    "status": "compressed",
                    "summary": m.compressed_summary,
                    "tool_call_id": original.get("tool_call_id", ""),
                })})
            continue
        if m.content is None or m.content.strip() == "":
            continue
        imgs = images_by_msg.get(m.id, [])
        if imgs:
            multipart_content: list[dict] = [{"type": "text", "text": m.content}]
            for img in imgs:
                multipart_content.append({"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"}})
            messages.append({"role": m.role, "content": multipart_content})
        else:
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
        "compressed_summary": m.compressed_summary,
        "compression_label": m.compression_label,
        "log_message": m.log_message,
        "tool_calls": json.loads(m.tool_calls) if m.tool_calls else None,
        "is_degenerate": bool(m.is_degenerate),
        "compressed_token_count": m.compressed_token_count,
    }


async def _delete_attachments_and_gc_images(sess: AsyncSession, msg_ids: list[str]) -> None:
    """Delete MessageImageAttachment rows for msg_ids, then GC Image rows with no remaining refs."""
    if not msg_ids:
        return
    # Collect image_ids that are about to lose a reference
    candidate_ids = list((await sess.scalars(
        select(db.MessageImageAttachment.image_id)
        .where(db.MessageImageAttachment.message_id.in_(msg_ids))
    )).all())
    await sess.execute(delete(db.MessageImageAttachment).where(db.MessageImageAttachment.message_id.in_(msg_ids)))
    if candidate_ids:
        # Delete images that are now unreferenced
        still_referenced = set((await sess.scalars(
            select(db.MessageImageAttachment.image_id)
            .where(db.MessageImageAttachment.image_id.in_(candidate_ids))
        )).all())
        orphaned = [iid for iid in candidate_ids if iid not in still_referenced]
        if orphaned:
            await sess.execute(delete(db.Image).where(db.Image.id.in_(orphaned)))


async def _fetch_images_by_msg(sess: AsyncSession, msg_ids: list[str]) -> dict[str, list[dict]]:
    """Return {message_id: [{id, mime_type}, ...]} for all given message IDs."""
    if not msg_ids:
        return {}
    rows = (await sess.execute(
        select(db.MessageImageAttachment, db.Image)
        .join(db.Image, db.MessageImageAttachment.image_id == db.Image.id)
        .where(db.MessageImageAttachment.message_id.in_(msg_ids))
        .order_by(db.MessageImageAttachment.position)
    )).all()
    result: dict[str, list[dict]] = {}
    for att, img in rows:
        result.setdefault(att.message_id, []).append({"id": img.id, "mime_type": img.mime_type})
    return result


def _enrich_branch(
    path: list[db.Message],
    all_msgs: list[db.Message],
    images_by_msg: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Return _msg_dict for each message in path, enriched with sibling navigation metadata."""
    children_by_parent: dict[str | None, list[db.Message]] = {}
    for m in all_msgs:
        children_by_parent.setdefault(m.parent_id, []).append(m)
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda m: m.created_at)

    images = images_by_msg or {}
    result = []
    for m in path:
        siblings = children_by_parent.get(m.parent_id, [m])
        idx = next((i for i, s in enumerate(siblings) if s.id == m.id), 0)
        count = len(siblings)
        result.append({
            **_msg_dict(m),
            "images": images.get(m.id, []),
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


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, sess: AsyncSession = Depends(get_db_session)):
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == conversation_id))).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": conv.id,
        "title": conv.title,
        "settings": conv.settings,
        "created_at": conv.created_at,
        "active_message_id": conv.active_message_id,
    }


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
    msg_ids = list((await sess.scalars(
        select(db.Message.id).where(db.Message.conversation_id == id)
    )).all())
    await _delete_attachments_and_gc_images(sess, msg_ids)
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
    if body.working_directory is not None:
        await _upsert_app_setting(sess, "last_working_directory", body.working_directory)
    return body


@app.get("/api/app-settings/{key}")
async def get_app_setting(key: str, sess: AsyncSession = Depends(get_db_session)):
    row = await sess.get(db.AppSettings, key)
    return {"key": key, "value": row.value if row else None}


@app.put("/api/app-settings/{key}")
async def put_app_setting(key: str, body: ld.AppSettingUpdate, sess: AsyncSession = Depends(get_db_session)):
    await _upsert_app_setting(sess, key, body.value)
    return {"key": key, "value": body.value}


async def _upsert_app_setting(sess: AsyncSession, key: str, value: str | None) -> None:
    row = await sess.get(db.AppSettings, key)
    if row is None:
        sess.add(db.AppSettings(key=key, value=value))
    else:
        row.value = value


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
    images_by_msg = await _fetch_images_by_msg(sess, [m.id for m in path])
    return {
        "active_message_id": leaf_id,
        "messages": _enrich_branch(path, all_msgs, images_by_msg),
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
    images_by_msg = await _fetch_images_by_msg(sess, [m.id for m in path])
    return _enrich_branch(path, all_msgs, images_by_msg)


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
    existing = await sess.get(db.Message, msg_id)
    if existing is not None:
        return {"id": msg_id, "parent_id": existing.parent_id}

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
            log_message=message.log_message,
            tool_calls=json.dumps(message.tool_calls) if message.tool_calls is not None else None,
            is_degenerate=message.is_degenerate,
        )
    )
    for position, image_id in enumerate(message.image_ids):
        sess.add(db.MessageImageAttachment(message_id=msg_id, image_id=image_id, position=position))
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
    # Copy image attachments from the original message to the new branch
    orig_atts = (await sess.scalars(
        select(db.MessageImageAttachment).where(db.MessageImageAttachment.message_id == original.id)
    )).all()
    for att in orig_atts:
        sess.add(db.MessageImageAttachment(message_id=new_id, image_id=att.image_id, position=att.position))
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

        await _delete_attachments_and_gc_images(sess, list(to_delete))
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

        await _delete_attachments_and_gc_images(sess, [msg_id])
        await sess.execute(delete(db.Message).where(db.Message.id == msg_id))
        return {"deleted": [msg_id]}


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}

# Qwen3.5-9B: patch_size=16, spatial_merge_size=2 → 32px per token side
_IMAGE_PIXELS_PER_TOKEN = 32 * 32


def _image_dimensions(data: bytes) -> tuple[int, int]:
    from PIL import Image as PILImage
    with PILImage.open(io.BytesIO(data)) as img:
        return img.width, img.height


def _image_token_count(width: int, height: int) -> int:
    return math.ceil(width / 32) * math.ceil(height / 32)


@app.post("/api/images")
async def upload_image(file: UploadFile, sess: AsyncSession = Depends(get_db_session)):
    if file.content_type not in _ALLOWED_IMAGE_MIME:
        raise HTTPException(415, f"Unsupported image type: {file.content_type}")
    data = await file.read()
    encoded = base64.b64encode(data).decode("ascii")
    width, height = _image_dimensions(data)
    image_id = str(uuid.uuid4())
    sess.add(db.Image(id=image_id, mime_type=file.content_type, data=encoded, width=width, height=height, created_at=_now()))
    await sess.flush()
    return {"id": image_id, "mime_type": file.content_type}


@app.get("/api/images/{image_id}")
async def get_image(image_id: str, sess: AsyncSession = Depends(get_db_session)):
    img = await sess.get(db.Image, image_id)
    if img is None:
        raise HTTPException(404)
    return Response(content=base64.b64decode(img.data), media_type=img.mime_type)


# ---------------------------------------------------------------------------
# System prompts (file-based — backend/prompts/*.yaml)
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DUMMY_USER: list[LLMMessage] = [{"role": "user", "content": "."}]


def _slugify(name: str) -> str:
    """Convert a display name to a safe kebab-case filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "prompt"


def _load_prompt_file(path: Path) -> dict:
    """Parse a prompt YAML file and return its data dict."""
    import yaml as _yaml
    data = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "id": path.stem,
        "name": data.get("name") or path.stem,
        "category": data.get("category") or "general",
        "content": data.get("content") or "",
        "is_default": bool(data.get("is_default")),
        "token_count": data.get("token_count") or None,
    }


def _write_prompt_file(path: Path, name: str, category: str, content: str, is_default: bool, token_count: int | None) -> None:
    """Write a prompt YAML file."""
    import yaml as _yaml
    data = {
        "name": name,
        "category": category,
        "is_default": is_default,
        "token_count": token_count,
        "content": content,
    }
    path.write_text(_yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False), encoding="utf-8")


async def _compute_prompt_token_count(content: str) -> int:
    """Count tokens contributed by this system prompt (with vs without)."""
    tools: list = []
    system_msg: LLMMessage = {"role": "system", "content": content}
    with_prompt = await backend.count_tokens([system_msg] + _DUMMY_USER, tools)
    baseline = await backend.count_tokens(_DUMMY_USER, tools)
    return with_prompt - baseline


@app.get("/api/system-prompts")
async def list_system_prompts():
    """List all system prompt YAML files from backend/prompts/."""
    _PROMPTS_DIR.mkdir(exist_ok=True)
    return [_load_prompt_file(f) for f in sorted(_PROMPTS_DIR.glob("*.yaml"))]


@app.post("/api/system-prompts")
async def create_system_prompt(body: ld.NewSystemPrompt):
    """Create a new system prompt YAML file. Slug is derived from the name."""
    _PROMPTS_DIR.mkdir(exist_ok=True)
    slug = _slugify(body.name)
    path = _PROMPTS_DIR / f"{slug}.yaml"
    if path.exists():
        # Avoid collision — append a suffix
        suffix = 2
        while (_PROMPTS_DIR / f"{slug}-{suffix}.yaml").exists():
            suffix += 1
        slug = f"{slug}-{suffix}"
        path = _PROMPTS_DIR / f"{slug}.yaml"
    token_count_value = await _compute_prompt_token_count(body.content)
    _write_prompt_file(path, body.name, body.category, body.content, body.is_default, token_count_value)
    return _load_prompt_file(path)


@app.put("/api/system-prompts/{slug}")
async def update_system_prompt(slug: str, body: ld.UpdateSystemPrompt):
    """Update an existing system prompt YAML file by slug."""
    path = _PROMPTS_DIR / f"{slug}.yaml"
    if not path.exists():
        raise HTTPException(404)
    current = _load_prompt_file(path)
    name = body.name if body.name is not None else current["name"]
    category = body.category if body.category is not None else current["category"]
    is_default = body.is_default if body.is_default is not None else current["is_default"]
    content = body.content if body.content is not None else current["content"]
    token_count = current["token_count"]
    if body.content is not None:
        token_count = await _compute_prompt_token_count(content)
    _write_prompt_file(path, name, category, content, is_default, token_count)
    return _load_prompt_file(path)


@app.delete("/api/system-prompts/{slug}")
async def delete_system_prompt(slug: str):
    """Delete a system prompt YAML file by slug."""
    path = _PROMPTS_DIR / f"{slug}.yaml"
    if path.exists():
        path.unlink()
    return ""


# ---------------------------------------------------------------------------
# Agents (file-based — backend/agents/*.yaml)
# ---------------------------------------------------------------------------

_AGENTS_DIR = Path(__file__).parent / "agents"


def _load_agent_file(path: Path) -> dict:
    """Parse an agent YAML file and return its data dict for the API."""
    import yaml as _yaml
    data = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "name": data.get("name") or path.stem,
        "description": data.get("description") or "",
        "system_prompt": data.get("system_prompt") or "",
        "tools": data.get("tools") or [],
        "finish_tool": data.get("finish_tool") or "finish_task",
        "max_iterations": data.get("max_iterations") if data.get("max_iterations") is not None else None,
        "inject_turn_reminders": bool(data.get("inject_turn_reminders")),
    }


def _write_agent_file(path: Path, data: dict) -> None:
    """Write an agent YAML file from the API data dict."""
    import yaml as _yaml
    path.write_text(_yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False), encoding="utf-8")


@app.get("/api/agents")
async def list_agents():
    """List all agent YAML files from backend/agents/."""
    _AGENTS_DIR.mkdir(exist_ok=True)
    return [_load_agent_file(f) for f in sorted(_AGENTS_DIR.glob("*.yaml"))]


@app.post("/api/agents")
async def create_agent(body: ld.NewAgent):
    """Create a new agent YAML file. Filename is derived from the name."""
    _AGENTS_DIR.mkdir(exist_ok=True)
    slug = _slugify(body.name)
    path = _AGENTS_DIR / f"{slug}.yaml"
    if path.exists():
        raise HTTPException(409, detail=f"Agent '{slug}' already exists")
    data = {
        "name": body.name,
        "description": body.description,
        "system_prompt": body.system_prompt,
        "tools": body.tools,
        "finish_tool": body.finish_tool,
        "max_iterations": body.max_iterations,
        "inject_turn_reminders": body.inject_turn_reminders,
    }
    _write_agent_file(path, data)
    return _load_agent_file(path)


@app.put("/api/agents/{name}")
async def update_agent(name: str, body: ld.UpdateAgent):
    """Update an existing agent YAML file."""
    path = _AGENTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(404)
    current = _load_agent_file(path)
    data = {
        "name": body.name if body.name is not None else current["name"],
        "description": body.description if body.description is not None else current["description"],
        "system_prompt": body.system_prompt if body.system_prompt is not None else current["system_prompt"],
        "tools": body.tools if body.tools is not None else current["tools"],
        "finish_tool": body.finish_tool if body.finish_tool is not None else current["finish_tool"],
        "max_iterations": body.max_iterations if body.max_iterations is not None else current["max_iterations"],
        "inject_turn_reminders": body.inject_turn_reminders if body.inject_turn_reminders is not None else current["inject_turn_reminders"],
    }
    _write_agent_file(path, data)
    return _load_agent_file(path)


@app.delete("/api/agents/{name}")
async def delete_agent(name: str):
    """Delete an agent YAML file."""
    path = _AGENTS_DIR / f"{name}.yaml"
    if path.exists():
        path.unlink()
    return ""


@app.get("/api/finish-tools")
async def list_finish_tools():
    """List builtin finish tool names available for agents."""
    from agent.workflow_loader import _BUILTIN_FINISH_TOOL_CLASSES
    return list(_BUILTIN_FINISH_TOOL_CLASSES.keys())


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
# Workflows
# ---------------------------------------------------------------------------

@app.get("/api/workflows")
async def list_workflows():
    """List available workflow definitions from the backend/workflows/ directory."""
    import yaml
    workflows_dir = Path(__file__).parent / "workflows"
    if not workflows_dir.exists():
        return []
    results = []
    candidates: list[Path] = sorted(workflows_dir.glob("*.yaml"))
    candidates += sorted(
        sub / "workflow.yaml"
        for sub in workflows_dir.iterdir()
        if sub.is_dir() and (sub / "workflow.yaml").exists()
    )
    for yaml_file in candidates:
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            results.append({
                "name": data.get("name", yaml_file.parent.stem if yaml_file.name == "workflow.yaml" else yaml_file.stem),
                "description": data.get("description", ""),
            })
        except Exception:
            pass
    return results


# Agent tools
# ---------------------------------------------------------------------------

@app.get("/api/agent/tools")
async def list_agent_tools():
    from agent.tools.base import TOOL_FRAMEWORK_OVERHEAD, STACKING_OVERHEAD_PER_ADDITIONAL_TOOL
    always_active = [
        {
            "name": t.name,
            "description": t.description,
            "token_count": t.token_count,
            "mode_context": mode_context,
        }
        for t, mode_context in [
            (CONVERSATIONAL_TOOLS["ask_user_question"], "Included in Standard and Plan modes"),
            (PLAN_MODE_TOOLS["propose_plan"], "Included in Plan mode only"),
        ]
    ]
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
        "always_active_tools": always_active,
    }




# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

@app.post("/api/conversations/{id}/count-tokens")
async def count_conversation_tokens(id: str, sess: AsyncSession = Depends(get_db_session)):
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)

    settings = _parse_conv_settings(conv)
    messages = await _build_inference_context(branch, settings.active_prompt_id, sess)
    tool_names = settings.active_tool_names if conv.settings is not None else list(TOOL_REGISTRY.keys())
    tools = get_ollama_tool_list(tool_names)
    token_count_value = await backend.count_tokens(messages, tools)

    # Add image tokens using real dimensions (patch=16, merge=2 → 32px/token side)
    img_rows = (await sess.execute(
        select(db.Image.width, db.Image.height)
        .join(db.MessageImageAttachment, db.MessageImageAttachment.image_id == db.Image.id)
        .where(db.MessageImageAttachment.message_id.in_([m.id for m in branch]))
    )).all()
    image_tokens = sum(_image_token_count(w or 0, h or 0) for w, h in img_rows)
    token_count_value += image_tokens

    last_id: str | None = None
    if branch:
        branch[-1].token_count = token_count_value
        last_id = branch[-1].id
        await sess.flush()

    return {"token_count": token_count_value, "message_id": last_id}


@app.post("/api/conversations/{id}/compress")
async def compress_conversation(
    id: str,
    protect_last: bool = False,
    is_mid_run: bool = False,
    sess: AsyncSession = Depends(get_db_session),
):
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)

    # Candidates: uncompressed tool messages on the active branch
    candidates = [m for m in branch if m.role == "tool" and not m.context_excluded]
    if not candidates:
        return {"compressions": [], "new_summary": ""}

    # Use the last 3 user messages as goal context (more context when mid-run)
    user_messages = [m.content for m in reversed(branch) if m.role == "user"][:3]
    user_message = "\n---\n".join(reversed(user_messages)) if user_messages else ""

    all_dicts: list[TrackedMessage] = [{"id": m.id, "role": m.role, "content": m.content, "thinking": m.thinking} for m in branch]
    candidate_dicts: list[TrackedMessage] = [{"id": m.id, "role": m.role, "content": m.content, "thinking": m.thinking} for m in candidates]

    compression_result = await compress_messages(
        candidate_dicts, 
        all_dicts, 
        user_message, 
        conversation_summary=None, 
        backend=backend,
        protect_last=protect_last, 
        is_mid_run=is_mid_run,
    )

    for c in compression_result.compressions:
        msg = next((m for m in candidates if m.id == c.message_id), None)
        if msg is not None:
            msg.context_excluded = True
            msg.exclusion_reason = "compressed"
            msg.compressed_summary = c.compressed_summary
            msg.compression_label = c.compression_label
            try:
                original: ToolResult = json.loads(msg.content)
            except (json.JSONDecodeError, ValueError, TypeError):
                original = {"tool": "tool", "status": "unknown"}
            compressed_content = json.dumps({
                "tool": original.get("tool", "tool"),
                "status": "compressed",
                "summary": c.compressed_summary,
                "tool_call_id": original.get("tool_call_id", ""),
            })
            msg.compressed_token_count = await backend.count_text_tokens(compressed_content)

    await sess.flush()

    compressed_branch = _build_active_branch_path(
        list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all()),
        conv.active_message_id,
    )
    compressed_dicts: list[LLMMessage] = [{"role": m.role, "content": m.compressed_summary if m.compressed_summary else m.content, "thinking": m.thinking} for m in compressed_branch]
    tools_list = get_ollama_tool_list([tool.name for tool in TOOL_REGISTRY.values()])
    ctx_tokens = await backend.count_tokens(backend.prepare_messages(compressed_dicts), tools_list)

    return {
      "compressions": compression_result.compressions, 
      "new_summary": compression_result.new_summary,
      "ctx_tokens": ctx_tokens,
    }


# ---------------------------------------------------------------------------
# Agent WebSocket
# ---------------------------------------------------------------------------

_PLAN_EXCLUDED_TOOLS = frozenset({"write_file", "edit_file", "run_shell"})


_VALID_MODES = frozenset({"standard", "auto", "plan", "yolo"})


async def _ws_receive_messages_from_frontend(websocket: WebSocket, session: AgentSession, agent_task: asyncio.Task) -> None:
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

_STT_EXAMPLES_FR = [
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

_STT_EXAMPLES_EN = [
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
    import aiohttp
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


@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile,
    language: str | None = Form(default=None),
):
    if _whisper is None:
        raise HTTPException(503, "Whisper pipeline is still loading, try again in a moment")
    data = await audio.read()
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, whisper_pipeline.transcribe, _whisper, data, language
    )
    logger.info("STT raw transcript: %r", text)
    return {"text": text}


@app.post("/api/correct")
async def correct_stt(req: ld.CorrectRequest):
    if not req.text:
        return {"text": req.text}
    corrected = await _correct_stt(req.text, req.language)
    logger.info("STT corrected: %r", corrected)
    return {"text": corrected}


@dataclass
class ConvBranch:
    """Result of loading a conversation and its active message branch from the database."""
    conv: db.Conversation | None
    branch: list[db.Message]


@dataclass
class ToolSet:
    """Tool list and optional injected extras assembled for a given conversation mode."""
    tools: list[dict]
    extra_tools: dict[str, Any] | None


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
        for injected_tool in CONVERSATIONAL_TOOLS.values():
            tools.append({"type": "function", "function": injected_tool.to_ollama_schema()})
        return ToolSet(tools=tools, extra_tools=dict(CONVERSATIONAL_TOOLS))
    else:
        tools = get_ollama_tool_list(active_tool_names)
        return ToolSet(tools=tools, extra_tools=None)


async def _apply_db_compressions(
    sess: AsyncSession, messages: list[LLMMessage], conv_id: str
) -> list[LLMMessage]:
    """Patch in-memory tool messages with their compressed summaries after mid-run compression.
    Matches by tool_call_id so assistant messages (not persisted to DB) are preserved."""
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
            pass

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        raw_content = msg.get("content")
        if not isinstance(raw_content, str):
            continue
        try:
            content_dict: ToolResult = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError):
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
    tools: list[dict],
    extra_tools: dict[str, Any] | None,
) -> asyncio.Task:
    """Dispatch either a named workflow or the standard agentic loop as an asyncio task."""
    if workflow_name is not None:
        workflows_dir = Path(__file__).parent / "workflows"
        flat_path = workflows_dir / f"{workflow_name}.yaml"
        dir_path = workflows_dir / workflow_name
        workflow_path = flat_path if flat_path.exists() else dir_path
        workflow_def = load_workflow(workflow_path)
        orchestrator = CustomWorkflowOrchestrator(workflow_def, working_directory, tools)
        return asyncio.create_task(orchestrator.run(session, user_message, messages))
    else:
        return asyncio.create_task(run_agent(session, messages, tools, working_directory, extra_tools))


async def _run_agent_event_loop(
    websocket: WebSocket, session: AgentSession, agent_task: asyncio.Task
) -> None:
    """Drive the agent session over WebSocket until the agent emits done or error.
    Forwards outbound events to the client and receives inbound control messages (confirm, abort, set_mode…)."""
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


@app.websocket("/api/agent/ws")
async def agent_websocket(websocket: WebSocket, sess: AsyncSession = Depends(get_db_session)):
    """WebSocket endpoint for the main agentic loop.
    Client sends one init message, then exchanges control messages while the agent streams events back."""
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
            user_message, messages, tool_set.tools, tool_set.extra_tools,
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
            pass


@app.websocket("/api/agent/pipeline/ws")
async def pipeline_websocket(websocket: WebSocket, sess: AsyncSession = Depends(get_db_session)):
    """
    WebSocket endpoint for the pipeline orchestrator.
    Parallel alternative to /api/agent/ws for comparison.
    First client message: {"message": "...", "conversation_id": "optional"}
    """
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
