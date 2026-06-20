"""Conversation, message, and image CRUD endpoints."""
import base64
import io
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from sqlalchemy import select, delete

from database import get_db_session, AsyncSession
import tables as db
import loaders as ld
from agent.tools import TOOL_REGISTRY
from conv_helpers import (
    _now,
    _build_active_branch_path,
    _find_deepest_leaf,
    _enrich_branch,
    _fetch_images_by_msg,
    _delete_attachments_and_gc_images,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# App settings helpers
# ---------------------------------------------------------------------------

async def _upsert_app_setting(sess: AsyncSession, key: str, value: str | None) -> None:
    """Insert or update a single AppSettings row."""
    row = await sess.get(db.AppSettings, key)
    if row is None:
        sess.add(db.AppSettings(key=key, value=value))
    else:
        row.value = value


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@router.get("/api/conversations")
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


@router.get("/api/conversations/{conversation_id}")
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


@router.post("/api/conversations")
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


@router.delete("/api/conversations/{id}")
async def delete_conversation(id: str, sess: AsyncSession = Depends(get_db_session)):
    msg_ids = list((await sess.scalars(
        select(db.Message.id).where(db.Message.conversation_id == id)
    )).all())
    await _delete_attachments_and_gc_images(sess, msg_ids)
    await sess.execute(delete(db.Message).where(db.Message.conversation_id == id))
    await sess.execute(delete(db.Conversation).where(db.Conversation.id == id))
    return ""


@router.put("/api/conversations/{id}/settings")
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


@router.get("/api/app-settings/{key}")
async def get_app_setting(key: str, sess: AsyncSession = Depends(get_db_session)):
    row = await sess.get(db.AppSettings, key)
    return {"key": key, "value": row.value if row else None}


@router.put("/api/app-settings/{key}")
async def put_app_setting(key: str, body: ld.AppSettingUpdate, sess: AsyncSession = Depends(get_db_session)):
    await _upsert_app_setting(sess, key, body.value)
    return {"key": key, "value": body.value}


@router.put("/api/conversations/{id}/active-branch")
async def set_active_branch(
    id: str, body: ld.BranchNavigation, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))
    ).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list(
        (await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all()
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

@router.get("/api/conversations/{id}/messages")
async def get_conversation_messages(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))
    ).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list(
        (await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all()
    )
    path = _build_active_branch_path(all_msgs, conv.active_message_id)
    images_by_msg = await _fetch_images_by_msg(sess, [m.id for m in path])
    return _enrich_branch(path, all_msgs, images_by_msg)


@router.get("/api/conversations/{id}/tree")
async def get_conversation_tree(
    id: str, sess: AsyncSession = Depends(get_db_session)
):
    conv = (
        await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))
    ).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list(
        (await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all()
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


@router.post("/api/messages")
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


@router.patch("/api/messages/{id}/token-count")
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


@router.put("/api/messages/{id}/branch")
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


@router.delete("/api/conversations/{conv_id}/messages/{msg_id}")
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
    """Return (width, height) of the given image bytes."""
    from PIL import Image as PILImage
    with PILImage.open(io.BytesIO(data)) as img:
        return img.width, img.height


@router.post("/api/images")
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


@router.get("/api/images/{image_id}")
async def get_image(image_id: str, sess: AsyncSession = Depends(get_db_session)):
    img = await sess.get(db.Image, image_id)
    if img is None:
        raise HTTPException(404)
    return Response(content=base64.b64decode(img.data), media_type=img.mime_type)
