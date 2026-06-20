"""Shared helpers for conversation branch traversal, inference context building, and session types.

Used by all routers that need to read or manipulate the conversation tree.
"""
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select, delete
from database import AsyncSession
import tables as db
import loaders as ld
from agent.tools.base import BaseTool, ToolDict
from message_types import LLMMessage
from tool_result_types import ToolResult
from llm import backend


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _slugify(name: str) -> str:
    """Convert a display name to a safe kebab-case filename stem."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "prompt"


def _find_superseded_read_file_indices(pairs: list[tuple[str, str]]) -> list[int]:
    """Given (role, content_json) pairs, return indices of superseded read_file results."""
    path_indices: dict[str, list[int]] = {}
    for i, (role, content_str) in enumerate(pairs):
        if role != "tool":
            continue
        try:
            content: ToolResult = json.loads(content_str or "")
        except (json.JSONDecodeError, ValueError):
            continue
        if content.get("tool") != "read_file" or content.get("status") != "success":
            continue
        path_indices.setdefault(content.get("path", ""), []).append(i)
    return [i for indices in path_indices.values() for i in indices[:-1]]


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


_PROMPTS_DIR = Path(__file__).parent / "prompts"


async def _build_inference_context(
    branch: list[db.Message],
    prompt_id: str | None,
    sess: AsyncSession,
) -> list[LLMMessage]:
    """Build the message list sent to the LLM. Prepends system prompt from file if prompt_id (slug) given."""
    import yaml as _yaml

    messages: list[LLMMessage] = []
    if prompt_id is not None:
        prompt_path = _PROMPTS_DIR / f"{prompt_id}.yaml"
        if prompt_path.exists():
            data = _yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
            content = f"Today's date: {today}\n\n{data.get('content') or ''}"
            messages.append({"role": "system", "content": content})

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

    non_excluded = [m for m in branch if not m.context_excluded]
    last_assistant = next((m for m in reversed(non_excluded) if m.role == "assistant"), None)
    interrupted_id = last_assistant.id if (last_assistant is not None and last_assistant.tool_calls is not None) else None

    for m in branch:
        if m.context_excluded:
            if m.compressed_summary:
                try:
                    original: ToolResult = json.loads(m.content)
                except (json.JSONDecodeError, ValueError, TypeError):
                    original = {"tool": "tool", "status": "unknown"}
                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "tool": original.get("tool", "tool"),
                        "status": "compressed",
                        "summary": m.compressed_summary,
                        "tool_call_id": original.get("tool_call_id", ""),
                    })
                })
            continue
        if m.content is None or m.content.strip() == "":
            continue
        if m.role == "context_summary":
            messages.append({"role": "user", "content": m.content})
            continue
        content = m.content or ""
        if m.id == interrupted_id:
            if m.thinking is not None and not content.startswith("<think>"):
                content = f"<think>{m.thinking}</think>{content}"
            try:
                stored_calls = json.loads(m.tool_calls)
                tool_calls_for_context = [
                    {
                        "id": tc.get("id", f"tc-{i}"),
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc.get("args", {})},
                    }
                    for i, tc in enumerate(stored_calls)
                ]
            except (json.JSONDecodeError, ValueError, KeyError):
                tool_calls_for_context = []
        else:
            tool_calls_for_context = []

        msg: LLMMessage = {"role": m.role, "content": content}
        if len(tool_calls_for_context) > 0:
            msg["tool_calls"] = tool_calls_for_context
        imgs = images_by_msg.get(m.id, [])
        if imgs:
            multipart_content: list[dict] = [{"type": "text", "text": content}]
            for img in imgs:
                multipart_content.append({"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"}})
            msg["content"] = multipart_content
        messages.append(msg)
    return messages


def _deduplicate_branch_file_reads(branch: list[db.Message]) -> None:
    """Mark superseded read_file tool messages as context_excluded in place."""
    pairs = [(m.role, m.content or "") for m in branch]
    for i in _find_superseded_read_file_indices(pairs):
        branch[i].context_excluded = True
        branch[i].exclusion_reason = "file_superseded"


def _parse_conv_settings(conv: db.Conversation) -> ld.ConversationSettings:
    """Parse conversation settings JSON, returning defaults on parse failure."""
    try:
        return ld.ConversationSettings.model_validate_json(conv.settings or "{}")
    except Exception:
        return ld.ConversationSettings()


def _msg_dict(m: db.Message) -> dict[str, Any]:
    """Serialize a Message row to a dict for API responses."""
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
    candidate_ids = list((await sess.scalars(
        select(db.MessageImageAttachment.image_id)
        .where(db.MessageImageAttachment.message_id.in_(msg_ids))
    )).all())
    await sess.execute(delete(db.MessageImageAttachment).where(db.MessageImageAttachment.message_id.in_(msg_ids)))
    if candidate_ids:
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
) -> list[dict[str, Any]]:
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


@dataclass
class ConvBranch:
    """Result of loading a conversation and its active message branch from the database."""
    conv: db.Conversation | None
    branch: list[db.Message]


@dataclass
class ToolSet:
    """Tool list and optional injected extras assembled for a given conversation mode."""
    tools: list[ToolDict]
    extra_tools: dict[str, BaseTool] | None
