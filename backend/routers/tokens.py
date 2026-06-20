"""Token counting, context inspection, and compression endpoints."""
import json
import logging
import math

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from database import get_db_session, AsyncSession
import tables as db
from agent.compress import (
    Compression,
    compress_messages,
    apply_working_memory,
)
from agent.tools import TOOL_REGISTRY, get_ollama_tool_list
from conv_helpers import (
    _build_active_branch_path,
    _build_inference_context,
    _deduplicate_branch_file_reads,
    _parse_conv_settings,
)
from llm import backend
from message_types import TrackedMessage
from tool_result_types import ToolResult

router = APIRouter()

logger = logging.getLogger(__name__)

# Qwen3.5-9B: patch_size=16, spatial_merge_size=2 → 32px per token side
_IMAGE_PIXELS_PER_TOKEN = 32 * 32


def _image_token_count(width: int, height: int) -> int:
    """Estimate the number of vision tokens for an image of the given dimensions."""
    return math.ceil(width / 32) * math.ceil(height / 32)



@router.post("/api/conversations/{id}/count-tokens")
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


@router.post("/api/conversations/{id}/debug-tokens")
async def debug_conversation_tokens(
    id: str,
    sess: AsyncSession = Depends(get_db_session),
):
    """Count tokens per message and log a detailed breakdown to stdout."""
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))).first()
    if conv is None:
        raise HTTPException(404)

    all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)

    in_context_total = 0
    raw_total = 0
    logger.info("=== DEBUG TOKEN BREAKDOWN conv=%s (%d messages) ===", id, len(branch))
    for m in branch:
        content_text = m.content or ""
        thinking_text = m.thinking or ""
        summary_text = m.compressed_summary or ""

        content_tokens = await backend.count_text_tokens(content_text) if content_text else 0
        thinking_tokens = await backend.count_text_tokens(thinking_text) if thinking_text else 0
        summary_tokens = await backend.count_text_tokens(summary_text) if summary_text else 0

        raw_tokens = content_tokens + thinking_tokens

        if m.context_excluded:
            in_context_tokens = summary_tokens
        else:
            in_context_tokens = raw_tokens

        in_context_total += in_context_tokens
        raw_total += raw_tokens

        label = f"[{m.role}]"
        if m.context_excluded:
            label += "[excl]"
        if m.compressed_summary:
            label += "[cmp]"
        logger.info(
            "  %s id=%.8s raw=%d (content=%d thinking=%d) summary=%d  in-ctx=%d",
            label, m.id, raw_tokens, content_tokens, thinking_tokens, summary_tokens, in_context_tokens,
        )
    logger.info("=== TOTAL raw=%d  in-context=%d (+ system/tools overhead not counted) ===", raw_total, in_context_total)


@router.post("/api/conversations/{id}/debug-context")
async def debug_conversation_context(
    id: str,
    sess: AsyncSession = Depends(get_db_session),
):
    """Log the exact message list the LLM would see for this conversation (prepared wire format)."""
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))).first()
    if conv is None:
        raise HTTPException(404)

    settings = _parse_conv_settings(conv)
    all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)
    _deduplicate_branch_file_reads(branch)
    messages = await _build_inference_context(branch, settings.active_prompt_id, sess)
    prepared = backend.prepare_messages(messages)

    logger.info("=== DEBUG CONTEXT conv=%s (%d prepared messages) ===", id, len(prepared))
    total_tokens = 0
    for m in prepared:
        role = m.get("role", "?")
        raw_content = m.get("content") or ""
        if isinstance(raw_content, list):
            text_parts = [p.get("text", "") for p in raw_content if isinstance(p, dict) and p.get("type") == "text"]
            image_count = sum(1 for p in raw_content if isinstance(p, dict) and p.get("type") == "image_url")
            display_content = " ".join(text_parts) + (f" [+{image_count} image(s)]" if image_count else "")
            count_text = " ".join(text_parts)
        else:
            display_content = raw_content
            count_text = raw_content

        tokens = await backend.count_text_tokens(count_text) if count_text else 0
        total_tokens += tokens

        if role == "system":
            logger.info("  [system] %dt  %s", tokens, display_content[:200].replace("\n", " "))
        elif role == "user":
            logger.info("  [user] %dt  %s", tokens, display_content[:300].replace("\n", " "))
        elif role == "tool":
            try:
                tool_data: ToolResult = json.loads(raw_content)
                tool_name = tool_data.get("tool", "?")
                status = tool_data.get("status", "?")
                path = tool_data.get("path", "")
                if status == "evicted":
                    logger.info("  [tool] %dt  %s %s [evicted]", tokens, tool_name, path)
                elif status == "compressed":
                    summary = (tool_data.get("summary") or "")[:120]
                    logger.info("  [tool] %dt  %s %s [compressed] %s", tokens, tool_name, path, summary.replace("\n", " "))
                elif tool_name == "read_file":
                    logger.info("  [tool] %dt  read_file %s [%s]", tokens, path, status)
                elif tool_name in ("list_directory", "glob_files", "grep_files"):
                    pattern = tool_data.get("pattern") or tool_data.get("glob_pattern") or ""
                    logger.info("  [tool] %dt  %s %s %s", tokens, tool_name, path, pattern)
                else:
                    logger.info("  [tool] %dt  %s [%s]", tokens, tool_name, status)
            except (json.JSONDecodeError, ValueError):
                logger.info("  [tool] %dt  %s", tokens, display_content[:120].replace("\n", " "))
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
                logger.info("  [assistant] %dt  %d tool call(s): %s", tokens, len(tool_calls), names)
            else:
                logger.info("  [assistant] %dt  %s", tokens, display_content[:200].replace("\n", " "))
        else:
            logger.info("  [%s] %dt  %s", role, tokens, display_content[:120].replace("\n", " "))
    logger.info("=== END DEBUG CONTEXT  total=%dt (+ system/tools overhead not counted) ===", total_tokens)


@router.get("/api/conversations/{id}/ctx-tokens")
async def get_conversation_ctx_tokens(
    id: str,
    sess: AsyncSession = Depends(get_db_session),
):
    """Return the actual current context token count for the conversation."""
    conv = (await sess.scalars(select(db.Conversation).where(db.Conversation.id == id))).first()
    if conv is None:
        raise HTTPException(404)
    settings = _parse_conv_settings(conv)
    all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    branch = _build_active_branch_path(all_msgs, conv.active_message_id)
    _deduplicate_branch_file_reads(branch)
    messages = await _build_inference_context(branch, settings.active_prompt_id, sess)
    tools_list = get_ollama_tool_list(list(TOOL_REGISTRY.keys()))
    ctx_tokens = await backend.count_tokens(backend.prepare_messages(messages), tools_list)
    return {"ctx_tokens": ctx_tokens}


@router.post("/api/conversations/{id}/compress")
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

    candidates = [m for m in branch if m.role == "tool" and not m.context_excluded]
    if not candidates:
        return {"compressions": [], "new_summary": ""}

    compressions: list[Compression] = []
    new_summary: str = ""

    if candidates:
        user_messages_goal = [m.content for m in reversed(branch) if m.role == "user"][:3]
        user_message = "\n---\n".join(reversed(user_messages_goal)) if user_messages_goal else ""

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

        compressions = compression_result.compressions
        new_summary = compression_result.new_summary
        await sess.flush()

    try:
        await apply_working_memory(conv, branch, id, sess)
    except Exception:
        logger.exception("Working memory synthesis failed — skipping")

    final_all_msgs = list((await sess.scalars(select(db.Message).where(db.Message.conversation_id == id))).all())
    final_branch = _build_active_branch_path(final_all_msgs, conv.active_message_id)
    settings = _parse_conv_settings(conv)
    inference_messages = await _build_inference_context(final_branch, settings.active_prompt_id, sess)
    tools_list = get_ollama_tool_list([tool.name for tool in TOOL_REGISTRY.values()])
    ctx_tokens = await backend.count_tokens(backend.prepare_messages(inference_messages), tools_list)

    return {
        "compressions": compressions,
        "new_summary": new_summary,
        "ctx_tokens": ctx_tokens,
    }
