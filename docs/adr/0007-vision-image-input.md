# ADR-0007: Vision / Image Input

**Date:** 2026-05-31  
**Status:** Accepted

## Context

The app targets Qwen3.5-9B, a native unified VLM (not a separate `-VL` variant). The Unsloth GGUFs are the text backbone only; the vision encoder ships as a separate `mmproj-F16.gguf` in the same HuggingFace repo (`unsloth/Qwen3.5-9B-GGUF`).

## Resolution (2026-05-31)

Qwen3.5-9B is a native unified VLM. The Unsloth GGUFs (`Q3_K_XL`, `Q4_K_M`) are the text backbone; the vision encoder is `mmproj-F16.gguf` (~918 MB, F16). Downloaded to `~/ai/models/unsloth/mmproj-F16.gguf`. Verified: llama-server b9399 loads both at 32 768-token context with `-ngl 99`. `llama_server.py` updated with `MMPROJ_PATH` constant and `--mmproj` launch arg.

The Ollama GGUF (monolithic, vision embedded) failed to load with `rope.dimension_sections` array-length mismatch — not pursued further.

The app is built assuming vision works. No graceful degradation or capability detection. Model is not switchable at runtime.

## Decisions

### DB schema

Two new tables (no changes to `messages.content`):

```sql
CREATE TABLE images (
    id       TEXT PRIMARY KEY,
    mime_type TEXT NOT NULL,   -- image/jpeg | image/png | image/webp
    data     TEXT NOT NULL,    -- base64-encoded blob
    created_at TEXT NOT NULL
);

CREATE TABLE message_image_attachments (
    message_id TEXT NOT NULL REFERENCES messages(id),
    image_id   TEXT NOT NULL REFERENCES images(id),
    position   INTEGER NOT NULL,  -- ordering within the message
    PRIMARY KEY (message_id, image_id)
);
```

**Rationale for two tables:** images are first-class entities independent of messages. When a user message is branched (edited), the new branch message copies `message_image_attachments` rows pointing to the same `image_id` — zero data duplication. Orphaned `images` rows (no attachments) can be GC'd.

### Upload flow

1. Frontend uploads each image immediately on attach via `POST /api/images` → returns `{ id, mime_type }`.
2. Frontend shows a thumbnail with a live upload speed indicator while the upload is in progress.
3. On send, `POST /api/messages` includes `image_ids: string[]`. Backend creates `message_image_attachments` rows.
4. Accepted formats: JPEG, PNG, WebP. No hard size limit (app is local, client and server on same machine). Frontend shows a warning if file is large.

### Context assembly (`_build_inference_context`)

For each message in the branch, if `message_image_attachments` exist, `_build_inference_context` assembles the OpenAI multimodal content array:

```python
[
  {"type": "text", "text": message.content},
  {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
  ...  # one entry per attached image, ordered by position
]
```

Text-only messages keep their plain string `content` — the wire format is unchanged for them.

### Token counting for image messages

Text tokens are counted via the existing `/tokenize` path. Images are not tokenizable this way. Token estimate: **text_tokens + 512 × image_count** per message. Stored as `token_count` on the message. The ⓘ tooltip notes the estimate is approximate.

### Message list API

`GET /api/conversations/{id}/messages` returns `images: [{id, mime_type}]` per message (no base64 — list stays lean). Frontend lazy-loads image data via `GET /api/images/{image_id}` only when rendering thumbnails.

### Edit / branch

`PUT /api/messages/{id}/branch` copies `message_image_attachments` rows from the original message to the new branch message (same `image_id` references, new `message_id`). No image data is duplicated.

### Context eviction (deferred)

Image eviction is intentionally deferred. Design intent when implemented:
- Evicted user message: strip image parts from context, replace with an AI-generated one-sentence image description stored as `compressed_summary`.
- Provide a new `reload_image` agent tool so the model can request the full image be re-injected if needed.

### Multiple images per message

Supported. Frontend shows a thumbnail strip (N thumbnails) below the textarea. Each thumbnail has an ✕ remove button. Paste (`Ctrl+V`) and drag-and-drop onto the textarea both append to the strip.

## Consequences (implemented 2026-05-31)

- Two new DB tables (`images` with `width`/`height`, `message_image_attachments`).
- `POST /api/images` and `GET /api/images/{id}` endpoints added.
- `POST /api/messages` accepts optional `image_ids: string[]`.
- `_build_inference_context` batch-fetches attachments, builds OpenAI multimodal content arrays.
- `prepare_messages` in `llama_server.py`: `m.get("content", "")` already passes list content through unchanged — no fix needed.
- Token counting: `ceil(w/32) × ceil(h/32)` per image via Pillow dimensions stored at upload. Formula confirmed via llama.cpp discussion #17172.
- llama-server launched with `--mmproj ~/ai/models/unsloth/mmproj-F16.gguf`.
- Orphaned `Image` rows GC'd on message/conversation delete via `_delete_attachments_and_gc_images()`.
- `agent.py` `_log_context` updated to handle list content.
- No changes to the agent loop, compression pipeline, or WebSocket protocol.
