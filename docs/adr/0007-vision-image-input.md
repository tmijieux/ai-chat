# ADR-0007: Vision / Image Input

**Date:** 2026-05-31  
**Status:** Accepted — pending model verification

## Context

The app targets Qwen3.5-9B, a multimodal model. The Unsloth Q3_K_XL GGUF currently in use appears to have the vision encoder tensors stripped during quantization. The Ollama GGUF has them but uses a tensor layout incompatible with llama.cpp (Ollama adapted the model before llama.cpp had native Qwen3.5 support). A separate investigation session will resolve the model question via one of:

1. Find a llama.cpp-native Qwen3.5-VL GGUF (bartowski confirmed broken; other sources TBD).
2. Convert `Qwen/Qwen3.5-VL` from HuggingFace using `convert_hf_to_gguf.py` + quantize.
3. Extract the vision encoder from the Ollama GGUF as a separate `--mmproj` file and test whether it is compatible with the Unsloth text backbone (same architecture, different quant — the hidden dim is invariant).

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

## Consequences

- Two new DB tables, one new `POST /api/images` endpoint, one new `GET /api/images/{id}` endpoint.
- `POST /api/messages` gains an optional `image_ids: string[]` field.
- `_build_inference_context` gains a DB join + content array assembly step.
- `prepare_messages` in `llama_server.py` must pass array `content` through unchanged (it currently overwrites with `m.get("content", "")` — needs a fix).
- Token counting gains the 512-per-image estimate.
- llama-server launch may need `--mmproj <file>` depending on which model path is chosen.
- No changes to the agent loop, compression pipeline, or WebSocket protocol.
