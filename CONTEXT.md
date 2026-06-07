
# App Mission

Local AI chat interface backed by Ollama, designed to support agentic coding workflows on low-memory devices with a 16 384-token context window. The central design constraint is that context is scarce — every feature either helps the user *see* where tokens are going or helps the agent *preserve* as many useful tokens as possible.

# Domain Glossary

## Workspace
A directory path scoped to a conversation. Restricts all file-access tools and sets the CWD for `run_shell`. When `null`, file tools and `run_shell` are disabled entirely. Stored as `working_directory` in `ConversationSettings`.

**Persistence rules:**
- Must not reset to `null` when switching conversations or starting a new chat — the last-used working directory must carry over.
- Survives page refresh: persisted to the backend `app_settings` table under key `last_working_directory` via `GET/PUT /api/app-settings/{key}`. Loaded on app init via `lastWorkingDirectory` query in `chat.service.ts` and pre-filled into `ConversationSettings.working_directory` for new conversations.

## DirectoryPickerComponent
Keyboard-first modal folder browser opened via the "Browse" button in the [[Conversation Settings Drawer]]. Implemented in `chat-client/src/components/directory-picker/`.

The filter input is auto-focused on open and retains focus throughout — entry buttons use `mousedown preventDefault` to avoid stealing focus, so typing always goes to the filter.

`".."` is treated as a plain entry name and filtered like any other: it appears when the filter matches `".."` and disappears otherwise.

Active entry is tracked by **path identity**: filtering keeps the current selection if the item is still visible in the filtered list; if it falls out, the first remaining entry is auto-selected. On each `browse()` call the filter and selection reset and the first entry is auto-selected.

**Keybindings:**

| Key | Effect |
|-----|--------|
| Typing | Filters the entry list |
| ArrowUp / ArrowDown | Move highlighted entry |
| Enter | Navigate into highlighted entry |
| Ctrl+Enter | Confirm selection of the current directory |
| Backspace (empty filter) | Go up one directory |
| Escape (non-empty filter) | Clear filter |
| Escape (empty filter) | Close picker |

## SystemPromptTemplate
A named, versioned system prompt stored in the DB (`system_prompt_templates` table). Has a `name`, `category`, `content`, `token_count` (computed on create/edit), and `is_default` flag. The `token_count` is evaluated by calling `count_token` with `[{"role": "system", "content": content}]`.

## is_default
Flag on `SystemPromptTemplate`. Rules:
- **At most one** prompt may have `is_default = true` at any time. Setting a prompt as default must clear the flag on the previously-default prompt (backend must enforce this atomically). Uniqueness may not yet be implemented — treat as a known gap.
- **Visual indicator**: the prompt list on the settings page shows a "default" badge on the default prompt.
- **New-chat pre-selection**: when starting a new chat, the default prompt is automatically set as `active_prompt_id` in the pending `ConversationSettings`.

## active_prompt_id
Per-conversation setting (`ConversationSettings.active_prompt_id: string | null`). Points to the active `SystemPromptTemplate` for that conversation. `null` means no system prompt is active.

## ConversationSettings
Per-conversation configuration stored as JSON in `Conversation.settings`. Fields: `active_prompt_id`, `active_tool_names`, `working_directory`. `active_tool_names` is the sole authority for tool availability — empty list means no tools, full list means all tools.

## Tool Confirmation Flow
When a tool with `requires_confirmation = true` is called, the backend emits a `tool_confirm` event before executing. The frontend renders an amber card showing the tool name and a preview of the operation. The user can:
- **Approve** — tool executes.
- **Reject** — opens a textarea for an optional rejection reason; the reason is sent back to the model as part of the tool result so the agent can adjust without the run being aborted.
The confirmation card disappears from the message list once the agent run is complete.

## Chat Auto-scroll
The message list auto-scrolls to the bottom when new content arrives, but only if the user was already at the bottom. Scrolling up manually disables auto-scroll — no jumps while reading. Scrolling back to within 50 px of the bottom re-enables it.

**Not yet implemented.**

## Vision / Image Input
Allow pasting or dragging images into the chat input area. Multiple images per message are supported.

**Model status:** Qwen3.5-9B is a unified VLM. Text backbone: `~/ai/models/unsloth/Qwen3.5-9B-UD-Q3_K_XL.gguf`. Vision encoder: `~/ai/models/unsloth/mmproj-F16.gguf` (Unsloth, F16, ~918 MB). llama-server is launched with `--mmproj mmproj-F16.gguf`. Investigation complete — see ADR-0007.

**DB schema:** two new tables. `images` stores the blob once (`id`, `mime_type`, `data` base64, `width`, `height`, `created_at`). `message_image_attachments` is a join table (`message_id`, `image_id`, `position`). Branching copies attachment rows without duplicating image data. Orphaned `images` rows are GC'd on message/conversation delete.

**Upload:** `POST /api/images` → `{ id, mime_type }`. Uploaded immediately on attach (before send), with a thumbnail loading indicator. On send, `POST /api/messages` includes `image_ids: string[]`. Accepted formats: JPEG, PNG, WebP.

**Context assembly:** `_build_inference_context` batch-fetches `message_image_attachments`, assembles the OpenAI multimodal content array per message: `[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:mime;base64,..."}}]`. Text-only messages stay as plain strings.

**Token counting:** text tokens (existing path) + `ceil(w/32) × ceil(h/32)` per image (Qwen3.5: patch=16, merge=2 → 32 px/token). Image dimensions stored at upload via Pillow. Formula source: llama.cpp discussion #17172.

**Message list API:** returns `images: [{id, mime_type}]` per message (no base64). Frontend lazy-loads via `GET /api/images/{image_id}`.

**Frontend:** paste (`Ctrl+V`) and drag-and-drop onto the textarea append to a 64×64 thumbnail strip. Each thumbnail has an ✕ remove button and a spinner while uploading. Send button disabled while any image is still uploading. Persisted images display in user message bubbles (max 200×320 px).

**Context eviction (deferred):** when implemented — strip image parts from evicted messages, store an AI-generated description as `compressed_summary`, provide a `reload_image` agent tool. See ADR-0007.

## Tool Result Display
Tool result bubbles currently render raw JSON in a `<pre>` block. Planned improvements:

- **Grep results** (`grep_files`): Parse `matches: [{file, line, content, match?}]`. Render grouped by file — file-path header per group, line numbers, matched lines highlighted (green) vs context lines (subdued). The `log_message` header already shows a one-liner summary.
- **Write / edit results**: Compact success/error line — just path and status, no raw JSON blob.
- **Run_shell results**: Exit code prominently + stdout in a scrollable code block.
- **Other tools**: Surface key fields (path, status, count) as a summary line; hide the rest.

`compressed_summary` always takes precedence; structured rendering applies only to the raw content fallback.

**Not yet implemented** (compressed_summary display is already live).

## Edit File Diff Preview
**Planned improvement.** The `edit_file` confirmation card currently shows a plain-text `--- OLD --- / --- NEW ---` preview. Target: render a colored unified diff with actual file line numbers — red lines for removals (`-`), green lines for additions (`+`), gray/neutral for unchanged context lines.

Implementation sketch:
- **Backend** (`edit_file.py:execute()`): after reading `current_content` (which already happens before `request_confirm()`), compute the start line of `old_string` in the file (`current_content[:idx].count('\n') + 1`), then run `difflib.unified_diff` on the before/after file slices to produce diff lines with correct `@@` offsets. Pass a `diff_lines: list[{type, text}]` structure into `request_confirm()` alongside the existing plain-text `preview` (kept as fallback). Note: `validate()` is not the right place — it lacks `working_directory` and doesn't read the file.
- **Agent event** (`agent.py` / `main.py`): `tool_confirm` WS event gains an optional `diff_lines` field.
- **Frontend** (`agent.service.ts`): parse `diff_lines` from the event into the `AgentUiMessage`.
- **Frontend** (`chat.component.html`): when `diff_lines` is present on a `tool_confirm` message, render a colored diff block (monospace, red bg for `-` lines, green bg for `+`, transparent for context) instead of the plain `<pre>`.
- **Scope**: only `edit_file` needs this; `write_file` and `run_shell` keep their existing plain-text previews.

## Tool
A self-contained agent capability implemented as a single Python file in `backend/agent/tools/`. Each tool subclasses a common base class that defines: `name`, `description`, `parameters` (JSON schema), `requires_confirmation`, `validate(args) → preview`, and `execute(args) → result`. The `validate` phase runs before user confirmation; `execute` runs after. All tools are explicitly registered in `backend/agent/tools/__init__.py`.

## Prompt Bootstrap
A directory (`backend/prompts/`) of `.md` files with YAML frontmatter. On backend startup, any file marked `is_current: true` is seeded into the DB as a `SystemPromptTemplate` (idempotent — skipped if a row with that name already exists). Purpose: makes it easy to start from a fresh DB without re-entering prompts manually. All prompt authoring happens in the settings UI (create, edit, delete). There is no mechanism to write prompts back to git files — that is a potential future feature.

## Branch
A linear path from a root message to a leaf in the conversation message tree. The active branch is determined by `Conversation.active_message_id` (always a leaf); the full path is reconstructed by walking `parent_id` chains up to the root.

## Edit User Message
Pencil icon (✎) revealed on user message hover. Opens an inline textarea pre-populated with the current content. Submitting creates a new sibling via `PUT /api/messages/{id}/branch` (the new message shares the same parent), sets it as `active_message_id`, reloads the branch, then triggers a new agent run from that message. The prior assistant response is on the now-inactive sibling — it is not deleted, just no longer on the active branch. This is the primary mechanism by which the message tree grows siblings.

## Sibling Navigator
Inline `← N/M →` arrows rendered on any message that has siblings (multiple children of the same parent). Switching siblings calls `PUT /api/conversations/{id}/active-branch` and auto-advances to the deepest leaf of the selected sibling.

## Subtree Delete
Deletes a message and all its descendants. The existing `DELETE /api/conversations/{conv_id}/messages/{msg_id}` endpoint with `?subtree=true`.

## Single-Message Delete
Deletes only one message, re-parenting its direct children to the deleted message's parent. If the deleted message was a root (`parent_id = null`), children become new roots (siblings at the top of the tree). Same endpoint with `?subtree=false`.

## Tool Result Envelope
The structured JSON dict that every tool's `execute()` returns. Contains at minimum `tool` (tool name) and any relevant paths or identifiers (e.g. `path` for file tools). Serialized to JSON by the framework layer, never by the tool itself. Allows deterministic parsing of tool-role messages by the context manager.

## Active File
**Not yet implemented.** Concept for the planned [[Post-Iteration Sub-Agent]] pipeline: the file most recently written to via `write_file` or `edit_file` in the current agent session, identified by parsing tool-role message history. Will always be kept at full content in context — never compressed.

## Reference File
**Not yet implemented.** Concept for the planned [[Post-Iteration Sub-Agent]] pipeline: a file read into context but not actively being modified, identified by parsing tool-role message history. Will be compressed to API surface (signatures, types, exports) by the post-iteration sub-agent when context pressure requires it.

## Voice Dictation

Hold-to-record mic button in `ChatInputComponent`. Three visual states: gray (idle), red pulsing (recording), yellow pulsing (transcribing). Hold mousedown → record; release anywhere (mouseup on document) → stop. Alt held 500ms in textarea → same. Partial transcripts appear in the textarea while speaking; final result replaces them on release.

**Pipeline:** `backend/whisper_pipeline.py` — OpenVINO inference, encoder and decoder both on GPU.0 (Intel Arc iGPU). Audio decoded via ffmpeg (handles WebM/WAV/OGG). Tensor names and statefulness auto-detected at load time (`_introspect`), so any Whisper variant works without code changes. Pipeline exposed as `WhisperPipeline` dataclass; call `load_pipeline()` once at startup, pass to `transcribe(pipeline, audio_bytes, language)`.

**Model variants** (defined in `whisper_pipeline.py`, swap by changing `ACTIVE_VARIANT`):

| Constant | HuggingFace ID | Dir | Stateful |
|---|---|---|---|
| `WHISPER_TINY` | `openai/whisper-tiny` | `whisper/ov_model_tiny` | yes |
| `WHISPER_BASE` | `openai/whisper-base` | `whisper/ov_model_base` | yes |
| `WHISPER_SMALL` | `openai/whisper-small` | `whisper/ov_model_small` | no |
| `WHISPER_LARGE_FR` | `bofenghuang/whisper-large-v3-french` | `whisper/ov_model_large_fr` | yes (re-exported with optimum-intel stateful) |

Stateful models use one-token-at-a-time decode with internal KV cache (`_decode_stateful`) — O(n). Non-stateful use full-sequence decode (`_decode_full_sequence`) — O(n²). Compiled blobs cached per variant+device in `whisper/compiled_blobs_<name>/`. Export new variants with `optimum-cli export openvino` from the `whisper/venv`.

**STT correction:** `POST /api/correct` calls `_correct_stt()` — non-streaming llama-server call with few-shot prompt. Fixes misheard technical terms and mangled French words.

**Streaming partials:** `VoiceDictationService` collects 200ms chunks via `MediaRecorder`. After every 3 new chunks (≈600ms), `_maybeFirePartial()` sends the full accumulated blob to `POST /api/transcribe` and updates `partialText` signal. On release: the last in-flight partial (or a new call if none) becomes the final, then chains to `/api/correct`. Frontend effect in `ChatInputComponent` writes `_startPrefix + partial` to `currentInput` on every partial update.

**Overlay + diff (not yet implemented):** On recording start, a `contenteditable` div appears absolutely positioned over the textarea, pre-filled with the current text and cursor at the same position. Partials update the overlay live. After the final + correction, a word-level LCS diff between `raw` and `corrected` is rendered: 1–2 changed words get `<span class="diff-word">` (orange bg), 3+ consecutive changed words get `<span class="diff-block">` — clicking either toggles between corrected and raw. Dismissal: Enter/send serializes overlay text to `currentInput` and hides the overlay; any printable key does the same then inserts the key at the cursor.

**Mic button states (not yet implemented):** Four states driven by signals — idle (gray, `title="Hold to dictate (or hold Alt)"`), idle with textarea selection (gray + replace-hint indicator), recording (red pulse), transcribing (yellow pulse). Selection state tracked via `selectionchange` event on the textarea.

**30-second limit (not yet implemented):** Whisper's mel spectrogram window is fixed at 30s — audio beyond that is silently truncated. The mic button shows a progress indicator as recording approaches 30s and auto-stops at 28s with a visual warning.

## Known Bugs
- **`is_default` uniqueness not enforced**: Setting a new prompt as default does not clear the previous default. Backend must clear the flag atomically.
- **Thinking-only assistant messages** (content = ""): Saved to DB — correctly filtered from LLM context but add noise to the message list.

## Open Questions
- **`summarize_subtask` tool**: whether the agent calls it autonomously or it is framework-triggered is TBD.
- **`context_excluded` UX**: beyond the "excluded from context" label on evicted messages, whether the user should be able to force-include an evicted message is TBD.

## Post-Iteration Sub-Agent
Framework-level compression pipeline that runs after each completed agent run. Two stages implemented (see ADR-0006):

**Stage 1 — Usefulness classification + deterministic compaction** ✅: a single batch LLM call receives the current user message, an optional one-sentence conversation summary, and for each tool call: `(tool_name, key_args, result_metadata, following_thinking)` — never the full tool output. Returns `compress | keep` per tool call plus an updated conversation summary. Compressed results get a compact one-liner (`glob_files("**/*ome-box*") → 0 files`) stored in `compressed_summary` and shown in the UI instead. `message.content` is never modified. Tools in `_SKIP_CLASSIFY = {write_file, edit_file}` are never classified. Thinking messages always kept verbatim.

**Stage 2 — Reference file summarization** ✅: for each `keep` `read_file` result exceeding ~2 000 chars, an LLM call produces an API surface summary (module purpose, function signatures, key constants, imports). Stored as `compressed_summary` with a `[compressed: N lines → ~M tokens]` header. Target: 400–800 tokens.

**Not yet implemented:**
- **Active file tracking**: the file currently being written/edited should never be summarized in Stage 2. Currently all `read_file` results are eligible. Design open: "last file touched" only protects the most recent edit in a multi-file run; "all files edited in the run" protects more but prevents compression even when edits are complete. Prompt engineering (instruct agent to re-read before editing) may reduce the problem.
- **Conversation title update**: after the first agent run, generate a short goal-framed title from the first user message via a small LLM call (not from the compression summary, which is agent-centric). Only update once — subsequent runs preserve the title. Update only if the title is still the auto-generated default (first 20 chars of first message), so manual renames are never overwritten.
- **On-demand compression**: `agent.py:241` already detects `ctx_after > CTX_LIMIT` after each tool result and emits an error. Instead of failing immediately, the agent should: (1) attempt compression of all eligible tool-role messages in the full conversation history, (2) re-count tokens, (3) continue the run if now within budget. If still over: emit an error telling the user to manually delete old tool results or start a new conversation. No new event type needed — uses the existing error bubble.
- **Oversized output summarization** for non-`read_file` tools (e.g. `run_shell` with large stdout).

Never compressed by Stage 2: `write_file`, `edit_file`, `run_shell` results; thinking messages.

## Status Bar
Always-visible top bar in the chat area. Shows token info only: `Context Tokens: N / 16,384 (%)`. The value is the last measured token count — always from a real API call, never estimated. Shows 0 on a new chat. The ⚙ button opens the [[Conversation Settings Drawer]].

## Conversation Turn
The unit of visual grouping in the chat. One top-level bubble per speaker per iteration:
- **User bubble** — user message.
- **Assistant bubble** — groups everything the assistant produced in one iteration: collapsed thinking block (expands while streaming, collapses when done) + response text + tool confirmation card + simplified tool call summary (tool name only, no full args). All nested inside one bubble so the "who is speaking" boundary is visually clear.
- **Tool result bubble** — the tool's response; separate because it is a different speaker.
This grouping is the **target design** (not yet fully implemented). Current state has thinking as a separate flat message from tools and assitant response.

## Sidebar
Left-side panel, always visible. Contains:
- App title ("AI Chat"). Subtitle should communicate "an AI chat app that lets you keep track of your context easily"
- "New Chat" entry at the top of the list.
- Conversation list — click to select, ⋮ menu → Delete.
- Settings button at the bottom linking to `/settings`.

Conversation title is set to the first 20 characters of the first user message. AI-generated titles are a low-priority nice-to-have.

## Conversation Settings Drawer
Right-side panel opened via the ⚙ button in the status bar. Per-conversation configuration that persists to DB. Three sections:
1. **Workspace** — text input + Browse (custom `DirectoryPickerComponent`) + clear. See [[Workspace]].
2. **System prompt** — dropdown of all `SystemPromptTemplate`s; selecting one sets `active_prompt_id`. Shows token count per option.
3. **Tools** — per-tool checkboxes with token cost and "requires confirmation" badge; select-all / deselect-all toggle; total token cost of all enabled tools shown at top of section.

## Message Actions
Hover-revealed action buttons on each message. Must be preserved on all message kinds:
- **User message**: Edit (✎) — opens inline textarea, Enter submits and creates a branch; ⋮ menu → Delete message / Delete branch.
- **Assistant message**: Raw toggle (¶/◈) — switches between rendered markdown and raw source; ⋮ menu → Delete message / Delete branch.
- **Tool result / Thinking**: ⋮ menu → Delete message / Delete branch only.
- **Tool confirmation card**: no action buttons (it has its own Approve/Reject UI).
"Delete branch" only appears in the ⋮ menu when the message has children (`has_children = true`).

## Markdown Rendering
Assistant message content is rendered as markdown via `ngx-markdown` with Prism syntax highlighting. Prism automatically injects a **copy-to-clipboard button** on every code block — this is a key UX feature and must not be broken by swapping the renderer or changing its configuration. The raw toggle (¶/◈) on each assistant message lets the user see the unrendered source when needed.

## Agent Message Persistence
During an agentic run, messages are saved to DB **incrementally** via a sequential promise queue (`saveQueue` in `chat.service.ts`). Saving order:
- Assistant message — saved when its streaming stops (on `tool_result` or `iteration_end`)
- Tool result — saved immediately when the `tool_result` event arrives
- Token counts — patched onto messages after each `iteration_end` (previous iteration's messages get the new iteration's `prompt_tokens`)

This means the DB reflects the run state in near-real-time. Do not refactor this into a single batch save at the end — it would delay DB writes and break the token count patching sequence which depends on message IDs already existing in the DB.

## Agentic Mode
The ai chat is agentic and can use tools intended mode. The agent runs a tool-calling loop over WebSocket, can call tools, ask for user confirmation on destructive ones, and run multiple iterations.

## Token Count (cumulative)
The `prompt_eval_count` value returned by the LLM backend for a given message. Stored in `Message.token_count`. Represents the total number of tokens in the context at the point that message was sent — system prompt + all prior messages + tools overhead. **Always a measured value, never an estimate.** The status bar always shows the cumulative token count of the last message that has one; it shows 0 on a new chat because no inference has happened yet.

## Token Delta
The difference between a message's cumulative token count and the closest preceding message that also has a token count. Displayed in the per-message ⓘ tooltip as "This message: ~N tokens". Computed at generation time from two consecutive API-measured cumulatives and stored in `Message.token_delta`. Stable forever — does not change when upstream messages are deleted or evicted. Critical because context size varies with: which system prompt is active, which tool responses have been evicted, and which messages have been deleted.

## Token Visibility Surfaces
Five places in the UI where token information is shown — all intentional, all must be preserved:
1. **Status bar** — always-visible; shows last measured cumulative / 16 384 (%). Shows 0 on new chat.
2. **Per-message ⓘ tooltip** — cumulative count, %, and delta. Only shown on messages that have a stored `token_count`.
3. **System prompt bubble** — shows prompt token count and `+ tools (~N tok)` when a prompt is active.
4. **Settings drawer / tools section** — total token cost of enabled tools + per-tool cost.
5. **Settings page / prompt list** — each prompt option label includes `(~N tok)`.

## Context Eviction
Framework-level pruning of tool-role messages before the next main iteration. Current implementation: duplicate file reads — evict older reads of the same path, keep only the most recent (fully implemented: backend deduplication + frontend "excluded from context" label).

Evicted messages remain visible in the UI with an "excluded from context" label. They retain their stored `token_count` so the user can see what was saved. The status bar and downstream deltas reflect the post-eviction reality from the next API call.

Reference file compression (Stage 2 of the [[Post-Iteration Sub-Agent]]) is implemented — large `read_file` results are summarized and stored as `compressed_summary`. Not yet implemented: **oversized output summarization** for non-`read_file` tools (e.g. `run_shell` with large stdout).

**Interaction with token counting — known hard problem:** after an eviction or manual deletion, downstream messages have stale stored cumulatives. Design intent: re-estimate their displayed cumulative by walking forward through still-in-context messages and summing their stored deltas (each delta was computed from two actual API measurements at generation time and remains valid). The status bar always shows the last real API measurement regardless. Exact re-estimation logic is still being refined.
