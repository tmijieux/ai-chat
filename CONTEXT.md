
# App Mission

Local AI chat interface backed by Ollama, designed to support agentic coding workflows on low-memory devices with a 16 384-token context window. The central design constraint is that context is scarce — every feature either helps the user *see* where tokens are going or helps the agent *preserve* as many useful tokens as possible.

# Domain Glossary

## Workspace
A directory path scoped to a conversation. Restricts all file-access tools and sets the CWD for `run_shell`. When `null`, file tools and `run_shell` are disabled entirely. Stored as `working_directory` in `ConversationSettings`.

**Persistence rules:**
- Must not reset to `null` when switching conversations or starting a new chat — the last-used working directory must carry over.
- Must survive page refresh / browser return. Currently this is a **known gap**: the value is only kept in memory for the session. Target: persist to `localStorage` or a backend user-preference so it survives reloads.

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
Per-conversation configuration stored as JSON in `Conversation.settings`. Fields: `active_prompt_id`, `active_tool_names`, `agentic_mode`, `working_directory`. `active_tool_names` is the sole authority for tool availability — empty list means no tools, full list means all tools.

## Tool Confirmation Flow
When a tool with `requires_confirmation = true` is called, the backend emits a `tool_confirm` event before executing. The frontend renders an amber card showing the tool name and a preview of the operation. The user can:
- **Approve** — tool executes.
- **Reject** — opens a textarea for an optional rejection reason; the reason is sent back to the model as part of the tool result so the agent can adjust without the run being aborted.
The confirmation card disappears from the message list once the agent run is complete.

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
The file most recently written to via `write_file` or `edit_file` in the current agent session. Identified by parsing tool-role message history. Always kept at full content in context — never compressed.

## Reference File
A file read into context but not actively being modified. Identified by parsing tool-role message history. Compressed to API surface (signatures, types, exports) by the post-iteration sub-agent when context pressure requires it.

## Known Bugs
- **`tool_name` lost on reload**: Tool result bubbles show "Tool result: `edit_file`" during a live run but just "Tool result" after reload. Fix: in `_fromDbMessages`, parse `m.content` as JSON and read the `tool` field (already present in every Tool Result Envelope) to populate `tool_name` instead of hardcoding `''`.
- **`is_default` uniqueness not enforced**: Setting a new prompt as default does not clear the previous default. Backend must clear the flag atomically.
- **Workspace not persisted across reloads**: See [[Workspace]].

## Open Questions
- **`summarize_subtask` tool**: whether the agent calls it autonomously or it is framework-triggered is TBD.
- **`context_excluded` UX**: beyond the "excluded from context" label on evicted messages, whether the user should be able to force-include an evicted message is TBD.

## Post-Iteration Sub-Agent
A fixed-prompt LLM call fired by the framework after each main agent iteration (not triggered by the main agent). Receives a digest of: tools called, changes made, last user message summary, and reference file contents to compress. Returns a JSON state object (`current_goal`, `current_assumption`, etc.) and API summaries for reference files. Enables context eviction without relying on the main agent to cooperate.

## Status Bar
Always-visible top bar in the chat area. Shows token info only: `Context Tokens: N / 16,384 (%)`. The value is the last measured `prompt_eval_count` — always from a real API call, never estimated. Shows 0 on a new chat. The agentic mode toggle has been removed from the status bar (and from the UI entirely — see [[Agentic Mode]]). The only interactive element remaining is the ⚙ button that opens the [[Conversation Settings Drawer]].

## Conversation Turn
The unit of visual grouping in the chat. One top-level bubble per speaker per iteration:
- **User bubble** — user message.
- **Assistant bubble** — groups everything the assistant produced in one iteration: collapsed thinking block (expands while streaming, collapses when done) + response text + tool confirmation card + simplified tool call summary (tool name only, no full args). All nested inside one bubble so the "who is speaking" boundary is visually clear.
- **Tool result bubble** — the tool's response; separate because it is a different speaker.
This grouping is the **target design** (not yet fully implemented). Current state has thinking as a separate flat message.

## Sidebar
Left-side panel, always visible. Contains:
- App title. Subtitle should communicate "an AI chat app that lets you keep track of your context easily" — not just "Context Token Counter".
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
The only intended mode. The agent runs a tool-calling loop over WebSocket, can call tools, ask for user confirmation on destructive ones, and run multiple iterations. **Planned:** the `agentic_mode` toggle is still present in the UI (`chat.component.html`) and in `ConversationSettings` but is scheduled for removal — the app should always use the agentic path. The non-agentic HTTP streaming code is a candidate for full removal; any logic it shares with the agentic path (DB persistence, token counting, message rendering) must be preserved when doing so.

## Token Count (cumulative)
The `prompt_eval_count` value returned by the Ollama API for a given message. Stored in `Message.token_count`. Represents the total number of tokens in the context at the point that message was sent — system prompt + all prior messages + tools overhead. **Always a measured value, never an estimate.** The status bar always shows the cumulative token count of the last message that has one; it shows 0 on a new chat because no inference has happened yet.

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
Framework-level pruning of tool-role messages before the next main iteration. Current implementation: duplicate file reads — evict older reads of the same path, keep only the most recent (partially implemented). Planned: reference file compression via [[Post-Iteration Sub-Agent]].

Evicted messages remain visible in the UI with an "excluded from context" label. They retain their stored `token_count` so the user can see what was saved. The status bar and downstream deltas reflect the post-eviction reality from the next API call.

**Interaction with token counting — known hard problem:** after an eviction or manual deletion, downstream messages have stale stored cumulatives. Design intent: re-estimate their displayed cumulative by walking forward through still-in-context messages and summing their stored deltas (each delta was computed from two actual API measurements at generation time and remains valid). The status bar always shows the last real API measurement regardless. Exact re-estimation logic is still being refined.
