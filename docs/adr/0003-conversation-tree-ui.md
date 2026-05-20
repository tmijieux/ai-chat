# ADR-0003: Conversation tree UI — sibling navigation, edit, and delete

**Date:** 2026-05-20  
**Status:** Accepted

## Context

The message tree exists in the DB but is invisible to the user. Branching only happens implicitly (e.g. after an agent edit) and there is no way to navigate siblings, edit a user message, or selectively delete messages.

## Decisions

### Sibling metadata on the branch endpoint

`GET /api/conversations/{id}/messages` enriches each message with:
- `sibling_count: int` — total children of the same parent
- `sibling_index: int` — 1-based position among siblings (ordered by `created_at`)
- `prev_sibling_id: string | null`
- `next_sibling_id: string | null`

No separate tree fetch or frontend tree state. The branch response is self-contained.

**Alternative rejected:** load `GET /api/conversations/{id}/tree` and derive sibling info on the frontend — adds a second fetch and requires the frontend to hold extra state.

### Sibling navigator placement

Rendered **above** the message content for any message where `sibling_count > 1`. Shows `← N/M →`. Arrows call `PUT /api/conversations/{id}/active-branch` with `prev_sibling_id` / `next_sibling_id`, then reload messages.

### Edit user messages

Pencil icon visible on user-message-row hover. Click replaces the message with an inline textarea (prepopulated). Submit:
1. `PUT /api/messages/{id}/branch` — creates a sibling with the new content; backend sets this as `active_message_id`
2. Reload messages
3. Trigger generation via `_generateResponse()` (extracted from `sendMessage`)

Editing discards the prior assistant response (it was on the now-inactive sibling branch). This is intentional — edit = start a new branch from the same parent.

Only user messages are editable in Phase 4.

### Delete

`DELETE /api/conversations/{conv_id}/messages/{msg_id}` gains a `?subtree` query param:
- `subtree=true` (default, existing behavior) — deletes message and all descendants
- `subtree=false` — deletes only the message; re-parents direct children to `msg.parent_id`; if msg was a root, children become new roots (`parent_id = null`). Backend adjusts `active_message_id` if the deleted message was on the active path.

All messages (user, assistant, tool) can be deleted.

### Delete menu

`⋮` button on every message, revealed by CSS on message-row hover (no JS). Click opens a dropdown:
- "Delete branch" — always shown (`subtree=true`)
- "Delete message" — shown only when the message has children (`subtree=false`); hidden for leaves since it would be identical to "Delete branch"

### Agent websocket — edit compatibility

`agent_websocket` currently inserts the user message into the DB. For the edit flow the user message is already in the DB (created by the branch endpoint). The first WS message gains an optional `user_message_id` field; when present, the backend skips insertion and uses the existing row.

## Consequences

- No new endpoints beyond the `?subtree` param and enriched branch response.
- `_generateResponse()` extracted from `sendMessage` — reused by both normal send and post-edit generation.
- Frontend holds no tree state; sibling navigation is purely driven by IDs in the branch response.
