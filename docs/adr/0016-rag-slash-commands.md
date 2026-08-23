# ADR-0016: Minimal `/rag-index` / `/rag-search` Slash Commands

**Date:** 2026-08-23
**Status:** Accepted

## Context

ADR-0015 built the RAG indexing/retrieval infrastructure but deliberately stopped short of any
agent tool wiring or UI. To actually try ingesting a directory and searching it, a minimal
interface was wired into chat via the [[Slash Command Palette]] — explicitly a testing surface,
not the eventual real feature (agent tool wiring and a spaces-management UI remain on `todo.md`).

## Decision

### Bypasses the agent loop entirely — no LLM call involved

`/rag-index` and `/rag-search` are a third `SlashCommand` variant alongside `mode` and `workflow`,
but unlike `workflow` (which still runs through the LLM-backed pipeline), selecting one routes to
a new `ChatService.runRagCommand()` that talks to `/api/rag/...` directly and never touches
`AgentService`/the WebSocket agent loop. This was the deliberate point: proving the retrieval
stack works should not require the chat model to be involved at all, and ADR-0015 already chose a
CPU embedding provider specifically so RAG operations don't need the chat model loaded.

### Result persisted as a plain assistant message, not a new message kind

The command's result (a formatted list of indexed files, or ranked search snippets) is persisted
as an ordinary `role: "assistant"` message via the same `POST /api/messages` endpoint every other
message already uses, immediately following a `role: "user"` message holding the typed command —
exactly the same two-message shape a normal agent turn produces, just computed synchronously
instead of streamed. This reuses 100% of existing message-tree/branch machinery (parent chaining,
active-branch advancement, markdown rendering) with no new message kind, no new rendering
component, and no persistence code beyond what already existed. The tradeoff: these messages are
indistinguishable in the DB from a real conversational turn — acceptable for a manual testing
surface, but a reason the real agent-tool integration (when built) may want a distinguishable
representation instead.

### One RAG space per workspace, auto-provisioned

Rather than asking the user to name/pick a space (there's no space-management UI yet), both
commands resolve "the space for the conversation's active workspace": look for an existing
`RagSpace` whose `workspace_path` matches, or create one named after the workspace's directory
name. This means `/rag-index` and `/rag-search` always agree on which space they're talking about
within one workspace, and repeated indexing accumulates into the same space rather than creating
duplicates — but it also means this flow can only ever produce workspace-bound spaces, never a
global one; multi-space-per-workspace and global-space creation are left to the real UI.

## Verification

No browser automation tool is available in this environment, so the UI itself wasn't clicked
through. Instead, verified: TypeScript compiles and `ng build` succeeds with no errors; and a
script replaying the exact HTTP call sequence `runRagCommand()` performs (create conversation →
set workspace → post user message → get-or-create space → ingest `docs/adr/` → post formatted
assistant message → post second user message → confirm the *same* space is reused, not
duplicated → query → post formatted assistant message) against a live, isolated (temp SQLite, no
LLM loading) backend instance — confirming correct branch chaining, message ordering, and search
ranking (a "workflow run visualisation stages" query correctly ranked ADR-0009 top).
