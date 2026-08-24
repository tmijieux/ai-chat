# ADR-0019: `rag_search` Agent Tool

**Date:** 2026-08-24
**Status:** Accepted

## Context

ADR-0015 built the RAG infrastructure and ADR-0016 added `/rag-search` as a manual, non-agentic
testing surface, explicitly deferring the real feature: letting a running agent query a space on
its own. `todo.md` flagged this as needing both a tool and "a decision on how a conversation
selects which space(s) it can search."

## Decision

### Reuses the one-space-per-workspace convention from ADR-0016, not a new selection mechanism

`rag_search` takes only a `query` parameter — no `space` argument. It resolves the same way
`/rag-search` does: find the `RagSpace` whose `workspace_path` matches the conversation's active
workspace, or create one named after the workspace's directory. This keeps the tool and the slash
command permanently in agreement about which space a given workspace means, and needs no new
UI or settings surface to ship. The tradeoff is the same one ADR-0016 already accepted: no global
spaces and no multiple spaces per workspace are reachable from this tool — that waits for the
RAG spaces management UI (still on `todo.md`).

### Read-only, no confirmation, always-safe in Auto/YOLO mode

Like `explore_codebase`, `grep_files`, etc., `rag_search` only reads already-indexed chunks — it
can't mutate anything, so it doesn't prompt for confirmation and is in Auto mode's
always-safe tool set.

### No workspace configured → tool error, not silent no-op

Since the space is derived entirely from the workspace, a conversation with no workspace set
can't resolve one — the tool returns an error telling the agent so, the same pattern
`explore_codebase` uses for the same reason.

## Verification

Ran the tool directly (in-process, not through the API) against a synthetic throwaway workspace
directory ingested via `rag.ingestion.ingest_workspace_path` — confirmed a natural-language query
correctly ranked the matching source above nothing else, and confirmed the auto-create path
returns an empty result set (not an error) for a workspace with no space yet. `measured_delta`
calibrated via the `recount-tool-tokens` skill against the live model (isolated delta: 342).
