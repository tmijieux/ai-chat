# ADR-0011: Persisted, Browsable Workflow Run History

**Date:** 2026-08-17
**Status:** Accepted

The run view (ADR-0009) is live-only: full per-invocation detail (thinking, tool calls, tool results) is kept in frontend memory only for the most recent stage invocations, and is gone for good once a stage ages out or the page refreshes. On a long-running workflow — the kind that legitimately takes hours (map-codebase style, unattended many-item loops) — that's exactly the detail you need afterward to work out why something failed.

## Decision

Every stage invocation, at every nesting depth including sub-workflow calls, writes its full transcript (thinking, tool calls and their results, and the stage's finish result) to disk as it runs, under `workflow_runs/<workflow_name>/<run_id>/`. The live websocket stream is unchanged — it still drives "what's currently running" exactly as ADR-0009 describes. What's new is purely additive: clicking any *finished* node in the run view now fetches that node's full content plus a shallow list of its own children's status from disk, one level at a time, instead of relying on whatever the frontend still happens to have buffered.

## The directory shape mirrors the runtime call tree, not the engine's flat invocation counter

The orchestrator already gives every stage dispatch a globally-unique identity (`execution_id = path#N`, a counter that does not reset per loop item — see ADR-0009/0010). That was the obvious key to persist under, and it was rejected: it doesn't nest. A loop nested inside another loop (e.g. map-codebase's per-chunk loop inside its per-file loop) has invocation numbers that only make sense relative to which outer item they belong to; keyed by the flat counter, reconstructing "which chunk of which file was invocation #41" needs a lookup table.

Instead, the tree is nested literally: a loop stage's directory holds one subdirectory per item number, and each item's inner stages get their own subdirectories under that — recursively, so a nested loop's item numbers reset naturally inside their parent item's own folder and can never collide with another parent item's numbers. This closes, as a side effect, the nested-loop item-identity ambiguity ADR-0009 flagged as a known gap in the live event stream — no event schema change was needed, because the disk layout disambiguates structurally instead.

Retries stay inside the same directory rather than adding another nesting level: a stage's directory holds a small `meta.json` (status, attempt count) plus one `attempt_N.json` per attempt, numbered from 1 per item — not a separate `attempt_N/` folder, since almost every stage only ever has one attempt and an extra directory level for the common case wasn't worth it.

## Every node persists its full transcript, not just its tidy result

A failed stage (e.g. an LLM stage that exhausted `max_iterations` without calling its finish tool) has no finish result at all — `stage_exit` already emits `result: None` for that case. Persisting only the finish result would leave exactly the nodes most worth inspecting after a failure with nothing recorded. Every invocation — success or failure, including `respond` stages, which don't otherwise write into the slot registry — persists its whole transcript: thinking, tool calls, tool results, and (when there is one) the finish result.

## Live streaming stays as-is; this does not replace it with polling

Considered making the frontend always render from the persisted tree, live run included (poll or push-refresh), so there would be one representation of a run instead of two. Rejected — it would trade the current low-latency token-by-token streaming for refresh-on-notify, for no real benefit: the live event stream and the disk tree never disagree, since the disk tree is written from the same events the live view already consumes. Keeping them separate — events drive the live view exactly as before; disk is a separate, on-demand read path for anything already finished — is simpler and leaves ADR-0009's existing mechanism untouched.

## Deferred: resumability

This ADR covers persistence and browsability only. Actually resuming a stuck or aborted run — reconstructing the slot registry from disk, letting the operator edit a completed node's result or skip a stage, and re-executing forward from an explicit point, with loop siblings treated as independent and thus skippable by default — was designed alongside this but was deliberately not built yet, pending validation that the persisted format held up in practice first.

**Update:** validated and built, across 4 phases — see `~/.claude/plans/enumerated-rolling-bee.md` for the implementation plan and `CONTEXT.md`'s "Resuming a run" for the user-facing behavior.
