# ADR-0009: Workflow Run Visualisation

**Date:** 2026-07-26
**Status:** Accepted

Running a workflow was a black box. Every event emitted inside a stage was tagged as coming from a sub-stage, and the frontend dropped all of them except tool confirmations. A translation workflow over an 84-chunk locale file therefore ran 168 LLM stages with no feedback at all until the final response. The per-stage summary events the orchestrator did emit were never handled and silently discarded.

## Decision

The orchestrator emits a small set of structured lifecycle events — the static stage tree once at the start, then enter/exit per stage invocation, plus per-item outcomes for loops. Events produced *inside* a stage are stamped with the identity of the stage invocation that produced them. A dedicated run view assembles all of this into a live picture of the workflow.

## Workflow activity gets its own surface, not the message list

The obvious cheap option was to render stages as cards in the conversation. Rejected for a reason specific to this app: stage sub-sessions are isolated, so their tokens are not part of the conversation context. The message list is this app's context ledger — every element in it is something the model is paying for. Putting non-context activity there would misreport where the conversation's tokens are going, which is the one thing the app exists to make legible.

The flooding problem reinforces this (168 ephemeral cards in a list that is also the persistent conversation record), but the context-ledger argument is the load-bearing one: it would still hold even for a two-stage workflow.

The run view therefore replaces the message list inside the chat area while open, rather than being a new split pane. The status bar and input stay put so a long run can still be aborted, and the existing inference-context pane remains usable alongside it. A `respond` stage is the exception: its output genuinely *is* conversation, so the view hands back to the message list when one starts.

## Live only, no persistence

Run state is built in the frontend from the event stream and is lost on refresh, while the run continues on the backend. Persisting runs would need new tables, endpoints, and incremental writes — real work whose only payoff is surviving a refresh and enabling after-the-fact debugging of workflow YAML. The events are shaped so a backend recorder can be added later without changing the view. Deliberately deferred rather than dismissed: authoring workflows is exactly when a replayable run log would help, so this may well be revisited.

## Loops render as one row, not one row per item

A loop stage draws a progress bar plus one status dot per item. Expanding 84 items into 84 rows would make the common case unreadable, and the interesting information — how far along, how many failed, which item is retrying — is denser as a bar and a dot grid. Per-item detail is reachable by clicking a dot.

## Bounded activity retention instead of unbounded logs

Full inner detail (thinking, tool calls, tool results) is kept only for the most recent stage invocations; older ones keep their status and result and are marked as having released their detail. Retaining everything would grow frontend memory without bound on a long loop run — 168 stages each holding a thinking stream. Stage status, results and token counts are small and are kept for the whole run, so the overview never degrades; only the deep detail ages out.

## Stage results are truncated before they are emitted

Some stage products are inherently large: the chunking action returns an entire source file, and a loop aggregate holds every item's output. Emitting those verbatim would duplicate the whole payload over the websocket, so oversized results are replaced by a description of their shape before emission, and loop stages report a compact tally instead of their aggregate. The per-item detail is already carried by the per-item events.

## Two token numbers, both measured

Per stage, the peak context its isolated session reached; per run, the total prompt + generated tokens put through the model across every iteration. Both come from real API measurements, never estimates, consistent with the rest of the app. They answer different questions — how big did this stage get, versus what did this workflow cost to run — and summing the former would be meaningless, so both are shown and labelled distinctly.

## Known gap: nested loops

The loop item/attempt identity an inner stage reports is tracked in a single slot rather than a stack, so a loop nested inside another loop would overwrite its parent's item numbers. No workflow nests loops today. The run view's stage rows handle arbitrary nesting depth; only the item numbering would be wrong.
