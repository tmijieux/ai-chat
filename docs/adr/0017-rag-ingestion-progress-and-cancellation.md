# ADR-0017: RAG Ingestion Progress and Cancellation via WebSocket

**Date:** 2026-08-24
**Status:** Accepted

## Context

ADR-0016's `/rag-index` ran as a single opaque HTTP request — no feedback until it finished, no
way to stop it, and nothing in the backend logs to watch either. For anything beyond a handful of
files this is a bad experience, and the backend embedding call was fully blocking besides.

## Decision

### A dedicated ingestion WebSocket, reusing the agent loop's own pattern

Rather than inventing new plumbing, `POST /api/rag/spaces/{id}/sources/workspace-path/ws` mirrors
`routers/ws.py`'s established shape for the main agent loop: an `asyncio.Queue` collects events,
one task runs the actual work (`ingest_workspace_path`) and pushes progress/done/error onto the
queue, a `send_loop` forwards the queue to the client, a `recv_loop` listens for a `{"type":
"cancel"}` control message, and `asyncio.wait(..., FIRST_COMPLETED)` races the two with a
`_cancel_and_wait` cleanup helper (a local copy of `ws.py`'s, kept local since each router here is
otherwise self-contained). The plain `POST` endpoint from ADR-0016 stays for simple/programmatic
callers that don't need progress.

`ingest_workspace_path` gained an `on_progress(current, total, filename)` callback, called once
per file as it starts — synchronous, just enqueues an event, no coupling to the transport.

### Cancellation is real task cancellation, not disconnect-polling

Sending `{"type": "cancel"}` makes `recv_loop` return, which unblocks `asyncio.wait`, whose
`finally` calls `task.cancel()` on the still-running ingestion task. This interrupts it at its next
`await` (between files, since `provider.embed()` itself is a single blocking call — see below), a
guaranteed, immediate mechanism — deliberately chosen over having the endpoint poll
`Request.is_disconnected()`, which depends on the ASGI server noticing a TCP close promptly and
doesn't fire at all for an explicit in-band cancel message the way this does.

### Explicit commit before the "done"/cancelled event, not on handler return

`get_db_session`'s dependency commits once the route handler function returns — for this endpoint
that's *after* `send_loop`/`recv_loop` have both settled, which is entangled with the connection's
own lifecycle. Testing this directly (via `TestClient`'s websocket support closing the connection
right after receiving "done") reproduced a real bug: the just-ingested rows were lost, apparently a
race between the implicit post-return commit and the connection teardown. Fixed by committing
explicitly inside `run_ingestion()` — once on success, and once in a `CancelledError` handler
before re-raising it (so a cancelled run still keeps whatever it managed to flush) — so persistence
no longer depends on how or when the socket happens to close.

### Embedding offloaded to a thread

`provider.embed()` (fastembed/ONNX) is a blocking CPU call with no internal `await` points. Calling
it directly inside an `async def` would stall the whole event loop — every other request, not just
this ingestion run — for the duration of each batch. Wrapped in `asyncio.to_thread`, mirroring why
`imagegen_pipeline.py`'s `sd-cli.exe` call is similarly offloaded (ADR-0013).

### Frontend: a dedicated RagService, not more logic on ChatService

`RagIndexService` is a thin, single-purpose WebSocket transport (mirrors `AgentService`'s shape:
owns the socket and an `events$` stream, no display state). A separate `RagService` composes it
with `ApiService` to own the actual RAG-command orchestration — resolving the one-space-per-
workspace convention, running an indexed/search command, formatting the result text, and owning
the `activity` signal that drives the progress banner — so that `ChatService.runRagCommand` stays
a thin caller: persist the user's command, call into `RagService`, persist whatever string comes
back as the assistant reply. The banner and its Cancel button read `RagService` directly from
`ChatComponent`, not proxied through `ChatService`.

## Verification

A script opens the ingestion websocket against an isolated temp SQLite DB (never `chat_db.sqlite`)
via `TestClient`, confirms: progress events arrive in order with correct current/total, a full run
produces a `done` event whose source count matches what's actually persisted (this is what caught
the commit-race bug above), and cancelling right after the first progress event produces a
`cancelled` event with fewer sources persisted than a full run — proving cancellation is real, not
just an event that looks like a no-op happened.
