# Handoff: Persisted Workflow Run History (ADR-0011)

**Date:** 2026-08-18
**Repo:** `C:\Users\tmijieux\ai\my_ai_chat`
**Session focus:** persisted, browsable workflow run history for the custom-workflow engine, plus a round of UX fixes to the run-view tree.

## Where the design and status actually live (don't re-derive, read these)

- **Design + rationale:** `docs/adr/0011-persisted-workflow-run-history.md`
- **Feature description (user-facing):** `CONTEXT.md`, "Persisted run history" paragraph under Workflow Run View
- **Approved implementation plan (phase 1 scope):** `C:\Users\tmijieux\.claude\plans\melodic-crunching-cosmos.md`
- **Remaining/deferred work, tracked tersely:** `todo.md`, under "Pipeline / Agent" — "Resumable/editable workflow runs" and "map-codebase: file-list preview". This doc is the detail behind those two lines.
- **Diff so far:** not committed. `git status` at end of session (uncommitted):
  - Modified: `CONTEXT.md`, `backend/.gitignore`, `backend/agent/custom_workflow.py`, `backend/agent/workflow_coordinator.py` (pre-existing unrelated change from before this session, not touched), `backend/main.py`, `chat-client/src/components/workflow-run-panel/*.{ts,html}`, `chat-client/src/services/api.service.ts`, `chat-client/src/services/workflow-run.service.ts`, `chat-client/src/types/message-types.ts`, `todo.md`
  - New: `backend/agent/workflow_run_recorder.py`, `backend/routers/workflow_runs.py`, `chat-client/src/components/workflow-run-panel/workflow-activity-entry.component.{ts,html}`, `docs/adr/0011-persisted-workflow-run-history.md`
  - **Not committed yet** — the user has been live-testing in the browser and reporting bugs to fix inline; commit once they're satisfied.

## What was built (phase 1, per the plan doc above)

Every workflow run now writes its full execution tree to `backend/workflow_runs/<workflow_name>/<run_id>/` as it runs — directory-per-stage mirroring the runtime call structure (loop items get numbered subdirectories, nesting recursively so nested loops never collide), `meta.json` + `attempt_N.json` per stage instance, `item_result.json` per loop item, `run.json` for top-level run status. A new `GET /api/workflow-runs/{workflow}/{run_id}/node?path=...` endpoint serves any node's content plus a shallow children list. The frontend's stage-list panel is now itself a lazy-expanding tree explorer (no separate "children" view in the detail pane, per explicit user feedback) — clicking a finished node/dot expands its real subtree inline; the bottom pane is a pure content viewer.

Verified end-to-end against real LLM calls (`map-codebase` over `backend/llm`, driven by directly invoking `CustomWorkflowOrchestrator.run()` in-process — see "Testing without touching the live app" below) — nested loops, sub-workflow-style nesting, coordinator vs LLM vs respond stage transcripts all persisted and served correctly.

## Bugs found and fixed during live iteration (all done, none re-verified live after the last few)

Roughly in the order the user hit them:

1. Stage-type label showed generic `"coordinator"` for a `run_script`-backed stage (`bucket_by_depth`) instead of something meaningful — relabeled to `"script"` at the two emission points only (`_display_stage_type` in `custom_workflow.py`), engine dispatch untouched. Verified with a script.
2. Nested loop items rendered as plain rows instead of the same progress-bar+dot-grid the top-level loop uses — fixed in `_buildFetchedRows`/template so any loop's items (live or fetched) render identically.
3. Children order was alphabetical (directory listing order), not execution order — e.g. `chunk_notes_loop` sorted before `chunk_the_file` despite running after it. Fixed by having the recorder stamp a `sequence` field in `meta.json` (assigned once, at first creation) and having the router sort by it; item-number-only directories (no meta.json) sort numerically instead of as strings.
4. The currently-running item's live subtree needs to auto-expand while following the run ("all active nodes must be expanded automatically") — but that auto-expand must stop dead the instant the user navigates to inspect anything else, live or historical (an old finished node must never show a flashing sibling). Implemented via an `isFollowing` flag computed once (`selectedFetchedAddress/selectedExecutionId/selectedLoopItem` all null) and threaded through `_buildStaticRows`.
5. Expanding one loop item left a previously-expanded sibling's subtree visible alongside it (ambiguous which stage belongs to which item) — user's explicit call: **at most one branch expanded at a time, tree-wide**, not just within one loop. Implemented as `_pruneUnrelated(address)` in `workflow-run.service.ts`: keeps only ancestors/descendants of the newly-selected/expanded address, collapses everything else. Applies from every entry point: `selectStage`, `selectLoopItem`, `selectFetchedRow`, `toggleExpand`.
6. Large stage results (e.g. `enumerate`'s file list) showed truncated even when fetched from disk — turned out `_truncate_stage_result` was being applied *before* the event ever reached the recorder, so the persisted copy was truncated too, not just the live websocket copy. Fixed: `_emit_stage_exit`/`loop_item_exit` now emit the full result; `_RecordingSession.emit()` truncates only the copy it forwards to the real session, after the recorder has already seen the full one.

**Items 4–6 (the last three) were fixed but never re-verified live in the browser** — each `ng build` passed cleanly, and the logic was reasoned through carefully, but the user was iterating fast enough that I didn't re-run a full LLM verification pass after each one. **First thing to do in a fresh session: have the user (or drive yourself, see below) re-check these three in the actual run view before doing anything else.**

## Testing without touching the live app

Two important constraints learned the hard way this session, both now effectively hard rules:

- **Never invoke workflows through the conversation/websocket layer for testing** — it creates real rows in the user's actual `chat_db.sqlite` (their real conversation history), which the user does not want touched even for throwaway tests. Clean up immediately (`DELETE /api/conversations/{id}`) if this happens by accident.
- **Instead, invoke the orchestrator directly, in-process, bypassing the DB entirely.** A working throwaway script for this exists (or can be recreated) — see below.

Pattern that worked (also handles `tool_confirm` by using `session.mode = "auto"`, which rule-based-approves in-workspace file writes without needing to answer confirm prompts):

```python
# run from backend/, with PYTHONPATH=. set (needed if the script lives outside backend/)
from agent.agent import AgentSession
from agent.custom_workflow import CustomWorkflowOrchestrator
from agent.workflow_loader import load_workflow
from pathlib import Path
import asyncio

async def main():
    wf = load_workflow(Path("workflows/map-codebase"))
    session = AgentSession()
    session.mode = "auto"
    session.working_directory = "C:/Users/tmijieux/ai/my_ai_chat"
    orchestrator = CustomWorkflowOrchestrator(wf, session.working_directory, tools=[])
    messages = [{"role": "user", "content": "scan backend/llm, write the map to tmp_test_output/codebase-map, max_fold_depth 2"}]
    task = asyncio.create_task(orchestrator.run(session, messages[0]["content"], messages))
    while True:
        event = await session.outbound.get()
        if event.get("type") == "tool_confirm":
            session.resolve_confirm(event["tool_id"], True, None)
        if event.get("type") in ("done", "error"):
            break
    await task

asyncio.run(main())
```

Backend + llama-server were left running for this session (`python serve.py` from `backend/`, port 8000; llama-server auto-starts on port 8080 via the lifespan hook). Check `curl http://127.0.0.1:8000/api/status` before assuming they're still up — the user was asked whether to shut them down and that question is still open (see below).

`backend/llm` (4 small files) is a good scan target for a fast, cheap map-codebase run — a full run takes several minutes with the local 9B model even for that small a scope, so don't default to a bigger target without a reason. Always clean up `backend/workflow_runs/map-codebase/*` test runs and any `tmp_test_output/` the workflow itself writes into the repo tree afterward — the former is gitignored but accumulates junk, the latter is not gitignored and must not be left behind.

## Open questions for the user

1. Should the backend/llama-server processes started this session be left running or shut down? (Asked, not yet answered.)
2. Ready to commit the accumulated diff, or still iterating on more bugs first?

## Remaining planned work (not started this session)

### 1. `append_jsonl_record` `key_field` parameter

Currently a blind append (`backend/agent/workflow_coordinator.py`, `_append_jsonl_record`). Add an optional `key_field` input: when given, read the existing file, replace any line whose `record[key_field]` matches, append otherwise, rewrite the whole file (fine at this scale — hundreds of records, not millions). Omitted, keep current pure-append behavior (default, backward compatible). This is a prerequisite for resumability (see below) — without it, redoing a stage that already appended a row (e.g. map-codebase's `record_file`) duplicates it. Independent, small, could be done standalone without the rest of resumability.

### 2. Full resumability (design already settled via a long grilling session, captured in ADR-0011's "Deferred: resumability" section and originally worked out in this conversation's history — re-read that ADR section before starting, it has the load-bearing reasoning)

Not yet started at all. Pieces, in rough dependency order:

- `append_jsonl_record` `key_field` (above) — do first, it's a prerequisite and stands alone.
- Reconstruct the slot registry from disk on resume: walk the persisted tree in execution order, `slots[stage.name] = <persisted result>` up to a chosen resume point.
- Resume overwrites in place (same `run_id`) — confirmed by the user, not a fork-a-new-run design.
- Invalidation model (confirmed with the user, non-obvious — worth re-reading the ADR before implementing): editing a node invalidates everything after it in its own sequence; a loop's aggregate output and everything after the loop invalidate once any of its items change; **sibling loop items are independent by default and are NOT re-run** on resume — this needs an explicit per-stage/loop opt-out (something like `assume_independent: false`) for workflows whose inner stages have real cross-item file side effects, since the engine can't know this on its own.
- `run.json`'s failure-path tracking is **already done** (`WorkflowRunRecorder.last_failed_address`, `mark_failed()`) — don't redo it, just wire the frontend to jump there when resuming a failed run.
- v1 editing is hand-editing the persisted JSON directly (no in-UI editor) — confirmed scope, don't build an editor UI unless asked.

### 3. map-codebase: file-list preview before the scan starts

User's original ask, deferred out of this session's scope early on and never revisited. `enumerate_files` (`workflow_coordinator.py`) already returns `size_bytes` per file — nothing currently surfaces that list to the user before the (potentially hours-long) per-file summarize loop begins, so a huge file that slipped past the binary/lockfile exclusion filters wouldn't be caught until it's already being chunked. Proposed fix (not yet built): a `request_confirm` gate on the `enumerate` stage/action, showing the file list sorted by size descending, reusing the exact confirmation mechanism `run_compile` already uses (`session.request_confirm` in `workflow_coordinator.py`). Small, independent, doesn't touch persistence/resumability at all.

## Suggested skills for the next session

- **`run`** — before doing anything else, use this to actually launch the app (or reuse the already-running backend/llama-server if still up) and drive the run view in a real browser, since the last three bug fixes were never visually confirmed. This project doesn't have a bundled `run` skill yet in `.claude/skills/` — expect to fall back to the server pattern (`python serve.py` from `backend/`, `npm start` from `chat-client/` if the Angular dev server is also needed) and consider recommending `/run-skill-generator` afterward per the `run` skill's own guidance.
- **`grill-with-docs`** — if resumability's invalidation model or the `assume_independent` design needs revisiting once implementation starts surfacing edge cases the earlier design pass didn't anticipate. The existing design in ADR-0011 was reached through a long grilling session — don't casually deviate from it without the same rigor.
- **`code-review`** — once the user is done iterating and ready to commit, run this over the full diff before committing (it hasn't been reviewed yet, only built and spot-verified).
