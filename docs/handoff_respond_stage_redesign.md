# Handoff: `respond` stage redesign — mode decided by orchestrator context, not YAML

## Context

A workflow's `respond` stage (`type: respond`) is meant to hand the run back to being a normal
conversation turn — it builds on the real message history plus an injected summary of what the
workflow did, and lets the model reply in free prose (unlike every other stage type, which is
bounded and must call a `finish_tool` to produce structured data).

Bug report that started this: after `/sync-locale-directory` finished, the run view swapped back
to chat as expected, but the reply never streamed, compression fired immediately, and reopening
the run view showed the `synthesis` (respond) stage stuck flashing "running" forever.

## Root causes found

1. **Context overflow.** `synthesis`'s `message_suffix` interpolated `{{sync_loop_summary.items}}`,
   which held the *entire* internal slot registry of the nested `translate-locale` sub-workflow
   invocation for every locale file — every chunk's raw source text and full translated text,
   duplicated per file. For a 3-file run this came out to 141,754 chars (~35k tokens), pushing the
   prompt to 35,937 tokens against the 32,768 limit → `llama-server` 400 → the respond stage's
   generation produced nothing at all. That's why "the reply never streamed": there was no reply.

2. **`done`/`error` leak.** `_run_respond` calls `run_agent` directly on the real, untagged,
   top-level `AgentSession`. `run_agent` always emits its own untagged `{"type": "done"/"error"}`
   when it finishes generating — correct for a normal standalone chat turn, but for a respond
   stage that's just one stage inside a bigger workflow. Both `ws.py`'s websocket relay
   (`_send_events_from_agent_to_frontend`) and `workflow-run.service.ts` treat *any* untagged
   `done`/`error` as "the whole run is over": the relay stops forwarding and cancels the
   orchestrator task right there, before the respond stage's own `stage_exit` — and the
   workflow's real final `done` — ever get sent. That's why the run view shows it stuck flashing.

## Interim fix already committed (commit `66049aa`) — to be unwound/superseded by this plan

- `backend/agent/custom_workflow.py`: `_RespondStageSession`, a session proxy that swallows
  `run_agent`'s internal `done`/`error` before it reaches the real session. Works, but the user's
  read (correctly) is that it's papering over a side effect instead of not producing it — "hiding
  shit behind more shit."
- `backend/agent/custom_workflow.py`: `_run_sub_workflow` no longer stores a `type: workflow`
  sub-stage's entire internal slot registry into the parent's slots — it filters
  `_run_stage_sequence`'s per-stage outcomes down to whichever look like a loop tally
  (`item_total`/`succeeded`/`failed`). Functional, but it's a blunt filter standing in for what
  should be a real per-item summary (see design below).
- `CONTEXT.md`: "Locale Directory Sync Workflow" was reworded to say the synthesis reports counts
  only, never translated content. Still true after this redesign — the *mechanism* changes, not
  the user-visible behavior.

**When resuming:** decide whether to build on top of these or revert them first (`git log` /
`git show 66049aa` on `backend/agent/custom_workflow.py`) — the design below replaces both pieces,
so the cleanest path is probably to revert `66049aa`'s changes to `custom_workflow.py` and build
the real design directly, keeping the `CONTEXT.md` wording (still accurate).

## Agreed design

### 1. `respond` is not "a stage like the others"

Every other stage type is genuinely a step *inside* the workflow: isolated slot registry, bounded
by a `finish_tool`. `respond` is different — it's a free-form generation (run until the model
gives a plain-text reply, tools allowed, no forced finish tool). What's actually optional is
**where its output goes**, not what it fundamentally is:

- **Wired to the live conversation** — runs on the real session, streams into chat, and is the
  workflow's terminal act (nothing else should happen after it).
- **Accumulator mode** — runs isolated (tagged, like every other stage type), and its output
  becomes `slots[stage.name]`, exactly like any other stage's result. No constraint on position or
  count — can be inside a loop, can repeat, can be a sub-workflow stage feeding its parent.

### 2. Mode is decided by orchestrator context, not by a YAML flag

This isn't authored per-stage. It falls out of which method drives the orchestrator instance —
a fact that already exists in the code today, just never given meaning:

- **Top-level**: `CustomWorkflowOrchestrator` constructed once in `ws.py`, driven via `.run()`.
  Its `respond` stage(s) run wired-to-session.
- **Nested**: constructed inside `_run_sub_workflow`, driven directly via `_run_stage_sequence()`
  (bypassing `.run()` entirely, as it already does today). Its `respond` stage(s) run in
  accumulator mode, automatically, with zero authoring effort in the sub-workflow's own YAML.

Consequence: `sync-locale-directory` no longer *needs* to `skip_stages: [..., synthesis]` when
invoking `translate-locale` — running it is now safe by construction (it can't talk to the user,
can't end the run). Left running, its `respond`/`synthesis` stage would produce a natural,
model-written one-line summary of that file's translation, landing in
`slots["translate_delta"]["synthesis"]` — which is a much better per-item summary for the parent's
loop than the loop-tally filter from the interim fix. `skip_stages` remains a general mechanism a
caller can still use if they don't want that extra LLM call at all — just no longer required for
correctness.

### 3. Validation only applies to top-level runs, checked at run-time (not load-time)

Because the *same* workflow file can be perfectly valid as a sub-workflow (any number/position of
accumulator-mode `respond` stages) and only needs the stricter rule when actually invoked
top-level, this can't be a static `workflow_loader.py` check. It has to run at the start of
`.run()`, scanning `self._workflow.stages` (top-level list only — not inside loop inner stages,
not as a branch target):

- **Zero `respond` stages** → inject a default one as the final stage — generic prompt along the
  lines of "Summarize what the workflow did and its outcome for the user." (User's call: fallback,
  not a load error.)
- **Exactly one, and it's last** → proceed normally.
- **Anything else** (not last, more than one, one nested inside a loop) → emit a plain
  `{"type": "error", "message": ...}` describing the problem (name the workflow, name the
  problem) and return — no `workflow_start`, no stage attempts. The workflow stays loadable and
  listable via `GET /api/workflows` regardless (user's explicit ask: visible in the API, error
  only surfaces on invocation, in chat).

### 4. `agent.py` does not change at all

No proxy, no extracted loop for the top-level case. `run_agent` keeps meaning exactly what it
always meant — because it's now only ever called from the one place (top-level respond, guaranteed
last by the validation above) where "this call finishing is the whole run finishing" is actually
true.

### 5. Frontend hardening (`chat-client/src/services/workflow-run.service.ts`)

Generic, not respond-specific: when the top-level `done`/`error` arrives (the existing
`event._pipeline_stage === undefined` branch in `_handle`), finalize whichever execution is still
marked `running` instead of only nulling `current_execution_id`. This is what actually fixes
stuck-flashing — there's no guarantee of ever seeing an explicit `stage_exit` for the handoff
stage, and none is needed once the run is known to be over.

## Implementation plan (file by file)

### `backend/agent/custom_workflow.py`

- Revert the `66049aa` proxy (`_RespondStageSession`) and the loop-tally-only filtering in
  `_run_sub_workflow` (superseded by accumulator-mode respond, see below) — restore
  `slots[stage.name] = sub_slots` there, OR keep collecting `outcome_sink` but store the whole
  per-stage outcome map rather than filtering to loop tallies (moot once nested `respond` produces
  its own natural summary — re-evaluate once accumulator mode exists, since at that point
  `sub_slots["synthesis"]` alone may be all the caller actually wants).
- Add an "am I top-level" fact to `CustomWorkflowOrchestrator` — e.g. a constructor flag defaulting
  to `True`, with `_run_sub_workflow` explicitly constructing its child with `is_top_level=False`.
- `_run_by_type`'s dispatch of `stage.type == "respond"` branches on that flag:
  - **top-level**: today's `_run_respond` logic (build `working_messages` with the
    `message_suffix` injection, call `run_agent` directly on the real `session`) — but treat it as
    genuinely terminal: no `stage_exit` needs to follow it in the usual generic-dispatch sense,
    and `_run_workflow`'s manual final `session.emit({"type": "done", ...})` goes away for the
    success path, since respond's own `run_agent` call already provides the one true `done`.
  - **nested**: same message-building, but run through an *isolated, tagged* path (reuse
    `run_stage`'s tag-and-forward wrapper, or a variant of it) instead of the real session, with no
    forced `finish_tool` — needs a loop that runs until the model gives a plain-text reply (same
    shape as `run_agent`'s loop) but on a session whose events are tagged so `ws.py`/the run view
    never mistake its `done`/`error` for anything terminal. **This is the one open design question
    to resolve before coding** — `run_stage` as it exists today assumes a `finish_tool`-driven
    stage; a free-form "loop until plain text" stage under a tagged session doesn't have one.
    Whatever the model finally answers becomes `slots[stage.name]`.
- `run()` gains the validation/default-injection logic described above. Important: don't mutate
  `self._workflow.stages` in place when injecting the default — `WorkflowDefinition` instances are
  cached process-wide (`_sub_workflow_cache`) and shared across invocations; build a local
  effective stage list for this run only.
- `_serialize_stage_nodes(...)` (called for the `workflow_start` event) needs to serialize the
  *effective* (possibly default-respond-augmented) stage list, not the raw cached one, so the run
  view's up-front plan includes the injected stage.

### `backend/agent/workflow_loader.py`

- Likely no changes if validation stays fully run-time in `custom_workflow.py` — double-check
  nothing else assumes today's "respond is wherever the YAML put it" shape.

### `backend/workflows/sync-locale-directory/workflow.yaml` (optional follow-up)

- Consider dropping `synthesis` from `translate_delta`'s `skip_stages`, and have the synthesis
  stage's template reference the nested summary (`{{sync_loop_summary.items}}` would then contain
  each file's natural-language blurb rather than raw counts) — not required for correctness, purely
  a UX improvement once accumulator mode exists.

### `chat-client/src/services/workflow-run.service.ts`

- In `_handle`, the branch handling `(event.type === 'done' || event.type === 'error') &&
  event._pipeline_stage === undefined`: alongside setting `current_execution_id: null`, finalize
  any execution whose `status` is still `'running'` to `'done'` or `'error'` (matching the run's
  own outcome).

### Docs

- New ADR documenting this decision (mode-by-orchestrator-context, run-time-only validation,
  default-respond fallback) — it's hard to reverse and non-obvious without the history above.
- Add a note to `docs/adr/0010-workflow-sub-invocation.md`'s "Skipping stages is explicit, not
  inferred" section: skipping a nested `respond` stage is now optional (accumulator mode makes it
  safe to leave running), not required for correctness as it effectively was before.
- `CONTEXT.md`: mention (in feature/glossary language, no implementation detail) that a called
  sub-workflow's own respond stage, if not skipped, contributes a natural-language result to the
  caller instead of talking to the user directly.

## Open questions to resolve when picking this up

- Exact mechanism for nested/accumulator-mode respond: adapt `run_stage` to support a
  finish-tool-free "loop until plain text" mode, or write a small parallel helper for it.
- Whether the default respond prompt (for workflows missing one) should be configurable per
  workflow or a single hardcoded generic string is enough.
- Re-confirm `_run_workflow`'s removed final `done` emission can't be reached by some stage-graph
  shape that finishes without ever executing the (guaranteed-present-when-top-level) respond stage
  — should be impossible given the validation, but worth a deliberate check once branches/jumps are
  involved.

## Suggested skills for the resuming session

- `/plan` (or the Plan agent) to firm up the exact diffs before touching files — this touches core
  session/event plumbing and the user has been explicit about wanting the problem re-confirmed
  before any solution shape.
- `/code-review` after implementing, given the blast radius (websocket relay, run view, every
  workflow's respond stage).
