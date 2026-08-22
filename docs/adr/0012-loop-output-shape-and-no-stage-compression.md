# ADR-0012: Loop Aggregates Split Input From Result; Workflow Stages Never Compress

**Date:** 2026-08-22
**Status:** Accepted

Resuming a real (large) map-codebase scan appeared to "freeze" — no error, no further events, the run stuck on "running" forever. Root cause, found via live testing: `chunk_notes_loop`'s per-chunk snapshot always included the raw chunk (a chunk's full source text), and `summarize_file`'s prompt re-embedded every chunk's raw text once per chunk — defeating the whole point of chunking and pushing that stage's isolated session over the compression threshold on any real-sized file. Once triggered, the stage silently hung forever: a workflow stage's isolated session shares the main chat loop's compression machinery (emit "compressing", await a reply) but nothing wires that round-trip for an isolated stage — the wait is never resolved.

## Decision

1. A loop's per-item snapshot is always `{input, success, result}` — the raw item it iterated on, kept structurally separate from what its inner stages produced. The loop's own aggregate (consumed by later stages' templates) always carries this whole shape; a consuming template chooses what it wants via a small path-projection syntax (`some_loop.items[].result`), rather than the loop declaring in advance what to strip.
2. A workflow stage's isolated session never compresses. If a stage's context overflows anyway, it fails immediately with a clear error instead of attempting to compress — the failure is resumable like any other (ADR-0011), once a human corrects the stage's inputs or definition.

## Why a per-consumer projection, not a per-loop "exclude the input" flag

The first design excluded the raw item from a loop's aggregate via a stage-level boolean (mirroring the existing inner-stage flag that already lets one inner stage opt its own result out of a loop item's snapshot). It worked, but flipping its default — needed so `chunk_notes_loop` didn't have to remember to opt out — meant every *other* loop whose aggregate is genuinely consumed downstream had to be annotated defensively just to keep working: four loops, across three unrelated workflow files, none of which had anything to do with the bug being fixed. That ripple is a tell the choice belonged to the wrong party. Instead, a loop's aggregate now always carries everything, and each downstream consumer — a template, or a script reading the same dict directly — picks what it needs. Same principle as a database storing full rows and letting `SELECT` choose columns, rather than the table deciding what to drop because one caller didn't need it.

The projection syntax itself — a `field[]` path segment: resolve `field` as a list, then map the rest of the path over each element — was chosen over a Python-style function call (e.g. `get_array_items(loop.items, "result")`) because it slots into the existing dotted-path resolver with one extra check per segment, instead of needing an actual expression parser (argument lists, quoted-string literals, nesting) that doesn't exist anywhere in this template language yet. A full template engine (Jinja) was also considered and set aside as unneeded for what is still a very small expression language — worth revisiting only if requirements grow past simple path projection.

## Why fail instead of trying to make compression work in a stage

Wiring up mid-stage compression for an isolated session was considered and rejected: chunking exists specifically so a stage's own context never needs it in the first place. A stage that overflows anyway means its inputs or definition are wrong for the data it's actually seeing — something a human needs to look at, not something the framework should paper over by compressing, which would also risk discarding exactly the context needed to diagnose why the stage was oversized. Failing loudly, in a run that's already resumable, costs nothing: correct the workflow.yaml or persisted inputs, then "Resume from here."

## Migration

The loop-item shape change is a breaking, on-disk format change (`item_result.json`'s `item` key becomes `input`; formerly-flattened stage outputs move under `result`). A one-off script, `backend/migrate_item_result_format.py`, walks a workflow-runs directory and upgrades old-format files in place — idempotent, dry-run by default. Delete it once no old-format runs remain on disk.

See `~/.claude/plans/concurrent-brewing-popcorn.md` for the implementation plan and `CONTEXT.md`'s "Context overflow inside a stage" (under Workflow Run View) for the user-facing behavior.
