# ADR-0002: Git-tracked prompt history with DB as runtime store

**Date:** 2026-05-20  
**Status:** Accepted

## Context

The system prompt was a hardcoded constant in `agent/system_prompt.py`. No versioning, no history, no way to annotate prompt effectiveness. The `system_prompt_templates` DB table was added but not yet seeded.

## Decision

Two-layer persistence:

1. **`prompts/` directory (git-tracked)** — one `.md` file per prompt version with YAML frontmatter (`name`, `date`, `is_current`, `category`, `effectiveness_notes`). This is the historical record. Files are never edited after creation; a new version = a new file with the old one's `is_current` set to `false`.

2. **DB (`system_prompt_templates` table)** — runtime store. On backend startup, the file marked `is_current: true` is seeded as a `SystemPromptTemplate` row. Seed is idempotent: skipped if a row with that `name` already exists. The DB is the live source of truth during runtime.

The hardcoded `SYSTEM_PROMPT` constant is deleted once migrated.

## Alternatives considered

**DB only** — no file history. Rejected: wiping the DB loses all prompt history; no git diff on prompt changes.

**File only** — read `.md` files directly on every request. Rejected: adds filesystem I/O to the hot path; makes per-conversation activation awkward.

**Manual import** — no auto-seed; user runs a script to populate DB. Rejected: easy to forget after a DB wipe; adds friction with no benefit.

## Consequences

- Prompt engineering history is git-diffable.
- Wiping the DB is safe — restart re-seeds from the current file.
- The `prompts/` directory is the place to annotate effectiveness after experiments.
