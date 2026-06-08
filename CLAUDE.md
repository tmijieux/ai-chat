# Project Instructions

## Context

Always read `CONTEXT.md` at the start of every conversation. It contains the app mission, domain glossary, feature intent, known bugs, and planned changes. Features documented there must not be removed or broken when implementing other features.

## Off-limits

Do not read `catchall.py` — it is currently unused and irrelevant.

## Code Style

- **No abbreviations** in variable or parameter names (e.g. `estimated_tokens` not `est_tokens`).
- **No implicit boolean conversions** — use explicit comparisons: `if x is None` or `if x == ""`, never `if not x` for strings or optional values.
- **No boolean lazy evaluation for fallbacks** — never `x or default` to substitute a missing value; use explicit `if x is None` checks instead.
- **Docstrings on every function** — document both purpose and important details about implementation. One sentence is enough for simple helpers.
