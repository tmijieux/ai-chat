# Project Instructions

## Context

The  app in this workspace is an AI agent. Do not be surprised if I speak about tools/agent stuff up front and read `CONTEXT.md`. I always speaking about my app and not about Claude unless I mention your name. If I did not explicitely talked about you by mentioning your name, then Starts investigating the code about the mentionned problem.

*ALWAYS* read `CONTEXT.md` at the start of every conversation. It contains the app mission, domain glossary, feature intent, known bugs, and planned changes. Features documented there must not be removed or broken when implementing other features.

CONTEXT.md is for storing features description of features we want to keep in app and a glossary of term and concepts.
When implementing features ensure that you are not removing features mentionned in CONTEXT.md  

Do not put implementation details in CONTEXT.md. 
Do not mention code snippets/method name about implementation in CONTEXT.md.

Implementation decisions belong in ADRs in `docs/adr/`. Use the `grill-with-docs` skill conventions: only create an ADR when the decision is hard to reverse, surprising without context, and the result of a real trade-off. Number sequentially (`0008-slug.md`). See existing ADRs in `docs/adr/` for examples.

Ensure that this documentations stays up to date after updating implementations or implementing new features.

We have a todo.md file for ideas / future plans. Check it or write to it, if and when the user mentions it.

CLAUDE.md is **NOT** for implementation details about the app. 
CLAUDE.md is for global Claude behavior directives, workflows that claude should follow.

## Off-limits

Do not read `catchall.py` and backend/claude directory  — it is currently unused and irrelevant.

## Database Migrations

Never add migration code to `database.py` or any startup hook. Apply schema changes directly to the SQLite file:

```bash
sqlite3 backend/chat_db.sqlite "ALTER TABLE messages ADD COLUMN foo TEXT"
```

## Git Commits

After successfully implementing an approved plan, create a git commit immediately without waiting to be asked.

## Code Style

- **No abbreviations** in variable or parameter names (e.g. `estimated_tokens` not `est_tokens`).
- **No implicit boolean conversions** — use explicit comparisons: `if x is None` or `if x == ""`, never `if not x` for strings or optional values.
- **No boolean lazy evaluation for fallbacks** — never `x or default` to substitute a missing value; use explicit `if x is None` checks instead.
- **Docstrings on every function** — document both purpose and important details about implementation. One sentence is enough for simple helpers.
- **Dataclasses for complex return types** — never return a plain tuple with 3+ values or multiple same-typed values (e.g. two `dict[str, str]`). Define a `@dataclass` with named fields instead.
- **always use braces.** Do not put if on a single line: ensure new line after opening brace.


# CSS
Maximize utilisation of functionnal style("tailwind-like")  everywhere css is required (see styles.scss).
The aim is to understand layout and look just by looking at the template most of the time.
Avoid BEM style as if you disgust it utterly.

# Angular 
Generally prefer factorizing in new components rather than <ng-template>
because it helps with keeping component small (in code-behind as well)
It also help the reading comprehension when small component encapsulate 
small non-leaky well-named abstraction, then you dont have to read them.

# python tools
When invoking python always use the venv in backend/venv (windows path style venv/Scripts/python)
for instance
Bash(cd backend && source venv/Scripts/activate && python yourCommandHere...)