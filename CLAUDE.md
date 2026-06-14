# Project Instructions

## Context

This app is about designing an AI agent do not be surprised if I speak about tools/agent stuff up front and read `CONTEXT.md`. I always speaking about my app and not about Claude unless i mention your name. If i did not explicitely talked about you by mentioning your name, then Starts investigating the code about the mentionned problem.

Always read `CONTEXT.md` at the start of every conversation. It contains the app mission, domain glossary, feature intent, known bugs, and planned changes. Features documented there must not be removed or broken when implementing other features.

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


# CSS
Try to maximize utilisation of functionnal style("tailwind-like")  everywhere css is required  (see styles.scss)
The aim is to understand layout and look just by looking at the template most of the time.
Avoid BEM style.

# Angular 
Generally prefer factorizing in new components rather than <ng-template>
because it helps with keeping component small (in code-behind as well)
It also help the reading comprehension when small component encapsulate 
small non-leaky well-named abstraction, then you dont have to read them.



