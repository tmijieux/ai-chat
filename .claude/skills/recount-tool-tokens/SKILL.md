---
name: recount-tool-tokens
description: Recompute and sync measured_delta token counts for agent tools after editing their descriptions or parameters. Use when a tool schema was edited and measured_delta may be stale.
---

# Recount Tool Tokens

## Quick start

Run from the `backend/` directory:
```bash
cd backend && venv/Scripts/python -m agent.count_tool_tokens
```

## Workflow

1. Run the script and capture output.

2. **Parse MISMATCH lines** — two sections have them:

   - **TOOL_REGISTRY tools** (section `--- verification: measured_delta ... ---`):
     ```
       grep_files: isolated=614, marginal=591, token_count=368 [MISMATCH (got 368, expected 391)]
     ```
     New `measured_delta` = `isolated` value (e.g. `614`).

   - **Always-on tools** (section `--- always-on tools ---`):
     ```
       ask_user_question: isolated_delta=410, token_count=187 [MISMATCH (stored=410, measured=450)]
     ```
     New `measured_delta` = `measured` value (e.g. `450`).

3. **Update each mismatched tool file** — tool name maps directly to its file:
   - `backend/agent/tools/{tool_name}.py`
   - Find the line `measured_delta = <old>` and replace with the new value.

4. Re-run the script to confirm all entries show `[OK]`.

## Notes

- `measured_delta` is the **raw isolated delta** (count with 1 tool − count with 0 tools).
- `token_count` (derived property) = `measured_delta − TOOL_FRAMEWORK_OVERHEAD` — never edit this directly.
- `run_shell.py` uses a conditional: update the right platform constant (`_MEASURED_DELTA_WINDOWS` or `_MEASURED_DELTA_UNIX`).
