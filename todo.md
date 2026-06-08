## UI / UX

- **Chat auto-scroll**: auto-scroll when at bottom, disable on scroll-up, re-enable on return to bottom (spec: [[Chat Auto-scroll]]) TO BE TESTED BY HUMAN
- **Tool result display improvements**: grep colored line view, compact write/edit/shell display, key-field summary for others (spec: [[Tool Result Display]]) TO BE TESTED BY HUMAN

- **Edit_file diff coloring**: backend computes unified diff with file line numbers; frontend renders red/green colored diff in the confirmation card (spec: [[Edit File Diff Preview]])  TO BE TESTED BY HUMAN


- i still want to  see content of files/tool results that are marked as compressed in the ui

- **Vision / image input**: paste or drag image into chat — vision projector tensors are bundled in the GGUF (same as Ollama); need to verify if current llama-server build auto-detects them or needs an explicit flag before implementing frontend/backend (spec: [[Vision / Image Input]])

- **Copy raw message to clipboard**: add button to the ⋮ action menu on each message (currently only code blocks have copy via Prism)

- **GPU/CPU memory panel**: VRAM used/free + layer distribution from `nvidia-smi`, exposed via a backend endpoint and shown in the UI

- **Conversation title update**: compute title update from one sentence generation based on user first message

## Context compression pending

- **Active file tracking**: file currently being written/edited must never be summarized in Stage 2 (see [[Post-Iteration Sub-Agent]])

- **Per-iteration compression**: run compression/summarization when required (file too big)

## Backend / Infrastructure

- **Fix uvicorn reload on Windows with ProactorEventLoop** — WinError 87 on socket accept when using reload=True. Pattern: parent spawns child subprocess (env flag `RELOADER=yes`), child runs `uvicorn.Server` directly (no `reload=True`) + `watchfiles.awatch` concurrently, sets `server.should_exit = True` on file change, exits with code 3, parent restarts. See `my_quart_reloader()` in the other quart project for the analog.

## Bugs

- **`is_default` uniqueness**: setting a new default prompt must atomically clear the previous one (see [[is_default]])
