## UI / UX

- **Chat auto-scroll**: auto-scroll when at bottom, disable on scroll-up, re-enable on return to bottom (spec: [[Chat Auto-scroll]])
- **Tool result display improvements**: grep colored line view, compact write/edit/shell display, key-field summary for others (spec: [[Tool Result Display]])
- **Edit_file diff coloring**: backend computes unified diff with file line numbers; frontend renders red/green colored diff in the confirmation card (spec: [[Edit File Diff Preview]])
- **Vision / image input**: paste or drag image into chat — vision projector tensors are bundled in the GGUF (same as Ollama); need to verify if current llama-server build auto-detects them or needs an explicit flag before implementing frontend/backend (spec: [[Vision / Image Input]])
- **Copy raw message to clipboard**: add button to the ⋮ action menu on each message (currently only code blocks have copy via Prism)
- **GPU/CPU memory panel**: VRAM used/free + layer distribution from `nvidia-smi`, exposed via a backend endpoint and shown in the UI

## Context compression pending

- **Active file tracking**: file currently being written/edited must never be summarized in Stage 2 (see [[Post-Iteration Sub-Agent]])
- **Conversation title update**: wire the compression summary produced by Stage 1 to update the conversation title in the DB
- **Per-iteration compression**: run compression after each agent iteration, not just at the end of the run

## Bugs

- **`is_default` uniqueness**: setting a new default prompt must atomically clear the previous one (see [[is_default]])
- **Thinking-only assistant messages**: content = "" messages are saved to DB and add noise; filter them out
