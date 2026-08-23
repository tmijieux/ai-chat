## UI / UX

- **Conversation title update**: compute title update from one sentence generation based on user first message
(could be done by extracting information from working memory compression!!)

## Installation / Setup

- **Installation guide**: write a step-by-step guide to install and run the app from scratch.

- **CLI flag to disable Speech-to-Text**: add a backend command-line parameter (e.g. `--no-stt`) to skip loading the STT model entirely, for faster installs and environments that don't need it.

- **CLI flag for AI model path**: add a backend command-line parameter (e.g. `--model-path`) to override the default path to the local AI model, so users can point to a different model without editing config files.

## Pipeline / Agent

- **Retire `PipelineOrchestrator`**: rather than instrumenting its hardcoded stages (classify / augment / critique / plan / execute / verify / compile_fix) for run-view visibility, replace it with an equivalent YAML workflow eventually, since YAML workflows already get this for free — see the Workflow Run View in `CONTEXT.md` and ADR-0009.

- **map-codebase: file-list preview before the scan loop starts** — deferred out of the ADR-0011 work, not started. See handoff doc for details.

- **In-UI editing of a stage's definition/inputs before resuming**: correcting a workflow bug that stopped or failed a run — a stage's `workflow.yaml` definition, or a persisted stage's own input/result JSON — currently requires editing files by hand outside the app before hitting "Resume from here." No in-UI editor exists yet. See ADR-0011 and ADR-0012.


## Pipeline / Agent

- **RAG agent tool wiring**: backend indexing/retrieval infrastructure for [[RAG Space]] exists (ADR-0015) but nothing lets a running agent query a space yet — needs a tool (e.g. analogous to `explore_codebase`) and a decision on how a conversation selects which space(s) it can search.

- **RAG spaces UI**: no frontend exists yet for creating/managing RAG spaces or their documents — only the backend API (`/api/rag/...`). Needs a settings-page-style CRUD surface, similar to system prompts.

- **RAG image/PDF ingestion**: current RAG ingestion is text-only (paste, uploaded text/markdown, workspace path). Image and PDF ingestion were explicitly deferred.

## Ideas to Explore

- **Text to speech**: explore adding text-to-speech capability.

- **RAG GPU-based embedding**: current RAG embedding runs CPU-only (fastembed) to avoid VRAM contention with the chat model — worth exploring a GPU-based embedding provider later if it meaningfully speeds up bulk-indexing a large repo, even at the cost of temporarily stopping the chat llama-server like `generate_image` already does. `EmbeddingProvider` was designed swappable for this.

