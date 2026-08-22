## UI / UX

- **Conversation title update**: compute title update from one sentence generation based on user first message
(could be done by extracting information from working memory compression!!)

## Installation / Setup

- **Installation guide**: write a step-by-step guide to install and run the app from scratch.

- **CLI flag to disable Speech-to-Text**: add a backend command-line parameter (e.g. `--no-stt`) to skip loading the STT model entirely, for faster installs and environments that don't need it.

- **CLI flag for AI model path**: add a backend command-line parameter (e.g. `--model-path`) to override the default path to the local AI model, so users can point to a different model without editing config files.

## Pipeline / Agent

- **Pipeline stage event visibility**: done for YAML workflows — see the Workflow Run View in `CONTEXT.md` and ADR-0009. Still open for `PipelineOrchestrator`, whose hardcoded stages (classify / augment / critique / plan / execute / verify / compile_fix) emit no lifecycle events and so do not appear in the run view. The event shapes are generic enough to cover it; it just needs the same instrumentation.

- **"Context mismatch" indicator after compression**: when the model's compressed context no longer matches the visible conversation (e.g. after an agent session stops mid-run without a response, tool calls and thinking get compressed away on the next session — the model has lost all prior progression), the frontend should show a clear button/badge that lets the user inspect exactly what context the model is actually seeing. This surfaces the divergence before the user is confused by the model apparently "forgetting" work it had already done.

- **`respond` stage redesign** — full plan saved at `docs/handoff_respond_stage_redesign.md`. Fixes a real bug (workflow run view stuck flashing on the respond/synthesis stage, plus a context-overflow crash in `sync-locale-directory`) by making "does this respond stage talk to the user or just produce a value" a fact of the orchestrator's context (top-level vs. nested sub-workflow) instead of something the YAML declares or the backend has to paper over with a session proxy. Supersedes the interim fix already committed in `66049aa`.

- **map-codebase: file-list preview before the scan loop starts** — deferred out of the ADR-0011 work, not started. See handoff doc for details.

- **In-UI editing of a stage's definition/inputs before resuming**: correcting a workflow bug that stopped or failed a run — a stage's `workflow.yaml` definition, or a persisted stage's own input/result JSON — currently requires editing files by hand outside the app before hitting "Resume from here." No in-UI editor exists yet. See ADR-0011 and ADR-0012.

## Local Model Exploration

- **Try ornith-1.5-9B-GGUF**: model on Hugging Face based on Qwen3.5-9B and Gemma, reportedly trained more recently with better benchmark results. Worth evaluating as a replacement for the current local model.

- **Search for other recent small models**: look for other recently-trained models around the same size class (fits current hardware) that might outperform the current local model on benchmarks.

