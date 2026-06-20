## UI / UX

- **Conversation title update**: compute title update from one sentence generation based on user first message
(could be done by extracting information from working memory compression!!)

## Installation / Setup

- **Installation guide**: write a step-by-step guide to install and run the app from scratch.

- **CLI flag to disable Speech-to-Text**: add a backend command-line parameter (e.g. `--no-stt`) to skip loading the STT model entirely, for faster installs and environments that don't need it.

- **CLI flag for AI model path**: add a backend command-line parameter (e.g. `--model-path`) to override the default path to the local AI model, so users can point to a different model without editing config files.

## Pipeline / Agent

- **Pipeline stage event visibility**: all events tagged `_pipeline_stage` are silently dropped by the frontend (`chat.service.ts:524`) — tool calls, results, and thinking inside workflow LLM stages and sub-agents are invisible to the user. Need to design a way to surface this (collapsible stage activity section, live log panel, etc.). Also covers sub-agent visibility.

- **"Context mismatch" indicator after compression**: when the model's compressed context no longer matches the visible conversation (e.g. after an agent session stops mid-run without a response, tool calls and thinking get compressed away on the next session — the model has lost all prior progression), the frontend should show a clear button/badge that lets the user inspect exactly what context the model is actually seeing. This surfaces the divergence before the user is confused by the model apparently "forgetting" work it had already done.

