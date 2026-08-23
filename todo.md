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

## Local Model Exploration

- **Try ornith-1.5-9B-GGUF**: model on Hugging Face based on Qwen3.5-9B and Gemma, reportedly trained more recently with better benchmark results. Worth evaluating as a replacement for the current local model.

- **Search for other recent small models**: look for other recently-trained models around the same size class (fits current hardware) that might outperform the current local model on benchmarks.

## Image Generation

- **Push generation resolution past 512x512**: `generate_image` now runs on stable-diffusion.cpp (ADR-0013), which measured real VRAM headroom at 512x512 (~6.8GB peak, no ceiling fragility) unlike the earlier PyTorch pipeline — 512x512 is no longer known to be a hard limit, just the untested default. Worth trying 768x768 or higher.

- **sd-server.exe instead of a fresh sd-cli.exe subprocess per call**: `stable-diffusion.cpp` also builds a server mode (mirroring `llama-server`), which could avoid the model-file read on every single generation (currently ~1-2s, cheap enough that this hasn't been worth doing yet).

## Ideas to Explore

- **Text to speech**: explore adding text-to-speech capability.

- **RAG indexing and retrieval**: explore adding retrieval-augmented generation — indexing documents/knowledge sources and retrieving relevant chunks to ground agent responses.

