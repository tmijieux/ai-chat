# ADR-0013: Local Image Generation (Flux.1-schnell)

**Date:** 2026-08-23
**Status:** Accepted

## Context

The machine has one GPU usable for both chat inference and image generation: an RTX 4070 Laptop, 8GB VRAM (the Intel Arc iGPU present is used for Whisper only and has no meaningful diffusion tooling on this hardware). The local chat model (`llama-server`, `-ngl 99`) already uses most of that 8GB when loaded, so image generation and chat inference cannot run concurrently — they must take turns on the same card.

## Decision

### Model: Flux.1-schnell, GGUF-quantized

Flux.1-schnell (Apache-2.0, distilled to 1-4 denoising steps) was chosen over SDXL/SD-Turbo variants for quality per step, and over Flux.1-dev because dev needs classifier-free guidance (roughly 2x the compute per step) for a quality gain the distilled schnell mostly already captures at this quantization level. Weights are GGUF-quantized, mirroring how the chat LLM itself is already served — `flux1-schnell-Q4_K_S.gguf` (transformer, 6.78GB) and `t5-v1_1-xxl-encoder-Q8_0.gguf` (text encoder, 5.06GB), both from city96's community GGUF re-uploads, loaded via `diffusers`' and `transformers`' native GGUF support respectively.

### Model sourcing: ungated community mirror for ancillary components

`black-forest-labs/FLUX.1-schnell` — the official repo — gates its files behind a Hugging Face license click-through despite the Apache-2.0 license, which blocks unattended/scripted downloading. The GGUF-quantized transformer and T5 encoder (city96's re-uploads) are *not* gated. The small ancillary components a `FluxPipeline` also needs — VAE, CLIP text encoder, tokenizers, scheduler config, ~500MB total — were pulled from an ungated community mirror (`manbeast3b/flux.1-schnell-full1`) of the same, unmodified, Apache-2.0-licensed files.

**Caught and fixed during setup:** that mirror's `vae/` folder is actually a substituted `AutoencoderTiny` (a fast, low-fidelity preview autoencoder used by some UIs), not the real Flux `AutoencoderKL` — silently swapped in, presumably to shrink the mirror's repo size. Using it produces visibly worse image quality. The correct `AutoencoderKL` weights and config were sourced from a second ungated mirror (`YuCollection/FLUX.1-schnell-Diffusers`) and `model_index.json` was hand-corrected to declare `AutoencoderKL` instead of `AutoencoderTiny`. Anyone re-provisioning `~/ai/models/flux1-schnell/` from scratch needs to repeat this substitution — a plain re-download of the first mirror's `vae/` folder reintroduces the bug.

### Resolution: 512x512, not 768x768 — a measured hardware ceiling, not a preference

Generation runs through `diffusers`' `enable_model_cpu_offload()`, which moves one major pipeline component onto the GPU at a time (confirmed directly: `text_encoder`/`text_encoder_2`/`vae` measured on `cpu` throughout denoising, only `transformer` on `cuda:0`) — necessary because the transformer and T5 encoder don't both fit in 8GB. At 768x768, the transformer's own resident footprint during denoising (weights dequantized on the fly from GGUF, plus activations at that resolution) exceeded the card's 8.19GB entirely, forcing Windows to page GPU memory over PCIe — measured at 69-80s per denoising step (worsening across the run) and, separately, degraded the whole machine's responsiveness while it ran (not just this process — GPU memory paging isn't visible as CPU load in Task Manager, which is why it initially looked unexplained). At 512x512 the same transformer's footprint is `allocated=6.84GB, reserved=8.13GB` — measured flat and stable across all 4 steps, no churn — and generation runs at a stable ~13-14s/step with no system-wide impact.

512x512 was picked pragmatically (first size tried that fit) and leaves only tens of MB of headroom under the card's real ceiling — not a comfortable margin. Revisiting this (e.g. a size between 512 and 768, or the T5 encoder at a lower quant to trade text-fidelity for a bit more margin at higher resolution) is reasonable future tuning, not a closed decision.

### GPU handoff: stop-and-restart `llama-server` around every generation

Chosen over trying to make the two coexist in 8GB. `generate_image`'s execution: load the Flux pipeline (doesn't touch the GPU, safe to do before anything is freed) → stop `llama-server` (added `LlamaServerBackend.stop()`, which finds and terminates `llama-server.exe` by process name rather than a locally-tracked handle, since `ensure_running()`'s already-running fast path never captures one) → generate → restart `llama-server`. This makes each generation call slow (order of a minute-plus end to end) but requires no new coordination machinery beyond what already exists for starting `llama-server` on demand.

The Flux pipeline itself is cached process-wide after first load (`imagegen_pipeline.get_pipeline()`) — reloading would re-pay the GGUF-to-bf16 conversion pass (about a minute) on every single image, which is avoidable since the loaded pipeline's GPU-resident cost is zero while idle (components sit on `cpu` until a generation call moves them).

### Future direction: stable-diffusion.cpp

PyTorch's dynamic caching allocator was directly responsible for the 768x768 failure mode above — GGUF weights are dequantized on the fly per forward pass, and the allocator's behavior right at a hard VRAM ceiling is what turned "not quite enough memory" into "OS-level paging and a stalled machine" rather than a clean, predictable slowdown. `llama-server` (GGML-based, like `stable-diffusion.cpp`) does not exhibit this — it pre-plans a static VRAM budget once at load and never churns allocations during inference, which is why it stays stable at comparably high VRAM occupancy. Migrating image generation to `stable-diffusion.cpp` — same GGML lineage as `llama.cpp`, native GGUF loading, likely able to reuse the exact GGUF files already downloaded — is expected to fix this at the root rather than by capping resolution, and is a planned follow-up (see `todo.md`), not part of this decision.

## Data flow

Reuses the existing `images` / `message_image_attachments` tables (ADR-0007) rather than adding new storage — `generate_image` creates an `Image` row directly (same shape `POST /api/images` produces) and returns its id in the tool result; the frontend attaches it to the tool-result message via the same generic `image_ids` mechanism a user-uploaded image attachment already uses, so orphan GC and branch-copy semantics apply unchanged.
