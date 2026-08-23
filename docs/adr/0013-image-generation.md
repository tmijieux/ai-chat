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

### Inference engine: stable-diffusion.cpp, not PyTorch/diffusers — a root-cause fix, not a starting choice

The first implementation ran through `diffusers`' `enable_model_cpu_offload()`, which moves one major pipeline component onto the GPU at a time — necessary because the transformer and T5 encoder don't both fit in 8GB. That worked at 512x512, but at 768x768 the transformer's own resident footprint during denoising (weights dequantized on the fly from GGUF, plus activations at that resolution) exceeded the card's 8.19GB entirely, forcing Windows to page GPU memory over PCIe — measured at 69-80s per denoising step (worsening across the run) and, separately, stalled the whole machine's responsiveness while it ran (not just this process — GPU memory paging isn't visible as CPU load in Task Manager, which is why it initially looked unexplained). Even at 512x512, PyTorch's own accounting showed `allocated=6.84GB, reserved=8.13GB` — stable, but only tens of MB of headroom under the card's real ceiling.

Diagnosis (confirmed directly, not theoretical): PyTorch's dynamic caching allocator is the root cause, not the resolution. GGUF weights are dequantized on the fly per forward pass — every one of the transformer's ~219 tensors gets a fresh scratch buffer allocated on the fly, every forward pass. Right at a hard VRAM ceiling, that turns "not quite enough memory" into "OS-level paging and a stalled machine" rather than a clean, predictable slowdown. `llama-server` (GGML-based) never exhibits this because it pre-plans a static VRAM budget once at load and never churns allocations during inference — the same reason it stays stable at comparably high VRAM occupancy while the PyTorch image pipeline didn't.

Migrated image generation to **stable-diffusion.cpp** (same GGML lineage as `llama.cpp`, cloned and built at `~/ai/stable-diffusion.cpp` — see `scripts/build_stable_diffusion_cpp.bat`) to fix this at the root. It loads the exact same GGUF/safetensors files already downloaded for the PyTorch pipeline, no reconversion needed. Measured result at the same 512x512: **peak VRAM ~6.8GB with real headroom** (params + compute buffer staged per component, not PyTorch's near-total-card reservation) and **generation in ~11s total**, versus ~60s+ with PyTorch — a 5-6x speedup with no VRAM-ceiling fragility. Confirmed directly: a first attempt without `--offload-to-cpu` (which keeps the transformer, T5, CLIP, and VAE all resident on GPU at once, ~12GB combined) reproduced the exact same system-wide stall as the PyTorch failure mode — `--offload-to-cpu` is required, not optional, and is set unconditionally in `imagegen_pipeline.py`.

**Toolchain note:** building stable-diffusion.cpp with CUDA on this machine required pinning the VS2022 (MSVC 14.44) toolset explicitly via `vcvars64.bat -vcvars_ver=14.44` — CUDA 13.0's `nvcc` crashes (`cudafe++` access violation) against the newer VS2026 toolset even with `-allow-unsupported-compiler`. Also needed `/bigobj` for one heavily-templated translation unit. Both are documented as comments in the build script.

512x512 remains the resolution in use, not because it's still a hard ceiling (it measurably isn't anymore — sd.cpp had real headroom at that size) but because it hasn't been pushed further yet. See `todo.md`.

### GPU handoff: stop-and-restart `llama-server` around every generation

Chosen over trying to make the two coexist in 8GB — a typical loaded chat model (5-7GB) plus sd.cpp's own ~6.8GB peak still exceeds 8GB combined, so this remains necessary regardless of inference engine. `generate_image`'s execution: stop `llama-server` (`LlamaServerBackend.stop()`, which finds and terminates `llama-server.exe` by process name rather than a locally-tracked handle, since `ensure_running()`'s already-running fast path never captures one) → generate (a plain `sd-cli.exe` subprocess call, run off the event loop via `asyncio.to_thread` so a ~11s generation doesn't stall every other concurrent connection the backend is serving) → restart `llama-server`. No pipeline object to keep resident between calls — sd.cpp's own model load is cheap enough (~1-2s for the transformer's tensors) that there's nothing worth caching, unlike the PyTorch pipeline's minute-plus GGUF-to-bf16 conversion pass.

## Data flow

Reuses the existing `images` / `message_image_attachments` tables (ADR-0007) rather than adding new storage — `generate_image` creates an `Image` row directly (same shape `POST /api/images` produces) and returns its id in the tool result; the frontend attaches it to the tool-result message via the same generic `image_ids` mechanism a user-uploaded image attachment already uses, so orphan GC and branch-copy semantics apply unchanged.
