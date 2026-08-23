"""Flux.1-schnell image generation, loaded on demand. Mirrors whisper_pipeline.py's shape: a
module-level load function returning a pipeline object, and a generate function that takes it.

Runs GGUF-quantized (transformer Q4_K_S, T5 text encoder Q8_0) with sequential CPU offload —
the RTX 4070 Laptop's 8GB VRAM can't hold the transformer and T5 encoder at once, so diffusers
moves each component to the GPU only while it's actually running and parks the rest in system RAM.

Model files live under ~/ai/models/flux1-schnell/. The transformer and T5 encoder GGUF files come
from city96's ungated quantized re-uploads; the small ancillary components (VAE, CLIP, tokenizers,
scheduler config) come from an ungated community mirror of the diffusers-format repo, since the
official black-forest-labs/FLUX.1-schnell repo gates those files behind a license click-through
despite the Apache-2.0 license — see ADR for the reasoning."""
import io
import logging
import os
from dataclasses import dataclass

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig  # pyright: ignore[reportPrivateImportUsage] — diffusers re-exports these via a lazy module pyright's stubs don't follow
from transformers import T5EncoderModel

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.expanduser("~/ai/models/flux1-schnell")
TRANSFORMER_GGUF = os.path.join(MODEL_DIR, "flux1-schnell-Q4_K_S.gguf")
T5_GGUF = "t5-v1_1-xxl-encoder-Q8_0.gguf"

_pipe: FluxPipeline | None = None


@dataclass
class GeneratedImage:
    """One generated image, ready to persist: raw PNG bytes plus its pixel dimensions."""
    png_bytes: bytes
    width: int
    height: int


def get_pipeline() -> FluxPipeline:
    """Return the process-wide Flux pipeline, loading it from disk on first call and caching it
    for the rest of the backend's lifetime — reloading would re-pay the GGUF dequantization pass
    (a minute or more) on every single image. Resident cost is ~12GB of system RAM while idle
    (weights stay in their GGUF-quantized form; enable_model_cpu_offload only moves a component
    to the GPU for the duration of its own forward pass). Loading itself doesn't touch the GPU,
    so it's safe to call before llama-server is stopped — only generate() needs the GPU free."""
    global _pipe
    if _pipe is None:
        _pipe = _load_pipeline()
    return _pipe


def _load_pipeline() -> FluxPipeline:
    transformer = FluxTransformer2DModel.from_single_file(
        TRANSFORMER_GGUF,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=MODEL_DIR,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_DIR,
        gguf_file=T5_GGUF,
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxPipeline.from_pretrained(
        MODEL_DIR,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def generate(pipe: FluxPipeline, prompt: str, width: int = 512, height: int = 512) -> GeneratedImage:
    """Generate one image from a text prompt. Flux.1-schnell is distilled for 1-4 steps and must
    be run with guidance_scale=0.0 (it has no classifier-free guidance, unlike Flux.1-dev).

    512x512 is the default, not 768x768, because of a measured hardware limit: at 768x768 the
    transformer's resident footprint during denoising left only ~280MB of headroom on this 8GB
    card, which caused Windows to page GPU memory and stall the whole system (not just this
    process) for tens of seconds per step. 512x512 leaves real headroom and runs 5-6x faster
    per step. See ADR-0013."""
    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=height,
        width=width,
    ).images[0]  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue] — return_dict defaults True so this is always a FluxPipelineOutput, not the tuple branch of the stubbed union

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return GeneratedImage(png_bytes=buf.getvalue(), width=width, height=height)
