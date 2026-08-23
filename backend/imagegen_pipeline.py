"""Flux.1-schnell image generation via stable-diffusion.cpp's CLI binary (GGML-based, same
lineage as llama.cpp — see llm/llama_server.py for the equivalent LLM-side pattern).

Replaced an earlier PyTorch/diffusers pipeline: that pipeline's dynamic caching allocator was
the root cause of a hard VRAM-ceiling failure mode on this 8GB card — right at capacity, it
would make Windows page GPU memory and stall the whole system, not just this process. GGML's
static allocation plan has real headroom at the same 512x512 resolution (~6.8GB peak, measured)
and generates in ~10s instead of ~60s. See ADR-0013.

Model files live under ~/ai/models/flux1-schnell/ — unchanged from the PyTorch setup, since
sd-cli.exe loads the same GGUF/safetensors files directly."""
import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)

SD_CLI_EXE = os.path.expanduser("~/ai/stable-diffusion.cpp/build/bin/sd-cli.exe")
MODEL_DIR = os.path.expanduser("~/ai/models/flux1-schnell")
DIFFUSION_MODEL = os.path.join(MODEL_DIR, "flux1-schnell-Q4_K_S.gguf")
T5XXL = os.path.join(MODEL_DIR, "t5-v1_1-xxl-encoder-Q8_0.gguf")
CLIP_L = os.path.join(MODEL_DIR, "text_encoder", "model.safetensors")
VAE = os.path.join(MODEL_DIR, "vae", "diffusion_pytorch_model.safetensors")


@dataclass
class GeneratedImage:
    """One generated image, ready to persist: raw PNG bytes plus its pixel dimensions."""
    png_bytes: bytes
    width: int
    height: int


def _run_sd_cli(prompt: str, width: int, height: int, output_path: str) -> None:
    """Blocking subprocess call — always invoked through asyncio.to_thread, never directly from
    an async function, so a ~10s generation doesn't stall the event loop (and every other
    concurrent connection the backend is serving) for its duration."""
    args = [
        SD_CLI_EXE,
        "--diffusion-model", DIFFUSION_MODEL,
        "--t5xxl", T5XXL,
        "--clip_l", CLIP_L,
        "--vae", VAE,
        "--prompt", prompt,
        "--cfg-scale", "1.0",  # Flux.1-schnell is distilled and has no CFG — 1.0 means "off"
        "--sampling-method", "euler",
        "--steps", "4",
        "--width", str(width),
        "--height", str(height),
        # Without this, sd-cli tries to keep the transformer, T5 encoder, CLIP, and VAE all
        # resident on the GPU at once (~12GB combined) — overcommits this 8GB card and stalls
        # the whole system. Confirmed directly (not theoretical): the first attempt at wiring
        # this in, without the flag, froze the machine the same way the PyTorch pipeline did.
        "--offload-to-cpu",
        "-o", output_path,
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"sd-cli.exe exited {result.returncode}: {result.stderr[-2000:]}")


async def generate(prompt: str, width: int = 512, height: int = 512) -> GeneratedImage:
    """Generate one image from a text prompt."""
    fd, output_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        await asyncio.to_thread(_run_sd_cli, prompt, width, height, output_path)
        with Image.open(output_path) as img:
            actual_width, actual_height = img.width, img.height
        with open(output_path, "rb") as f:
            png_bytes = f.read()
    finally:
        os.unlink(output_path)

    return GeneratedImage(png_bytes=png_bytes, width=actual_width, height=actual_height)
