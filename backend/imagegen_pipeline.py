"""Z-Image-Turbo image generation via stable-diffusion.cpp's CLI binary (GGML-based, same
lineage as llama.cpp — see llm/llama_server.py for the equivalent LLM-side pattern).

Z-Image-Turbo (6B params, Apache-2.0) replaced an earlier Flux.1-schnell (12B) setup on the same
sd.cpp backend — smaller model, noticeably faster (~9s vs ~11s at 512x512) and more VRAM headroom
(~5.3GB peak vs ~6.8GB) on this 8GB card, comparable output quality. Both were themselves a
migration off an earlier PyTorch/diffusers pipeline, whose dynamic caching allocator was the root
cause of a hard VRAM-ceiling failure mode — right at capacity it would make Windows page GPU
memory and stall the whole system, not just this process. GGML's static allocation plan doesn't
have that failure mode. See ADR-0013.

Model files live under ~/ai/models/z-image-turbo/ (diffusion model + Qwen3 text encoder) and
~/ai/models/flux1-schnell/vae/ (the VAE is reused as-is from the earlier Flux setup — same
architecture, no need to duplicate it)."""
import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)

SD_CLI_EXE = os.path.expanduser("~/ai/stable-diffusion.cpp/build/bin/sd-cli.exe")
MODEL_DIR = os.path.expanduser("~/ai/models/z-image-turbo")
DIFFUSION_MODEL = os.path.join(MODEL_DIR, "z-image-turbo-Q4_K_M.gguf")
LLM_TEXT_ENCODER = os.path.join(MODEL_DIR, "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
VAE = os.path.expanduser("~/ai/models/flux1-schnell/vae/diffusion_pytorch_model.safetensors")


@dataclass
class GeneratedImage:
    """One generated image, ready to persist: raw PNG bytes plus its pixel dimensions."""
    png_bytes: bytes
    width: int
    height: int


def _run_sd_cli(prompt: str, width: int, height: int, output_path: str) -> None:
    """Blocking subprocess call — always invoked through asyncio.to_thread, never directly from
    an async function, so a ~9s generation doesn't stall the event loop (and every other
    concurrent connection the backend is serving) for its duration."""
    args = [
        SD_CLI_EXE,
        "--diffusion-model", DIFFUSION_MODEL,
        "--llm", LLM_TEXT_ENCODER,
        "--vae", VAE,
        "--prompt", prompt,
        "--cfg-scale", "1.0",  # Z-Image-Turbo is distilled and has no CFG — 1.0 means "off"
        "--diffusion-fa",
        "--steps", "8",
        "--width", str(width),
        "--height", str(height),
        # Without this, sd-cli tries to keep the diffusion model and text encoder both resident
        # on the GPU at once — on the earlier, larger Flux setup that overcommitted this 8GB
        # card and stalled the whole system (confirmed directly, not theoretical). Z-Image-Turbo
        # has enough headroom that it's probably not strictly required here, but there's no
        # reason to find out the hard way a second time.
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
