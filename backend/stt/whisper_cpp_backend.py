"""whisper.cpp-based STT backend: a resident whisper-server.exe (Vulkan, Arc iGPU only) hit over
HTTP for each transcription. Mirrors llm/llama_server.py's subprocess lifecycle pattern.

GGML_VK_VISIBLE_DEVICES=0 is required, not optional: ggml-vulkan defaults to using every
dedicated *and* integrated GPU it finds, which would otherwise also pull in the RTX 4070 and
reintroduce the VRAM contention this backend exists to avoid (llama-server already uses the RTX
for the chat model). Device 0 is the Intel Arc iGPU on this machine — confirmed via
`whisper-cli.exe`'s device enumeration, not assumed.

See scripts/build_whisper_cpp.bat for the build, and todo.md's "Voice Dictation" section for the
remaining open question (preserving streaming-partial UX) this backend was validated against.

Audio is transcoded to WAV in Python before it ever reaches whisper-server, not via the server's
own --convert flag. That flag shells out to ffmpeg through std::system() on every request and
was found, in real use, to intermittently misdetect the Opus codec inside the browser's Ogg/WebM
blobs ("Codec not found" — ffmpeg's own format-probing flakiness, not a request-timing race:
whisper-server serializes /inference behind a single mutex, so requests are never concurrent).
Converting here instead reuses whisper_pipeline.py's already-proven approach: an in-memory ffmpeg
pipe (stdin/stdout, no temp files, no shell layer) — the same mechanism that never showed this
failure mode for the OpenVINO backend.
"""
import asyncio
import logging
import os
import subprocess
from pathlib import Path

import aiohttp

from .base import SttBackend

WHISPER_BASE_URL = "http://127.0.0.1:8090"
WHISPER_HEALTH_URL = f"{WHISPER_BASE_URL}/health"
WHISPER_INFERENCE_URL = f"{WHISPER_BASE_URL}/inference"
WHISPER_SERVER_EXE = str(Path.home() / "ai/whisper.cpp/build/bin/whisper-server.exe")
WHISPER_MODEL_PATH = str(Path.home() / "ai/models/whisper-cpp/ggml-small.bin")
WHISPER_LOG_PATH = str(Path.home() / "ai/whisper.cpp/whisper-server.log")
SAMPLE_RATE = 16_000

logger = logging.getLogger(__name__)


def _transcode_to_wav(audio_bytes: bytes) -> bytes:
    """Blocking ffmpeg call — always invoked through asyncio.to_thread, never directly."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "wav", "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", "pipe:1"],
        input=audio_bytes, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode to WAV failed: {result.stderr.decode(errors='replace')}")
    return result.stdout


class WhisperCppBackend(SttBackend):
    """Launches and talks to a resident whisper-server.exe. See stt/__init__.py to switch back
    to OpenVinoBackend if this one turns out to have problems in real use."""

    async def ensure_running(self) -> None:
        async with aiohttp.ClientSession() as http:
            try:
                async with http.get(WHISPER_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        logger.info("whisper-server already running.")
                        return
            except Exception:
                pass

        logger.info("whisper-server not detected — launching ...")
        # stdout/stderr go to a log file, not an unread PIPE: whisper-server's startup output
        # (Vulkan device enumeration, model load logging) can fill an OS pipe buffer before it
        # starts listening, blocking the child on write() since nothing drains the pipe while
        # this polls /health — reproduced directly (a PIPE run stalled 30s+, a file-redirected
        # run was healthy in ~1.7s with identical args).
        log_file = open(WHISPER_LOG_PATH, "wb")
        p = subprocess.Popen(
            [
                WHISPER_SERVER_EXE,
                "-m", WHISPER_MODEL_PATH,
                "-bs", "5", "-bo", "5",  # beam search — greedy (1/1) traded too much accuracy for speed
                # No --convert: we transcode to WAV ourselves before uploading (see module
                # docstring) — whisper.cpp's own reader (miniaudio) accepts WAV natively.
                "--port", "8090",
                "--host", "127.0.0.1",
            ],
            env={**os.environ, "GGML_VK_VISIBLE_DEVICES": "0"},
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()  # child keeps its own duplicated handle; safe to close ours

        async with aiohttp.ClientSession() as http:
            for _ in range(60):  # 30s — model load is much faster than the LLM's
                ret = p.poll()
                if ret is not None:
                    output = Path(WHISPER_LOG_PATH).read_text(errors="replace")
                    logger.error("whisper-server exited with code %s:\n%s", ret, output)
                    return
                await asyncio.sleep(0.5)
                try:
                    async with http.get(WHISPER_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=1)) as r:
                        if r.status == 200:
                            logger.info("whisper-server started successfully.")
                            return
                except Exception:
                    pass

        logger.warning("whisper-server did not respond within 30s — continuing anyway.")

    async def transcribe(self, audio_bytes: bytes, language: str | None, translate: bool = False) -> str:
        wav_bytes = await asyncio.to_thread(_transcode_to_wav, audio_bytes)

        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("language", language if language is not None else "auto")
        if translate == True:
            # whisper.cpp can only translate *into* English (X -> en); there is no other target.
            form.add_field("translate", "true")
        form.add_field("response_format", "json")

        async with aiohttp.ClientSession() as http:
            async with http.post(WHISPER_INFERENCE_URL, data=form) as r:
                if r.status != 200:
                    body = await r.text()
                    raise RuntimeError(f"whisper-server /inference failed ({r.status}): {body}")
                data = await r.json()
                # whisper-server's output_str() unconditionally joins Whisper's per-segment
                # output with "\n" (server.cpp, not configurable via a request parameter) —
                # segments break on natural speech pauses, so continuous dictation otherwise
                # comes back with a hard newline at every pause. The OpenVINO pipeline never had
                # this (its own decode loop produces one continuous string), so collapse it here
                # to match: join with spaces instead of newlines.
                return " ".join(data["text"].split())
