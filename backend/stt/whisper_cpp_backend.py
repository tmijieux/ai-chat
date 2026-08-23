"""whisper.cpp-based STT backend: a resident whisper-server.exe (Vulkan, Arc iGPU only) hit over
HTTP for each transcription. Mirrors llm/llama_server.py's subprocess lifecycle pattern.

GGML_VK_VISIBLE_DEVICES=0 is required, not optional: ggml-vulkan defaults to using every
dedicated *and* integrated GPU it finds, which would otherwise also pull in the RTX 4070 and
reintroduce the VRAM contention this backend exists to avoid (llama-server already uses the RTX
for the chat model). Device 0 is the Intel Arc iGPU on this machine — confirmed via
`whisper-cli.exe`'s device enumeration, not assumed.

See scripts/build_whisper_cpp.bat for the build, and todo.md's "Voice Dictation" section for the
remaining open question (preserving streaming-partial UX) this backend was validated against.
"""
import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import aiohttp

from .base import SttBackend

WHISPER_BASE_URL = "http://127.0.0.1:8090"
WHISPER_HEALTH_URL = f"{WHISPER_BASE_URL}/health"
WHISPER_INFERENCE_URL = f"{WHISPER_BASE_URL}/inference"
WHISPER_SERVER_EXE = str(Path.home() / "ai/whisper.cpp/build/bin/whisper-server.exe")
WHISPER_MODEL_PATH = str(Path.home() / "ai/models/whisper-cpp/ggml-small.bin")
WHISPER_LOG_PATH = str(Path.home() / "ai/whisper.cpp/whisper-server.log")

logger = logging.getLogger(__name__)


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
                "-bs", "1", "-bo", "1",  # greedy decode — matches the OpenVINO pipeline's decoding
                "--convert",  # ffmpeg WebM/OGG -> WAV, needed for MediaRecorder's audio/webm blobs
                "--tmp-dir", tempfile.gettempdir(),  # else defaults to "." — whatever the app's CWD is
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

    async def transcribe(self, audio_bytes: bytes, language: str | None) -> str:
        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename="audio.webm", content_type="application/octet-stream")
        form.add_field("language", language if language is not None else "auto")
        form.add_field("response_format", "json")

        async with aiohttp.ClientSession() as http:
            async with http.post(WHISPER_INFERENCE_URL, data=form) as r:
                if r.status != 200:
                    body = await r.text()
                    raise RuntimeError(f"whisper-server /inference failed ({r.status}): {body}")
                data = await r.json()
                return data["text"].strip()
