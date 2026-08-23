"""OpenVINO STT backend — thin async adapter around whisper_pipeline.py, kept unmodified as the
validated reference implementation. See stt/__init__.py to switch back to this from
WhisperCppBackend if the whisper.cpp backend has problems in real use."""
import asyncio
import logging

import whisper_pipeline
from .base import SttBackend

logger = logging.getLogger(__name__)


class OpenVinoBackend(SttBackend):

    def __init__(self) -> None:
        self._pipeline: whisper_pipeline.WhisperPipeline | None = None

    async def ensure_running(self) -> None:
        loop = asyncio.get_event_loop()
        self._pipeline = await loop.run_in_executor(None, whisper_pipeline.load_pipeline)

    async def transcribe(self, audio_bytes: bytes, language: str | None) -> str:
        if self._pipeline is None:
            raise RuntimeError("OpenVINO Whisper pipeline is not loaded")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, whisper_pipeline.transcribe, self._pipeline, audio_bytes, language
        )
