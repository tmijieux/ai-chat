"""Common interface for STT backends, mirroring llm/base.py's LLMBackend pattern."""
from abc import ABC, abstractmethod


class SttBackend(ABC):

    @abstractmethod
    async def ensure_running(self) -> None:
        """Load the model / start whatever process serves it. Called once at app startup."""
        ...

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str | None, translate: bool = False) -> str:
        """Transcribe raw audio bytes (any format ffmpeg can decode) to text.

        When `translate` is True the output is translated to English (Whisper only supports
        translation into English, regardless of the source language)."""
        ...
