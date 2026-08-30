"""Turns a continuous 16 kHz mono frame stream into committed + provisional transcript text.

Segmentation strategy (see ADR-0022):

- **webrtcvad** on each 20 ms frame tracks how long the trailing audio has been non-speech.
- **Commit on pause:** once trailing silence passes a threshold and enough speech is buffered,
  the buffered speech is transcribed whole and emitted as final ("committed") text.
- **Forced cut when long:** a pause-free monologue that reaches a hard ceiling is cut anyway,
  transcribing all but a short trailing overlap of audio that the *next* commit de-duplicates
  at the word level so no word is lost or doubled at the seam.
- **Provisional display:** while a segment is still growing, its buffer is re-transcribed every
  couple of seconds and emitted as replaceable "provisional" text, cleared when the segment
  commits.

Only one transcription runs at a time (`_busy`); frames keep buffering while it is in flight and
the segmentation check re-runs when it finishes.
"""
import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np
import webrtcvad

from stt import backend as stt_backend

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320


@dataclass
class StreamingConfig:
    """Tuning knobs for the segmentation loop. Defaults chosen for talk / video / call audio."""
    vad_aggressiveness: int = 2
    silence_commit_seconds: float = 0.6
    min_speech_seconds: float = 1.0
    force_cut_seconds: float = 24.0
    force_cut_transcribe_seconds: float = 22.0
    force_cut_overlap_seconds: float = 2.0
    pause_keep_tail_seconds: float = 0.3
    provisional_min_buffer_seconds: float = 6.0
    provisional_interval_seconds: float = 2.0
    committed_tail_words: int = 15


class StreamingTranscriber:
    """Feed it 20 ms frames via `feed()`; it calls `emit` with committed / provisional / status
    / error events. `feed()` must be called from the asyncio event loop thread."""

    def __init__(
        self,
        language: str | None,
        translate: bool,
        emit: Callable[[dict], None],
        config: StreamingConfig | None = None,
    ) -> None:
        """`language` is a Whisper code or None for auto-detect. `emit` is a synchronous sink
        (typically `asyncio.Queue.put_nowait`)."""
        self._language = language
        self._translate = translate
        self._emit = emit
        self._config = config if config is not None else StreamingConfig()
        self._vad = webrtcvad.Vad(self._config.vad_aggressiveness)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._speech_sample_count = 0
        self._trailing_silence_seconds = 0.0
        self._has_speech = False
        self._busy = False
        self._last_provisional_at = 0.0
        self._committed_tail: list[str] = []

    def feed(self, frame: np.ndarray) -> None:
        """Append one 20 ms frame, update the speech/silence state, and act if a boundary hit."""
        self._buffer = np.concatenate([self._buffer, frame])
        if self._is_speech(frame):
            self._has_speech = True
            self._speech_sample_count += len(frame)
            self._trailing_silence_seconds = 0.0
        else:
            self._trailing_silence_seconds += len(frame) / SAMPLE_RATE
        self._maybe_act()

    def _is_speech(self, frame: np.ndarray) -> bool:
        """Run webrtcvad on one frame; non-standard frame sizes are treated as silence."""
        if len(frame) != FRAME_SAMPLES:
            return False
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        try:
            return self._vad.is_speech(pcm16, SAMPLE_RATE)
        except Exception:
            return False

    def _reset_speech_state(self) -> None:
        """Clear the per-segment speech/silence counters after a segment is taken."""
        self._speech_sample_count = 0
        self._trailing_silence_seconds = 0.0
        self._has_speech = False

    def _maybe_act(self) -> None:
        """Decide whether to commit a segment, force-cut, or refresh the provisional transcript."""
        if self._busy:
            return
        config = self._config
        buffer_seconds = len(self._buffer) / SAMPLE_RATE
        speech_seconds = self._speech_sample_count / SAMPLE_RATE

        if (
            self._has_speech
            and self._trailing_silence_seconds >= config.silence_commit_seconds
            and speech_seconds >= config.min_speech_seconds
        ):
            keep = int(config.pause_keep_tail_seconds * SAMPLE_RATE)
            if len(self._buffer) > keep:
                segment = self._buffer[: len(self._buffer) - keep].copy()
                self._buffer = self._buffer[len(self._buffer) - keep :].copy()
            else:
                segment = self._buffer.copy()
                self._buffer = np.zeros(0, dtype=np.float32)
            self._reset_speech_state()
            self._start_transcription(segment, provisional=False)
            return

        if buffer_seconds >= config.force_cut_seconds:
            cut = int(config.force_cut_transcribe_seconds * SAMPLE_RATE)
            overlap = int(config.force_cut_overlap_seconds * SAMPLE_RATE)
            segment = self._buffer[:cut].copy()
            self._buffer = self._buffer[cut - overlap :].copy()
            self._reset_speech_state()
            self._has_speech = True
            self._speech_sample_count = overlap
            self._start_transcription(segment, provisional=False)
            return

        now = time.monotonic()
        if (
            buffer_seconds >= config.provisional_min_buffer_seconds
            and now - self._last_provisional_at >= config.provisional_interval_seconds
        ):
            self._last_provisional_at = now
            self._start_transcription(self._buffer.copy(), provisional=True)

    def _start_transcription(self, segment: np.ndarray, provisional: bool) -> None:
        """Mark busy and schedule the async transcription of `segment`."""
        self._busy = True
        asyncio.create_task(self._transcribe_segment(segment, provisional))

    async def _transcribe_segment(self, segment: np.ndarray, provisional: bool) -> None:
        """Transcribe one segment and emit the result, then re-check for the next boundary."""
        try:
            self._emit({"type": "status", "state": "transcribing"})
            wav_bytes = _pcm_to_wav(segment)
            text = (await stt_backend.transcribe(wav_bytes, self._language, self._translate)).strip()
            if provisional:
                if len(text) > 0:
                    self._emit({"type": "provisional", "text": text})
            else:
                new_text = self._dedupe_against_committed(text)
                if len(new_text) > 0:
                    self._emit({"type": "committed", "text": new_text})
                self._emit({"type": "provisional", "text": ""})
        except Exception:
            logger.exception("segment transcription failed")
            self._emit({"type": "error", "message": "transcription failed"})
        finally:
            self._busy = False
            self._emit({"type": "status", "state": "listening"})
            self._maybe_act()

    def _dedupe_against_committed(self, text: str) -> str:
        """Drop any leading words of `text` that repeat the tail of already-committed text
        (the forced-cut overlap, or a word Whisper echoed across a pause)."""
        words = text.split()
        if len(words) == 0:
            return ""
        tail = self._committed_tail
        max_overlap = min(len(tail), len(words))
        overlap = 0
        for candidate in range(max_overlap, 0, -1):
            if _normalize(tail[-candidate:]) == _normalize(words[:candidate]):
                overlap = candidate
                break
        remaining = words[overlap:]
        self._committed_tail = (self._committed_tail + remaining)[-self._config.committed_tail_words :]
        return " ".join(remaining)


def _normalize(words: list[str]) -> list[str]:
    """Lowercase and strip trailing punctuation so overlap matching survives Whisper rephrasing."""
    return [word.lower().strip(".,!?;:") for word in words]


def _pcm_to_wav(samples: np.ndarray) -> bytes:
    """Wrap float32 [-1, 1] mono samples in a 16-bit PCM WAV container at 16 kHz."""
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()
