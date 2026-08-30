"""System-audio capture via the `soundcard` library.

An output device is captured through WASAPI loopback (recording what is being played); a
microphone is captured directly. Recording runs on a dedicated worker thread because
`soundcard`'s recorder is a blocking context manager — 20 ms mono frames at 16 kHz are handed
back through the `on_frame` callback, which the websocket handler wires to
`loop.call_soon_threadsafe` so the asyncio side stays single-threaded.
"""
import logging
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundcard

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20 ms at 16 kHz — the frame size webrtcvad requires


@dataclass
class AudioSource:
    """One selectable capture source shown in the picker."""
    id: str
    label: str
    kind: str  # "output" = system loopback, "input" = microphone


def list_sources() -> list[AudioSource]:
    """Enumerate system output devices (captured via loopback) followed by real microphones."""
    sources: list[AudioSource] = []
    try:
        for speaker in soundcard.all_speakers():
            sources.append(AudioSource(id=str(speaker.id), label=f"System output: {speaker.name}", kind="output"))
    except Exception:
        logger.exception("failed to enumerate output devices")
    try:
        for microphone in soundcard.all_microphones(include_loopback=False):
            sources.append(AudioSource(id=str(microphone.id), label=f"Microphone: {microphone.name}", kind="input"))
    except Exception:
        logger.exception("failed to enumerate microphones")
    return sources


class AudioCapture:
    """Background-thread capture of one source, delivering float32 mono frames of FRAME_SAMPLES
    at 16 kHz to `on_frame` until `stop()`."""

    def __init__(self, device_id: str, kind: str, on_frame: Callable[[np.ndarray], None]) -> None:
        """Prepare (but do not start) capture of `device_id`. `kind` is "output" or "input"."""
        self._device_id = device_id
        self._kind = kind
        self._on_frame = on_frame
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="system-audio-capture", daemon=True)
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        """Set to a human-readable message if the capture thread failed to start recording."""
        return self._error

    def start(self) -> None:
        """Start the capture thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and wait briefly for it to unwind."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        """Thread body: open the recorder and pump fixed-size frames until stopped."""
        try:
            include_loopback = self._kind == "output"
            microphone = soundcard.get_microphone(self._device_id, include_loopback=include_loopback)
            with microphone.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SAMPLES) as recorder:
                while not self._stop_event.is_set():
                    block = recorder.record(numframes=FRAME_SAMPLES)
                    frame = block[:, 0] if block.ndim > 1 else block
                    self._on_frame(np.ascontiguousarray(frame, dtype=np.float32))
        except Exception as exc:
            self._error = str(exc)
            logger.exception("system-audio capture thread crashed")
