# ADR-0022: System Audio Transcription

**Date:** 2026-08-30
**Status:** Accepted

## Context

Voice Dictation transcribes the microphone. A separate need: transcribe audio *coming out of the
machine* — a foreign-language video, a call, a stream playing in any application — for live
subtitling or for recording/analysis. The languages wanted go well beyond the French/English of
dictation (Spanish, Chinese, Japanese, Russian, Arabic…), sometimes translated to a language the
user reads.

## Decision

### Capture on the backend via WASAPI loopback, not in the browser

`navigator.mediaDevices.getDisplayMedia({ audio: true })` is the browser-native route, but the
user's browser is Firefox, which has never supported audio capture in `getDisplayMedia` (video
only, every platform). Chromium's tab/screen audio capture would have worked but is not portable
and forces a share-picker prompt on every activation.

The backend runs on the same machine as the browser, so it can open the default (or a chosen)
render endpoint in WASAPI loopback mode and capture everything played on it, from any app, with no
prompt and a free choice of source device. This does couple the feature to the app being local —
already true everywhere else in this project.

### `soundcard`, not `pyaudiowpatch`

`soundcard` is pure-ctypes (no compiled extension, clean `uv` install on Windows), enumerates
render endpoints and microphones, exposes loopback capture directly, and resamples to 16 kHz mono
in the recorder so no ffmpeg step is needed on this path. `pyaudiowpatch` (a PyAudio fork with
WASAPI loopback) was the alternative; it ships wheels but adds a C extension and the PyAudio API
surface for no gain here.

### Per-device selection only; no per-application capture

Capturing a single application's audio needs the Windows process-loopback activation API
(`ActivateAudioInterfaceAsync` with `VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`), which means
hand-rolled `ctypes` against COM interfaces `soundcard` doesn't expose. Not worth it for the first
version — the picker offers each output device and each microphone.

### Segmentation: VAD-anchored, with a forced cut + word-level stitch

Whisper's encoder window is fixed at 30 s, so continuous audio must be segmented. The approach:

- `webrtcvad` on each 20 ms frame tracks trailing-silence duration (chosen over an energy gate —
  system audio carries music and effects an energy gate would read as "not silence").
- A pause past a threshold commits the buffered speech as final text.
- A pause-free stretch that hits a hard ceiling (~24 s) is force-cut anyway, transcribing all but
  a ~2 s audio overlap; the next commit de-duplicates the seam with a token-level
  longest-common-run match (Whisper rephrases across runs, so the match is case- and
  punctuation-insensitive rather than literal).
- While a segment is still growing, its buffer is re-transcribed every ~2 s and shown as
  replaceable "provisional" text.

A pure sliding-window + LocalAgreement scheme (à la whisper_streaming) was the alternative — lower
latency to first text, but substantially more state and tuning, and it needs word timestamps to
trim its buffer cleanly. `<|notimestamps|>` stays on; timestamp-based stitching is a possible
future refinement.

`webrtcvad-wheels` (a maintained fork) is used rather than `webrtcvad`, whose 2.0.10 release
imports `pkg_resources` and fails without `setuptools` in the environment.

### Translation is English-only

Whisper's `translate` task only ever targets English (X→en). The "Translate to English" toggle
reflects that exactly; there is no target-language choice.

### Coupled to the whisper.cpp STT server

The `translate` flag is threaded through `SttBackend.transcribe` and implemented for both
backends, but the feature was validated against the whisper.cpp server (the active backend per
ADR-0014). The OpenVINO backend remains only a rollback path.

## Consequences

- New backend package `backend/system_audio/` (capture + streaming transcriber) and a
  `/api/system-audio/ws` endpoint; new frontend service + right-side drawer + a chat-input button.
- New dependencies: `soundcard`, `webrtcvad-wheels`.
- Transcription quality on arbitrary system audio (music, far-mic speakers, overlapping talkers)
  is inherently lower than close-mic dictation, and this path deliberately skips the LLM
  STT-correction pass, which is tuned for dictated coding commands.
