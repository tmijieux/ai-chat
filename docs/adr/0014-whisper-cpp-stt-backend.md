# ADR-0014: whisper.cpp STT Backend (Vulkan, Arc iGPU)

**Date:** 2026-08-23
**Status:** Accepted

## Context

Voice dictation ran on an OpenVINO pipeline (encoder + decoder on the Intel Arc iGPU). Following the same pattern that already paid off for chat inference (`llama.cpp`) and image generation (ADR-0013's move to `stable-diffusion.cpp`) — both GGML-based tools proved dramatically more stable and efficient than the equivalent PyTorch-family pipelines on this machine — the same logic was tried for Whisper: replace the OpenVINO pipeline with `whisper.cpp`.

## Decision

### Backend: Vulkan, not `WHISPER_OPENVINO` or CUDA

`whisper.cpp`'s own `WHISPER_OPENVINO` flag only accelerates the encoder; the decoder always falls back to CPU regardless. `GGML_VULKAN=ON` accelerates the whole ggml graph (encoder + decoder) instead, across any Vulkan-capable GPU, without needing Intel's heavier oneAPI/SYCL toolkit. Built via Vulkan SDK 1.4.357.0 + the same VS2022 (MSVC 14.44) toolset already used for `stable-diffusion.cpp` — see `scripts/build_whisper_cpp.bat`.

**Critical, easy-to-miss gotcha, confirmed directly:** ggml-vulkan's device enumeration defaults to using *every* dedicated and integrated GPU it finds — on this machine, that means both the Arc iGPU and the RTX 4070, silently reintroducing the exact VRAM contention with the chat LLM this migration exists to avoid. `GGML_VK_VISIBLE_DEVICES=0` (device 0 = Arc, confirmed via `whisper-cli.exe`'s own device listing, not assumed from ordering) is required on every invocation to restrict it to the iGPU — set unconditionally when the server subprocess is spawned.

### Serving shape: a resident `whisper-server.exe` over HTTP, not a subprocess-per-call or a C++ binding

Three options existed: (a) spawn a fresh CLI process per transcription, like `imagegen_pipeline.py` does for `sd-cli.exe`; (b) hit a resident `whisper-server.exe` over HTTP, keeping the model loaded between calls; (c) bind `whisper.cpp`'s C API directly into the Python process (ctypes/pybind), skipping the network hop and per-call ffmpeg-in-server overhead entirely.

(a) was ruled out early: model load alone measured ~500ms, comparable to or larger than the transcription compute itself for short clips, and the existing streaming-partials UX (see `CONTEXT.md`) fires many transcription calls per dictation, not one.

(b) was chosen first, deliberately the cheaper option, with (c) explicitly deferred as the fallback if HTTP overhead turns out to be a real problem — not evaluated further here since (b) already measured comfortably fast enough (see Validation below).

### Feasibility question this had to answer: is a synchronous re-transcribe-the-growing-buffer approach fast enough?

The existing frontend (`VoiceDictationService`) already implements streaming partials by resubmitting the *entire accumulated audio blob* to a synchronous transcription endpoint roughly every 600ms, self-paced (the next call only fires once the previous one resolves — no concurrent-request pileup risk regardless of per-call latency). This is not a new design question whisper.cpp introduces; whisper.cpp's inference API is exactly this same shape (no streaming/incremental-input endpoint of its own — confirmed by reading `examples/server/server.cpp` and `examples/cli/cli.cpp` directly: both take one complete buffer in, one result out).

What had to be confirmed empirically was whether a *whisper.cpp-backed* version of that same resubmit-the-growing-buffer pattern stays fast enough as the buffer grows toward Whisper's fixed 30-second encoder window. Measured directly against a resident `whisper-server` (Arc iGPU only, greedy decoding): encode cost is flat (~470-550ms) regardless of buffer length, since the encoder always processes a fixed-size padded 30s window; decode cost grows with actual speech content (~9-10ms/token), reaching ~640ms compute at a full 30s buffer. Total per-call latency through the server (including its own ffmpeg conversion and HTTP overhead) stayed under ~1.9s even at the 30s ceiling — comfortably within the existing self-paced cadence. Quality and output text were identical to the OpenVINO pipeline on the same test clip.

### Decode settings: greedy, matching the OpenVINO pipeline

`whisper-server` is started with `-bs 1 -bo 1` (beam size 1, best-of 1) rather than its own defaults (beam size 2, best-of 2) — the OpenVINO pipeline only ever did greedy (argmax) decoding, and matching it keeps the quality/speed comparison and real-world behavior equivalent rather than introducing an unrelated variable.

### Two subprocess-management bugs found and fixed during validation, not copied from the `llama_server.py` precedent

1. **Pipe deadlock risk.** The natural pattern to copy from `llm/llama_server.py`'s `ensure_running()` is `stdout=subprocess.PIPE, stderr=subprocess.STDOUT` with nothing draining the pipe while polling `/health`. Reproduced directly: a `Popen`-launched server with an unread `PIPE` took 30s+ (health polling gave up entirely) to become healthy; the identical binary and arguments, output redirected to a file instead, was healthy in ~1.7s. whisper-server's startup log volume is apparently enough to fill an unread OS pipe buffer and block the child on `write()` before it starts listening. Fixed by redirecting to a log file (`~/ai/whisper.cpp/whisper-server.log`) instead of an in-memory pipe. `llama_server.py` has the same latent pattern but wasn't touched here — out of scope, and not confirmed to actually manifest there.
2. **Temp-file location.** `whisper-server`'s `--tmp-dir` (for ffmpeg-transcoded uploads) defaults to `.` — the process's working directory, which without an explicit flag would be wherever the Python backend was launched from (e.g. the repo's `backend/` directory), not a scratch location. Fixed by passing `--tmp-dir` pointed at the OS temp directory explicitly.

### Kept side by side with OpenVINO, not a full replacement

Unlike the image-gen PyTorch→sd.cpp migration (ADR-0013), which fully removed the old pipeline, both STT backends are kept: `backend/stt/openvino_backend.py` (unmodified `whisper_pipeline.py` wrapped in the same async interface) and `backend/stt/whisper_cpp_backend.py`, switchable at a single line in `backend/stt/__init__.py` (mirrors `llm/__init__.py`'s existing backend-swap pattern). Reasoning: dictation is primarily used in French (the STT-correction prompt's default), and validation so far only covered English audio — keeping a fast, no-code-change rollback path was judged worth the maintenance cost of two backends until French quality is confirmed in real use.

### Correction after initial deployment: intermittent FFmpeg failures traced to Firefox's Ogg default, not the server-side `--convert` path

The initial version used `whisper-server`'s built-in `--convert` flag, which writes the upload to a temp file and shells out to ffmpeg via `std::system()`. In real use (real French dictation, not synthetic test clips) this intermittently failed with `{"error":"FFmpeg conversion failed."}`. The server's own log showed the actual ffmpeg error: `[ogg @ ...] Codec not found` / `failed to create or replace stream`.

First fix attempt — moving audio conversion into Python (reusing `whisper_pipeline.py`'s in-memory ffmpeg pipe, no temp files, no `cmd.exe` shell layer) and dropping `--convert` entirely — turned out to be treating a symptom, not the cause: the exact same ffmpeg error recurred through the Python pipe too, proving it was never about `std::system()` or whisper-server's request handling (which is fully serialized behind a single mutex — no concurrency involved either). The real cause: `VoiceDictationService` created its `MediaRecorder` with no explicit `mimeType` (`new MediaRecorder(stream)`), so Firefox picked its own default — Ogg, not WebM, for audio-only capture — while the code elsewhere labeled the resulting `Blob` `type: 'audio/webm'` regardless (cosmetic, ignored by ffmpeg's content-based probing). Ogg's page-based structure isn't safely truncatable at an arbitrary boundary the way WebM/Matroska's cluster-based structure is, which matters here specifically because every streaming-partial fire re-slices the *entire* growing chunk buffer from the start — an unlucky page boundary in that growing buffer produced an unparseable stream, at a frequency across real dictation that a single synthetic (already-complete) test clip never exercised.

Fixed by explicitly requesting `'audio/webm;codecs=opus'` on the `MediaRecorder` (with an `isTypeSupported` check, falling back to the browser default if unavailable). The Python-side conversion move is kept regardless — it's a real improvement (no temp files, no shell layer, matches the OpenVINO backend's already-proven approach) even though it wasn't the actual fix.

## Not yet done

- French transcription quality has not been directly compared between the two backends — only English (`jfk.wav`).
- Preserving streaming partials past Whisper's fixed 30-second window (`CONTEXT.md`'s "30-second limit (not yet implemented)") is a pre-existing gap in the OpenVINO pipeline too, not something this migration introduces or resolves.
- Only the `small` multilingual model is wired up for the whisper.cpp backend; the OpenVINO backend's other three variants (tiny/base/French-specialized large) have no whisper.cpp equivalent yet.
