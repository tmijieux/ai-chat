"""WebSocket endpoint for System Audio Transcription: capture an output device (or mic) and
stream a live transcript. See ADR-0022 and CONTEXT.md's "System Audio Transcription"."""
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from system_audio.capture import AudioCapture, list_sources
from system_audio.streaming_transcriber import StreamingTranscriber

router = APIRouter()
logger = logging.getLogger(__name__)

_TASK_CANCEL_TIMEOUT_SECONDS = 5.0


async def _cancel_and_wait(task: asyncio.Task, label: str) -> None:
    """Cancel a task and wait, bounded, for it to unwind. Mirrors routers/ws.py's helper."""
    task.cancel()
    done, _pending = await asyncio.wait({task}, timeout=_TASK_CANCEL_TIMEOUT_SECONDS)
    if len(done) == 0:
        logger.warning("[system-audio ws] %s did not stop within %ss", label, _TASK_CANCEL_TIMEOUT_SECONDS)


@router.websocket("/api/system-audio/ws")
async def system_audio_ws(websocket: WebSocket):
    """Send the device list on connect, then on {action: start} capture the chosen source and
    stream {type: committed|provisional|status|error} events until {action: stop} or disconnect."""
    import main as _main

    await websocket.accept()
    capture: AudioCapture | None = None
    try:
        if _main._whisper_ready == False:
            await websocket.send_json({"type": "error", "message": "STT backend is still loading, try again in a moment"})
            return

        await websocket.send_json({
            "type": "devices",
            "devices": [{"id": source.id, "label": source.label, "kind": source.kind} for source in list_sources()],
        })

        start_data = await websocket.receive_json()
        if start_data.get("action") != "start":
            return
        device_id = start_data["device_id"]
        kind = start_data.get("kind", "output")
        language_raw = start_data.get("language")
        language = language_raw if (language_raw is not None and language_raw != "") else None
        translate = start_data.get("translate", False) == True

        loop = asyncio.get_running_loop()
        outbound: asyncio.Queue = asyncio.Queue()
        transcriber = StreamingTranscriber(language, translate, outbound.put_nowait)

        def on_frame(frame) -> None:
            loop.call_soon_threadsafe(transcriber.feed, frame)

        capture = AudioCapture(device_id, kind, on_frame)
        capture.start()
        await websocket.send_json({"type": "status", "state": "listening"})

        await asyncio.sleep(0.4)
        if capture.error is not None:
            await websocket.send_json({"type": "error", "message": f"Could not capture that device: {capture.error}"})
            return

        async def send_loop() -> None:
            while True:
                event = await outbound.get()
                await websocket.send_json(event)

        async def recv_loop() -> None:
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("action") == "stop":
                        return
            except WebSocketDisconnect:
                return

        send_task = asyncio.create_task(send_loop())
        recv_task = asyncio.create_task(recv_loop())
        try:
            await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await _cancel_and_wait(send_task, "send loop")
            await _cancel_and_wait(recv_task, "receive loop")

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("error in system-audio websocket handling")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if capture is not None:
            await asyncio.to_thread(capture.stop)
