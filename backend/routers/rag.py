"""RAG spaces: create/list/delete spaces, ingest text sources (paste/upload/workspace-path),
list/delete sources, and query a space's chunks. Backend-only surface — no agent tool wiring yet."""
import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

import loaders as ld
import tables as db
from conv_helpers import _now
from database import get_db_session
from rag.embedding import get_embedding_provider
from rag.ingestion import ingest_pasted_text, ingest_uploaded_file, ingest_workspace_path
from rag.store import search as rag_search

router = APIRouter()
logger = logging.getLogger(__name__)

_TASK_CANCEL_TIMEOUT_SECONDS = 5


def _space_dict(space: db.RagSpace) -> dict:
    return {
        "id": space.id, "name": space.name, "description": space.description,
        "workspace_path": space.workspace_path, "embedding_model": space.embedding_model,
        "embedding_dim": space.embedding_dim, "created_at": space.created_at,
    }


def _source_dict(source: db.RagSource) -> dict:
    return {
        "id": source.id, "space_id": source.space_id, "source_type": source.source_type,
        "title": source.title, "origin_path": source.origin_path, "status": source.status,
        "error_message": source.error_message, "created_at": source.created_at,
        "updated_at": source.updated_at,
    }


async def _get_space_or_404(sess: AsyncSession, space_id: str) -> db.RagSpace:
    """Return the space or raise HTTP 404 if it doesn't exist."""
    space = await sess.get(db.RagSpace, space_id)
    if space is None:
        raise HTTPException(404, f"No such RAG space: {space_id}")
    return space


@router.post("/api/rag/spaces")
async def create_space(body: ld.NewRagSpace, sess: AsyncSession = Depends(get_db_session)):
    provider = get_embedding_provider()
    space = db.RagSpace(
        id=str(uuid.uuid4()), name=body.name, description=body.description,
        workspace_path=body.workspace_path, embedding_model=provider.model_name,
        embedding_dim=provider.dimension, created_at=_now(),
    )
    sess.add(space)
    await sess.flush()
    return _space_dict(space)


@router.get("/api/rag/spaces")
async def list_spaces(sess: AsyncSession = Depends(get_db_session)):
    spaces = (await sess.execute(select(db.RagSpace))).scalars().all()
    return [_space_dict(space) for space in spaces]


@router.delete("/api/rag/spaces/{space_id}")
async def delete_space(space_id: str, sess: AsyncSession = Depends(get_db_session)):
    await sess.execute(delete(db.RagChunk).where(db.RagChunk.space_id == space_id))
    await sess.execute(delete(db.RagSource).where(db.RagSource.space_id == space_id))
    await sess.execute(delete(db.RagSpace).where(db.RagSpace.id == space_id))
    return {"deleted": space_id}


@router.post("/api/rag/spaces/{space_id}/sources/paste")
async def add_pasted_source(space_id: str, body: ld.NewPastedRagSource, sess: AsyncSession = Depends(get_db_session)):
    await _get_space_or_404(sess, space_id)
    source = await ingest_pasted_text(sess, space_id, body.title, body.text)
    return _source_dict(source)


@router.post("/api/rag/spaces/{space_id}/sources/upload")
async def add_uploaded_source(space_id: str, file: UploadFile, sess: AsyncSession = Depends(get_db_session)):
    await _get_space_or_404(sess, space_id)
    raw_bytes = await file.read()
    try:
        source = await ingest_uploaded_file(sess, space_id, file.filename or "untitled.txt", raw_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return _source_dict(source)


@router.post("/api/rag/spaces/{space_id}/sources/workspace-path")
async def add_workspace_path_source(space_id: str, body: ld.NewWorkspacePathRagSource, sess: AsyncSession = Depends(get_db_session)):
    await _get_space_or_404(sess, space_id)
    try:
        sources = await ingest_workspace_path(space_id, body.workspace, body.path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return [_source_dict(source) for source in sources]


async def _cancel_and_wait(task: asyncio.Task, label: str) -> None:
    """Cancel a task and wait, bounded, for it to unwind. Mirrors routers/ws.py's helper of the
    same shape — kept local since each router here stays self-contained."""
    task.cancel()
    done, _pending = await asyncio.wait({task}, timeout=_TASK_CANCEL_TIMEOUT_SECONDS)
    if len(done) == 0:
        logger.warning("[rag ws] %s did not stop within %ss of cancellation", label, _TASK_CANCEL_TIMEOUT_SECONDS)


@router.websocket("/api/rag/spaces/{space_id}/sources/workspace-path/ws")
async def ingest_workspace_path_ws(websocket: WebSocket, space_id: str, sess: AsyncSession = Depends(get_db_session)):
    """Streams per-file ingestion progress ({type: progress, current, total, filename}) and
    supports cancellation ({type: cancel} from the client) — the interactive counterpart to the
    plain POST endpoint above, used by the /rag-index slash command so the UI can show real
    progress instead of a bare spinner. The POST endpoint stays for simple/programmatic use."""
    await websocket.accept()
    try:
        init_data = await websocket.receive_json()
        workspace: str = init_data["workspace"]
        path: str = init_data.get("path") or "."

        space = await sess.get(db.RagSpace, space_id)
        if space is None:
            await websocket.send_json({"type": "error", "message": f"No such RAG space: {space_id}"})
            return

        outbound: asyncio.Queue = asyncio.Queue()

        def on_progress(current: int, total: int, filename: str) -> None:
            outbound.put_nowait({"type": "progress", "current": current, "total": total, "filename": filename})

        async def run_ingestion() -> None:
            # ingest_workspace_path commits each file through its own short-lived session, so
            # progress is already durable as it goes — nothing left to commit here on completion
            # or cancellation.
            try:
                sources = await ingest_workspace_path(space_id, workspace, path, on_progress=on_progress)
                await outbound.put({"type": "done", "sources": [_source_dict(s) for s in sources]})
            except ValueError as e:
                await outbound.put({"type": "error", "message": str(e)})
            except asyncio.CancelledError:
                raise

        ingest_task = asyncio.create_task(run_ingestion())

        async def send_loop() -> None:
            while True:
                event = await outbound.get()
                await websocket.send_json(event)
                if event["type"] in ("done", "error", "cancelled"):
                    return

        async def recv_loop() -> None:
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("type") == "cancel":
                        await outbound.put({"type": "cancelled"})
                        return
            except WebSocketDisconnect:
                return

        send_task = asyncio.create_task(send_loop())
        recv_task = asyncio.create_task(recv_loop())
        try:
            await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await _cancel_and_wait(recv_task, "rag ingest websocket receive task")
            await _cancel_and_wait(send_task, "rag ingest websocket send task")
            await _cancel_and_wait(ingest_task, "rag ingest task")
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("error in rag ingest websocket handling")


@router.get("/api/rag/spaces/{space_id}/sources")
async def list_sources(space_id: str, sess: AsyncSession = Depends(get_db_session)):
    sources = (await sess.execute(select(db.RagSource).where(db.RagSource.space_id == space_id))).scalars().all()
    return [_source_dict(source) for source in sources]


@router.delete("/api/rag/spaces/{space_id}/sources/{source_id}")
async def delete_source(space_id: str, source_id: str, sess: AsyncSession = Depends(get_db_session)):
    await sess.execute(delete(db.RagChunk).where(db.RagChunk.source_id == source_id))
    await sess.execute(delete(db.RagSource).where(db.RagSource.id == source_id))
    return {"deleted": source_id}


@router.post("/api/rag/spaces/{space_id}/query")
async def query_space(space_id: str, body: ld.RagQuery, sess: AsyncSession = Depends(get_db_session)):
    await _get_space_or_404(sess, space_id)
    results = await rag_search(sess, space_id, body.query, body.top_k)
    return [
        {
            "chunk_id": r.chunk_id, "source_id": r.source_id, "source_title": r.source_title,
            "origin_path": r.origin_path, "text": r.text, "score": r.score,
        }
        for r in results
    ]
