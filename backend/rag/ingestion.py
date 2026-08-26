"""Text ingestion for RAG spaces — pasted text, an uploaded file, or a workspace path — all
converging on rag.store.add_source_and_chunks once reduced to (title, origin_path, text)."""
import logging
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import tables as db
from agent.file_utils import file_in_directory, is_path_ignored, load_ignore_spec, resolve_workspace_path
from rag.store import add_source_and_chunks, compute_content_hash

logger = logging.getLogger(__name__)

# Called with (current index starting at 1, total file count, workspace-relative path) as each
# file starts processing. Synchronous — the caller (e.g. the ingest websocket) just enqueues an
# event, it doesn't await anything here.
ProgressCallback = Callable[[int, int, str], None]


async def ingest_pasted_text(sess: AsyncSession, space_id: str, title: str, text: str) -> db.RagSource:
    """Ingest directly-pasted text as one source."""
    return await add_source_and_chunks(
        sess, space_id=space_id, source_type="paste", title=title, origin_path=None, text=text,
    )


async def ingest_uploaded_file(sess: AsyncSession, space_id: str, filename: str, raw_bytes: bytes) -> db.RagSource:
    """Ingest an uploaded text/markdown file as one source. Raises ValueError if not valid UTF-8 text."""
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"'{filename}' is not a valid UTF-8 text file") from e
    return await add_source_and_chunks(
        sess, space_id=space_id, source_type="upload", title=filename, origin_path=filename, text=text,
    )


async def ingest_workspace_path(
    sess: AsyncSession,
    space_id: str,
    workspace: str,
    relative_path: str,
    on_progress: ProgressCallback | None = None,
) -> list[db.RagSource]:
    """Ingest a file or directory from a workspace. A directory is walked recursively (same
    .gitignore/hardcoded-dir filtering the agent's file tools use) and produces one source per
    matched text file. A file whose content hash matches an existing source at the same
    (space_id, origin_path) is skipped — makes re-running ingestion on the same path incremental.
    `on_progress`, if given, is called once per file as it starts processing — the caller can
    cancel between calls (e.g. via asyncio task cancellation), which this function does not catch,
    so whatever's already been flushed to the session stays (nothing is lost on cancel)."""
    target = resolve_workspace_path(relative_path, workspace)
    if not file_in_directory(str(target), workspace) and target != Path(workspace).resolve():
        raise ValueError(f"Ingesting outside the workspace is forbidden: {relative_path}")

    if target.is_file():
        candidate_paths = [target]
    else:
        spec = load_ignore_spec(workspace)
        candidate_paths = [
            path for path in sorted(target.rglob("*"))
            if path.is_file()
            and not is_path_ignored(path, workspace, spec)
        ]

    total = len(candidate_paths)
    logger.info("[rag] Ingesting %d candidate file(s) from '%s' (space %s)", total, relative_path, space_id)

    sources: list[db.RagSource] = []
    for index, path in enumerate(candidate_paths, start=1):
        origin_path = path.relative_to(workspace).as_posix()
        if on_progress is not None:
            on_progress(index, total, origin_path)

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            logger.info("[rag] (%d/%d) skipped (unreadable): %s", index, total, origin_path)
            continue

        content_hash = compute_content_hash(text)

        existing = await sess.scalar(
            select(db.RagSource).where(
                db.RagSource.space_id == space_id,
                db.RagSource.origin_path == origin_path,
            )
        )
        if existing is not None and existing.content_hash == content_hash:
            logger.info("[rag] (%d/%d) unchanged, skipped: %s", index, total, origin_path)
            sources.append(existing)
            continue

        logger.info("[rag] (%d/%d) indexing: %s", index, total, origin_path)
        source = await add_source_and_chunks(
            sess, space_id=space_id, source_type="workspace_path", title=path.name,
            origin_path=origin_path, text=text, existing_source=existing,
        )
        sources.append(source)

    logger.info("[rag] Finished ingesting '%s': %d source(s) processed (space %s)", relative_path, len(sources), space_id)
    return sources
