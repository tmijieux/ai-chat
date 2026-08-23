"""Text ingestion for RAG spaces — pasted text, an uploaded file, or a workspace path — all
converging on rag.store.add_source_and_chunks once reduced to (title, origin_path, text)."""
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import tables as db
from agent.file_utils import file_in_directory, is_path_ignored, load_ignore_spec, resolve_workspace_path
from rag.store import add_source_and_chunks, compute_content_hash

# Extensions treated as non-text — no useful chunk comes from reading these as source text, and
# several (media, archives) can be large enough to matter.
_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".mov", ".avi", ".webm",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pdf", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pyc", ".pyo", ".o", ".a", ".so", ".dll", ".exe", ".bin", ".wasm",
    ".gguf", ".safetensors", ".onnx", ".pt", ".pth",
    ".sqlite", ".sqlite3", ".db",
}


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


async def ingest_workspace_path(sess: AsyncSession, space_id: str, workspace: str, relative_path: str) -> list[db.RagSource]:
    """Ingest a file or directory from a workspace. A directory is walked recursively (same
    .gitignore/hardcoded-dir filtering the agent's file tools use) and produces one source per
    matched text file. A file whose content hash matches an existing source at the same
    (space_id, origin_path) is skipped — makes re-running ingestion on the same path incremental."""
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
            and path.suffix.lower() not in _BINARY_EXTENSIONS
        ]

    sources: list[db.RagSource] = []
    for path in candidate_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        origin_path = path.relative_to(workspace).as_posix()
        content_hash = compute_content_hash(text)

        existing = await sess.scalar(
            select(db.RagSource).where(
                db.RagSource.space_id == space_id,
                db.RagSource.origin_path == origin_path,
            )
        )
        if existing is not None and existing.content_hash == content_hash:
            sources.append(existing)
            continue

        source = await add_source_and_chunks(
            sess, space_id=space_id, source_type="workspace_path", title=path.name,
            origin_path=origin_path, text=text, existing_source=existing,
        )
        sources.append(source)

    return sources
