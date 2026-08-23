"""Storage and retrieval for RAG chunks: chunk + embed + persist a source, and brute-force cosine
similarity search over a space's chunks.

Vectors are stored as base64-encoded float32 bytes in a Text column (rag_chunks.embedding), the
same base64-blob-in-Text convention Image.data already uses, rather than a separate on-disk vector
file — avoids running two storage systems that would have to stay in sync. Similarity is computed
by scanning every chunk in the space (a single numpy matmul) rather than through a prebuilt index
structure (e.g. HNSW/IVF) — exact results, and fast enough at the corpus sizes a personal, per-space
collection realistically reaches; an index would trade exactness for speed at a scale this doesn't
need yet, and can replace this function's internals later without changing anything above it.
"""
import asyncio
import base64
import hashlib
import logging
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

import tables as db
from conv_helpers import _now
from rag.chunking import chunk_text
from rag.embedding import get_embedding_provider

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Return a stable hash of text content, used to skip re-embedding unchanged sources."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _encode_embedding(vector: list[float]) -> str:
    """Encode a float vector as base64 text, matching Image.data's base64-blob-in-Text convention."""
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def _decode_embedding(encoded: str) -> np.ndarray:
    """Decode a base64-encoded float32 embedding back into a numpy array."""
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


async def add_source_and_chunks(
    sess: AsyncSession,
    space_id: str,
    source_type: str,
    title: str,
    origin_path: str | None,
    text: str,
    existing_source: db.RagSource | None = None,
) -> db.RagSource:
    """Chunk `text`, embed all chunks in one batch, and persist a RagSource + its RagChunk rows.
    If `existing_source` is given, its old chunks are deleted and the row updated in place rather
    than duplicated — used when re-indexing a workspace-path source whose content changed."""
    space = await sess.get(db.RagSpace, space_id)
    if space is None:
        raise ValueError(f"No such RAG space: {space_id}")

    chunks = chunk_text(text)
    provider = get_embedding_provider()
    logger.info("[rag] Embedding %d chunk(s) for source '%s'", len(chunks), title)
    # fastembed's embed() is a blocking CPU call — offload it so a large ingestion run doesn't
    # stall the event loop for every other request while it computes (same reasoning as offloading
    # sd-cli.exe's blocking call in imagegen_pipeline.py).
    vectors = await asyncio.to_thread(provider.embed, [chunk.text for chunk in chunks])

    now = _now()
    content_hash = compute_content_hash(text)

    if existing_source is not None:
        source = existing_source
        await sess.execute(delete(db.RagChunk).where(db.RagChunk.source_id == source.id))
        source.content_hash = content_hash
        source.status = "indexed"
        source.error_message = None
        source.updated_at = now
    else:
        source = db.RagSource(
            id=str(uuid.uuid4()), space_id=space_id, source_type=source_type, title=title,
            origin_path=origin_path, content_hash=content_hash, status="indexed",
            error_message=None, created_at=now, updated_at=now,
        )
        sess.add(source)

    await sess.flush()

    for chunk, vector in zip(chunks, vectors):
        sess.add(db.RagChunk(
            id=str(uuid.uuid4()), source_id=source.id, space_id=space_id, chunk_index=chunk.index,
            start_line=chunk.start_line, end_line=chunk.end_line, text=chunk.text,
            embedding=_encode_embedding(vector), created_at=now,
        ))

    return source


@dataclass
class SearchResult:
    """One ranked chunk returned from a RAG space query."""
    chunk_id: str
    source_id: str
    source_title: str
    origin_path: str | None
    text: str
    score: float


async def search(sess: AsyncSession, space_id: str, query: str, top_k: int = 5) -> list[SearchResult]:
    """Embed `query` and return the top_k most similar chunks in the space via brute-force cosine
    similarity."""
    rows = (await sess.execute(
        select(db.RagChunk, db.RagSource)
        .join(db.RagSource, db.RagChunk.source_id == db.RagSource.id)
        .where(db.RagChunk.space_id == space_id)
    )).all()

    if len(rows) == 0:
        return []

    provider = get_embedding_provider()
    query_vector = np.asarray(provider.embed([query])[0], dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return []

    matrix = np.stack([_decode_embedding(chunk.embedding) for chunk, _source in rows])
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-12
    scores = (matrix @ query_vector) / (norms * query_norm)

    ranked_indices = np.argsort(-scores)[:top_k]
    return [
        SearchResult(
            chunk_id=rows[i][0].id, source_id=rows[i][0].source_id,
            source_title=rows[i][1].title, origin_path=rows[i][1].origin_path,
            text=rows[i][0].text, score=float(scores[i]),
        )
        for i in ranked_indices
    ]
