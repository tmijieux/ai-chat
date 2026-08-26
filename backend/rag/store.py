"""Storage and retrieval for RAG chunks: chunk + embed + persist a source, and brute-force cosine
similarity search over a space's chunks.

Vectors are stored as base64-encoded float32 bytes in a Text column (rag_chunks.embedding), the
same base64-blob-in-Text convention Image.data already uses, rather than a separate on-disk vector
file — avoids running two storage systems that would have to stay in sync. Similarity is computed
by scanning every chunk in the space one row at a time, keeping only a top-k heap, rather than
through a prebuilt index structure (e.g. HNSW/IVF) — exact results, and fast enough at the corpus
sizes a personal, per-space collection realistically reaches; an index would trade exactness for
speed at a scale this doesn't need yet, and can replace this function's internals later without
changing anything above it.
"""
import asyncio
import base64
import hashlib
import heapq
import itertools
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

# Chunks are pulled off chunk_text's generator and embedded/persisted this many at a time, so
# ingesting one huge file (e.g. a multi-million-line log) never holds more than one window's worth
# of chunk text + embeddings in memory — only the window's worth of Chunk objects and vectors are
# ever resident, regardless of how many chunks the whole file produces.
_INGEST_WINDOW_SIZE = 200


def compute_content_hash(text: str) -> str:
    """Return a stable hash of text content, used to skip re-embedding unchanged sources."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_embedding(vector: list[float]) -> str:
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
    """Chunk `text` and persist a RagSource + its RagChunk rows, embedding chunks in bounded
    windows (see _INGEST_WINDOW_SIZE) rather than all at once — a multi-million-line file produces
    far more Chunk objects and embeddings than should ever be resident in memory simultaneously.
    If `existing_source` is given, its old chunks are deleted and the row updated in place rather
    than duplicated — used when re-indexing a workspace-path source whose content changed."""
    space = await sess.get(db.RagSpace, space_id)
    if space is None:
        raise ValueError(f"No such RAG space: {space_id}")

    # Use the model THIS space is already on, not whatever the current global default is — a space
    # predating a default-model change keeps its existing model's chunks comparable until it's
    # explicitly recomputed (see rag_recompute.py), rather than silently mixing two models' vectors
    # in one space.
    provider = get_embedding_provider(space.embedding_model)

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

    chunk_iterator = chunk_text(text)
    total_chunks = 0
    while True:
        window = list(itertools.islice(chunk_iterator, _INGEST_WINDOW_SIZE))
        if len(window) == 0:
            break

        # fastembed's embed() is a blocking CPU call — offload it so a large ingestion run doesn't
        # stall the event loop for every other request while it computes (same reasoning as
        # offloading sd-cli.exe's blocking call in imagegen_pipeline.py).
        vectors = await asyncio.to_thread(lambda texts=[c.text for c in window]: list(provider.embed(texts)))

        rows = [
            db.RagChunk(
                id=str(uuid.uuid4()), source_id=source.id, space_id=space_id, chunk_index=chunk.index,
                start_line=chunk.start_line, end_line=chunk.end_line, text=chunk.text,
                embedding=encode_embedding(vector), created_at=now,
            )
            for chunk, vector in zip(window, vectors)
        ]
        sess.add_all(rows)
        await sess.flush()
        # Already flushed to the DB (within this still-open transaction) — expunge so this
        # window's chunk text/embeddings can be garbage-collected instead of sitting in the
        # session's identity map for the rest of the file's ingestion.
        for row in rows:
            sess.expunge(row)

        total_chunks += len(window)

    logger.info("[rag] Embedded %d chunk(s) for source '%s' with model '%s'", total_chunks, title, provider.model_name)
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
    similarity. The query is embedded with the SAME model the space's chunks were embedded with
    (RagSpace.embedding_model) — using any other model's vector would be meaningless to compare
    against these chunks' vectors, and could even be a different dimension entirely.

    Scored in two passes so a large space is never loaded into memory at once: the first pass
    streams only each chunk's id and embedding (no chunk text, no join to RagSource) and keeps a
    top_k heap of the best scores seen so far; the second pass fetches the full chunk + source rows
    for just those top_k winners, once the winners are already known."""
    space = await sess.get(db.RagSpace, space_id)
    if space is None:
        raise ValueError(f"No such RAG space: {space_id}")

    provider = get_embedding_provider(space.embedding_model)
    query_vector = np.asarray(await asyncio.to_thread(provider.embed, [query]), dtype=np.float32)[0]
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return []

    best: list[tuple[float, str]] = []  # min-heap of (score, chunk_id), capped at top_k entries
    id_and_embedding = await sess.stream(
        select(db.RagChunk.id, db.RagChunk.embedding).where(db.RagChunk.space_id == space_id)
    )
    async for chunk_id, embedding in id_and_embedding:
        vector = _decode_embedding(embedding)
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        score = float(np.dot(vector, query_vector) / (norm * query_norm))
        if len(best) < top_k:
            heapq.heappush(best, (score, chunk_id))
        elif score > best[0][0]:
            heapq.heapreplace(best, (score, chunk_id))

    if len(best) == 0:
        return []

    score_by_chunk_id = {chunk_id: score for score, chunk_id in best}
    rows = (await sess.execute(
        select(db.RagChunk, db.RagSource)
        .join(db.RagSource, db.RagChunk.source_id == db.RagSource.id)
        .where(db.RagChunk.id.in_(score_by_chunk_id.keys()))
    )).all()

    results = [
        SearchResult(
            chunk_id=chunk.id, source_id=chunk.source_id, source_title=source.title,
            origin_path=source.origin_path, text=chunk.text, score=score_by_chunk_id[chunk.id],
        )
        for chunk, source in rows
    ]
    results.sort(key=lambda r: -r.score)
    return results
