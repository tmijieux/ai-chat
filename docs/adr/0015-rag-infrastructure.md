# ADR-0015: RAG Infrastructure — Spaces, Ingestion, Embedding, Storage

**Date:** 2026-08-23
**Status:** Accepted

## Context

`todo.md` listed retrieval-augmented generation as an idea to explore. Discussion settled the
shape: multiple independent, named "spaces" (not one shared index for everything), each indexable
from arbitrary text sources, queryable by embedding similarity. This ADR covers the backend
infrastructure only — data model, ingestion, chunking, embedding, and storage/query. Agent tool
wiring (how/when the agent invokes retrieval) and any frontend UI are deliberately deferred to
follow-up work; this infrastructure was built and verified standalone first.

## Decision

### Storage: embeddings as base64 blobs in existing SQLite, not a separate vector file

Initially considered "brute-force vectors in a flat file per space." Settled instead on storing
each chunk's embedding as base64-encoded float32 bytes in a `Text` column on a new `rag_chunks`
table inside the existing `chat_db.sqlite` — mirroring how `Image.data` already stores a base64
blob in a `Text` column (ADR for vision input). Avoids running two storage systems (DB + a
filesystem vector store) that would have to stay in sync; new tables (`rag_spaces`, `rag_sources`,
`rag_chunks`) appear automatically via SQLAlchemy `create_all()`, the same mechanism every prior
feature's tables went through.

### Similarity search: brute-force cosine, not an ANN index

`rag.store.search()` loads every chunk row for a space, decodes embeddings into a numpy matrix, and
computes cosine similarity via one matmul — exact results, no index structure (HNSW/IVF/etc.)
involved. This is a deliberate scale call: an index trades exactness for query speed once a corpus
reaches roughly tens of thousands of vectors; a personal, per-space collection realistically sits
far below that, and brute-force cosine over a numpy matrix stays comfortably fast there. The
similarity computation is isolated inside `search()` specifically so it can be replaced later
(e.g. `sqlite-vec`'s HNSW) without any ingestion/API code above it changing, if a space ever
outgrows this.

### Embedding model: CPU via fastembed, not a second llama-server

Chose a CPU-based embedding provider (`fastembed`, ONNX Runtime, no torch dependency) over running
an embedding-capable GGUF model through a second `llama-server` instance. The chat model and image
generation already contend for the machine's single 8GB GPU (see ADR-0013's stop/restart handoff);
paying that same cost on every RAG query — not just an occasional explicit action like image
generation — would make retrieval prohibitively slow. Embedding models are tiny next to the chat
LLM and fast on CPU, so this trades nothing meaningful for zero VRAM contention.

Default model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim,
multilingual — the app's STT already handles French, so the embedding model needed to too).
`intfloat/multilingual-e5-small` was the original choice discussed but isn't in fastembed's
supported-model catalog; this was the smallest multilingual alternative fastembed actually ships.

`rag.embedding.EmbeddingProvider` is an abstract interface, introduced now rather than deferred,
specifically because a GPU-based provider was flagged as worth exploring later — for fast bulk
ingestion of a large repo, accepting the same stop-the-chat-model cost `generate_image` already
pays, if that ever proves worthwhile for large-scale indexing. Only the CPU provider
(`FastEmbedProvider`) is implemented now; the interface exists so that swap wouldn't touch
`ingestion.py`/`store.py`.

### Chunking: retrieval-sized with overlap, distinct from map-codebase's summarization chunking

`rag.chunking.chunk_text()` reuses the same character-budget-per-line approach as
`agent/workflow_coordinator.py`'s `chunk_file` (`~4 chars/token` estimate) but is a separate
function tuned for a different purpose: ~400 tokens/chunk (vs. map-codebase's 12,000, sized for
summarization context budgets) with ~50-token overlap between consecutive chunks. `chunk_file`
doesn't need overlap because its chunks are always folded back together by an LLM pass immediately
after; RAG chunks are retrieved and read in isolation from each other, so a fact split across a
chunk boundary needs to survive whole in at least one chunk.

### Ingestion: one source per file, content-hash-gated re-indexing

A `RagSource` is one document — one pasted text, one uploaded file, or (for a workspace-path
source) one file on disk. Pointing ingestion at a directory produces one source *per file* inside
it (recursive walk, reusing the same `.gitignore`/hardcoded-dir filtering the agent's own file
tools use), not one source for the whole directory. Each source's content hash is checked against
any existing source at the same path before re-embedding, so re-running ingestion on a workspace
path only re-embeds files that actually changed — this was a stated priority (ingestion needs to
stay fast on repeat/incremental runs, not just first-time).

### Space binding: global by default, optionally pinned to a workspace

`RagSpace.workspace_path` is nullable — most spaces are global and reusable across any conversation
or workspace, but a space can optionally record a workspace it's conceptually tied to (e.g. one
auto-populated from a specific repo). This is informational only at the infrastructure level: the
`workspace-path` ingestion endpoint always takes an explicit `workspace` argument regardless of a
space's own binding, so a global space can still ingest from any workspace on demand.

## Verification

Two levels, both against throwaway/isolated data — never `chat_db.sqlite`:
1. Unit-level, temp SQLite DB: ingested English/French text via `rag.ingestion`, queried via
   `rag.store.search`, confirmed the multilingual model ranks the correct chunk top in both
   languages.
2. HTTP-level, temp SQLite DB, through the real `routers/rag.py` endpoints via `TestClient`:
   created a space, ingested a pasted note and this repo's own `docs/adr/` directory (14 files) via
   the workspace-path endpoint, confirmed cross-document semantic queries ranked correctly (e.g.
   "workflow run visualisation stages" → ADR-0009 top), then deleted the space and confirmed full
   cascade cleanup.
