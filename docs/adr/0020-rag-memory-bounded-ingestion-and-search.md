# ADR-0020: Memory-Bounded Chunking, Ingestion, and Search

**Date:** 2026-08-26
**Status:** Accepted

## Context

Two related memory blowups showed up in practice, both from treating "all of a file's chunks" or
"all of a space's chunks" as something safe to materialize at once:

- Ingesting a single very large file (observed: ~1.4M lines, ~20GB of Python objects once chunked)
  went through `chunk_text` returning a fully-built `list[Chunk]`, then embedded every chunk in one
  `provider.embed()` call — holding every chunk's text, every input string passed to the model, and
  every output vector for the *entire file* resident simultaneously, on top of the raw file text
  already read into memory. Small files were never the problem; multi-million-line files are, and
  per-file session commits (see ADR-0017) don't help here since the blowup happens processing a
  single file, not across many.
- `rag.store.search()` queried every chunk *and* its joined source row for the whole space via
  `.all()`, then stacked every embedding into one dense numpy matrix, to do one vectorized cosine
  similarity matmul. Correct and fast for a small space, but memory scales with the total number of
  chunks ever ingested into that space, and every chunk's full text is pulled along for the ride
  even though only the top few results are ever used.

## Decision

### `chunk_text` is a generator, not a list-builder

`rag/chunking.py`'s `chunk_text` now yields `Chunk`s lazily as it scans lines, instead of building
the whole list before returning. Same chunking logic and boundaries, just lazy — a caller can
process (and discard) chunks as it goes instead of forcing the whole file's chunk set into memory
up front.

### Ingestion embeds and persists in bounded windows, not one shot

`rag.store.add_source_and_chunks` pulls chunks off `chunk_text`'s generator in fixed-size windows
(`_INGEST_WINDOW_SIZE = 200`), embeds one window at a time (still offloaded to a thread — fastembed
remains a blocking CPU call), persists that window's `RagChunk` rows, flushes, and then
`session.expunge()`s them. Expunging right after flush matters because `AsyncSessionLocal` is
configured with `expire_on_commit=False` (see ADR-0017) —
without it, the session's identity map would keep every window's chunk text and embedding resident
for the rest of that file's ingestion regardless of flushing. Net effect: memory during ingestion of
one file is bounded by window size, not by the file's total chunk count, whether that file has 50
chunks or 500,000.

### Search scores in two passes: id+embedding scan, then fetch winners only

`search()` first streams (`AsyncSession.stream()`) only `RagChunk.id` and `RagChunk.embedding` for
the space — no chunk text, no join to `RagSource` — scoring each row against the query vector and
keeping a size-`top_k` min-heap (`heapq`) of the best scores seen so far. Only once the winning
chunk ids are known does a second, small query fetch the full `RagChunk`+`RagSource` rows (a `WHERE
id IN (...)` over just those ids) to build the actual `SearchResult`s. Trade-off accepted: this
replaces a single vectorized numpy matmul over the whole space with a per-row Python loop, which is
slower per-chunk — acceptable since it turns "memory scales with total chunks in the space" into
"memory scales with `top_k`", and per-space corpora here are small enough that the loop's cost
stays negligible in absolute terms. Still exact brute-force cosine similarity, not an approximate
index — same reasoning as the original design in ADR-0015.

## Verification

Against an isolated temp SQLite DB (never `chat_db.sqlite`): confirmed `chunk_text` is a generator
function; forced `_INGEST_WINDOW_SIZE` down to 7 against a ~400-line synthetic text and confirmed
the windowed ingestion persists exactly the same chunks, in the same order, as calling `chunk_text`
directly — no drops or duplicates across window boundaries. Ran `search()` against that same space
and compared its ranking to an independent brute-force reference computed by hand over every
chunk's embedding — the two-pass streamed result matched the reference top-5 exactly, in the same
order. Also confirmed `search()` on a space with zero chunks returns `[]` without error.
