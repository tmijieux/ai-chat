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

### Embedding window size doubles as the model's batch size — kept small deliberately

After the fixes above, ingesting a small (1.3MB) but real file — a password-strength frequency
list shipped as a few giant comma-separated lines — still drove RSS to 15-20GB. The cause this time
wasn't Python-level retention at all: it was inside the ONNX Runtime inference call itself.
`_INGEST_WINDOW_SIZE` (200 at the time) is also the batch size passed to `provider.embed()`, and
self-attention memory scales with `batch_size * sequence_length^2`. This file's dense,
comma-separated text (short "words" with no natural spaces) tokenizes far worse than the
chars-per-token estimate `TARGET_CHUNK_CHARS` is tuned for — measured directly, its 1600-character
chunks average ~740 tokens and peak at 858, roughly double the ~400-token target. Batching 200 such
chunks into one inference call was enough to blow ONNX's memory arena up to double-digit GB (arena
allocators grow to fit the largest request and don't shrink back down between calls).

Fix: lowered `_INGEST_WINDOW_SIZE` from 200 to 16. Verified empirically, not assumed: embedding the
file's actual longest chunks in escalating batch sizes (1/8/16/32) showed clearly super-linear
growth as the batch grew, while re-running with a *constant* window of 16 across the whole file
held memory flat window over window — confirming a small, constant batch size is what actually
bounds this, not just a smaller total window count. No attempt to estimate token count ahead of
time and size the window adaptively — ordinary source code (this app's primary RAG use case, see
ADR-0018) tokenizes much closer to the character-based estimate, but nothing upstream guarantees
that for an arbitrary ingested file, so the window stays small unconditionally rather than trusting
a heuristic that this exact file already broke.

### Oversized-line splitting is also lazy, and now overlaps

A real source of the reported blowup: some real-world files (e.g. a frequency-list-style file) are
effectively one giant line, which hits `_split_oversized_line` rather than the normal line-based
path. That helper originally built its full list of fixed-size slices via a list comprehension
before returning — for a large enough single line, this alone measured hundreds of MB to
multi-GB, materialized in one shot, *before* `chunk_text` (a generator everywhere else) could yield
even its first chunk back. `_split_oversized_line` is now itself a generator, yielding one slice at
a time. While fixing it, also gave it the same overlap guarantee normal chunk boundaries already
had: consecutive slices now overlap by `OVERLAP_CHARS`, so a fact split at one of these fixed
character cuts still appears whole in at least one slice — previously it didn't, since the slices
were plain non-overlapping `line[i:i+max_chars]` cuts.

## Verification

Against an isolated temp SQLite DB (never `chat_db.sqlite`): confirmed `chunk_text` is a generator
function; forced `_INGEST_WINDOW_SIZE` down to 7 against a ~400-line synthetic text and confirmed
the windowed ingestion persists exactly the same chunks, in the same order, as calling `chunk_text`
directly — no drops or duplicates across window boundaries. Ran `search()` against that same space
and compared its ranking to an independent brute-force reference computed by hand over every
chunk's embedding — the two-pass streamed result matched the reference top-5 exactly, in the same
order. Also confirmed `search()` on a space with zero chunks returns `[]` without error.

Measured actual process RSS (via `psutil`) around a synthetic 300MB single line — before the
`_split_oversized_line` fix, RSS jumped by ~365MB on the very first chunk pulled from `chunk_text`
(the eager list-comprehension materializing all ~225K overlapping slices at once); after the fix,
RSS stayed flat through pulling all ~225K slices in windows of 200, confirming the fix actually
bounds memory rather than just preserving correct output.

Finally, ran the real `debug_freq_list/frequency_lists.ts` file (the actual file that triggered the
15-20GB reports) through the full real pipeline — real fastembed/ONNX embeddings, isolated temp DB
— with `_INGEST_WINDOW_SIZE = 16`: memory held flat across all real windows (observed directly via
Task Manager during the run), and a smaller slice of the same file through `add_source_and_chunks`
+ `search()` end-to-end persisted the expected chunk count and returned relevant results.
