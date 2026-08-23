# ADR-0018: Code-Tuned Embedding Model, Per-Space Model Tracking, Recompute Tool

**Date:** 2026-08-24
**Status:** Accepted

## Context

Live-tested `/rag-search` against real source code with the ADR-0015 default model
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a general sentence-paraphrase
model) and got poor results: a query like "subprocess execute myapp cpp" (a keyword-style
description of a function that shells out to a C++ binary) ranked an unrelated file higher
(0.509) than the genuinely relevant one (0.471), with all scores tightly clustered. That model was
trained on natural-language sentence pairs (paraphrase/NLI/STS datasets) — never on source code,
and never on short keyword-style queries — so outside that distribution it has little
discriminative power: ranking becomes dominated by superficial lexical overlap rather than real
semantics.

## Decision

### Default model: `jinaai/jina-embeddings-v2-base-code`

Trained specifically on (natural language, code) and (code, code) pairs across 30 programming
languages — matches this app's actual retrieval task far better than a general paraphrase model.
Trade-off: English + code only, not multilingual, so a space holding non-code, non-English prose
(the "index anything" spaces ADR-0015 designed for) would get worse semantic quality from this
model than the old one. Accepted for now since every space created through `/rag-index` so far is
source code; revisit if/when a genuinely general-prose space becomes a real use case.

### Provider registry keyed by model name, not a single global singleton

Switching the default doesn't retroactively change what any existing space's stored chunk vectors
mean — those vectors are only comparable to a query embedded with the *same* model. So
`rag.embedding.get_embedding_provider()` changed from a single cached instance to a `dict[model
name, provider]`, and `rag.store.add_source_and_chunks`/`search` now resolve the provider via the
specific space's own `RagSpace.embedding_model` rather than always the current global default.
This means two spaces can legitimately sit on different models at once — one predating this
change, not yet recomputed, keeps working correctly against its original model; a newly-created
space already uses the new default. Loading more than one small CPU model at once costs little
(each is well under a GB, no GPU/VRAM involved).

This was flagged as a likely future need in ADR-0015 ("per-space model choice... left to the real
UI") — it turned out to be needed sooner, not for a UI, but simply to keep old and new spaces both
correct during a model transition.

### `rag_recompute.py`: re-embed a space's existing chunks onto a different model

A standalone CLI script (matches the existing `migrate_item_result_format.py` pattern: argparse,
dry-run by default, `--apply` to actually write) that re-embeds a space's chunks in place and
updates its recorded `embedding_model`/`embedding_dim`. Deliberately re-embeds only — it does not
re-chunk or re-read source files, since chunk boundaries don't depend on the embedding model and
the original files/workspace may not even be available anymore. `--list` shows every space's
current model so it's easy to see what still needs recomputing after a default change.

## Verification

Reproduced the reported failure shape directly: a pasted "subprocess-launches-a-C++-binary"
function vs. an unrelated "folder grouping backend" class, queried with the same kind of
keyword-style text ("subprocess execute myapp cpp"), against an isolated temp DB. The old model
already separated these two particular synthetic examples correctly (0.574 vs 0.412) — a reminder
that the synthetic repro isn't a perfect match for the messier real files that produced the
original 0.509/0.471 near-tie — but recomputing the same space onto the new model produced a much
larger, cleaner separation (0.649 vs 0.158), consistent with meaningfully better discrimination
for this task. `rag_recompute.py --apply` was exercised directly (in-process, not via subprocess,
to keep the test on the isolated DB) and confirmed to update both the chunk vectors and the
space's recorded model/dimension correctly. The existing ingestion/search/websocket regression
scripts were re-run afterward and still pass unchanged.
