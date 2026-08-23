"""Recompute a RAG space's chunk vectors onto a different embedding model, updating the space's
recorded embedding_model/embedding_dim to match. Re-embeds each chunk's already-stored text in
place — it does not re-chunk or re-read source files, so this works even if the workspace/files
that originally fed the space are no longer available.

Use this after changing rag.embedding.DEFAULT_EMBEDDING_MODEL, or to move a specific space onto a
different model: a space keeps using whatever model it was built with until recomputed here — new
spaces created after a default change already use the new default automatically, this script is
only for spaces that predate it. See ADR-0018.

Usage:
    python rag_recompute.py --list                              # list spaces and their current model
    python rag_recompute.py <space_id>                          # dry run, reports what would change
    python rag_recompute.py <space_id> --apply                  # recompute onto the current default model
    python rag_recompute.py <space_id> --model <name> --apply   # recompute onto a specific model
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import func, select

import tables as db
from database import AsyncSessionLocal
from rag.embedding import DEFAULT_EMBEDDING_MODEL, get_embedding_provider
from rag.store import encode_embedding

_EMBED_BATCH_SIZE = 64


async def _list_spaces() -> None:
    async with AsyncSessionLocal() as sess:
        spaces = (await sess.execute(select(db.RagSpace))).scalars().all()
        if len(spaces) == 0:
            print("No RAG spaces.")
            return
        for space in spaces:
            chunk_count = await sess.scalar(
                select(func.count()).select_from(db.RagChunk).where(db.RagChunk.space_id == space.id)
            )
            print(f"{space.id}  {space.name!r:30}  model={space.embedding_model!r}  chunks={chunk_count}")


async def _recompute(space_id: str, model_name: str, apply: bool) -> None:
    async with AsyncSessionLocal() as sess:
        space = await sess.get(db.RagSpace, space_id)
        if space is None:
            sys.exit(f"No such RAG space: {space_id}")

        chunks = (await sess.execute(select(db.RagChunk).where(db.RagChunk.space_id == space_id))).scalars().all()
        print(f"Space {space.name!r} ({space.id}): {len(chunks)} chunk(s), "
              f"current model={space.embedding_model!r} -> target model={model_name!r}")

        if not apply:
            print("[DRY RUN] pass --apply to actually recompute.")
            return

        if len(chunks) == 0:
            print("No chunks to recompute.")
            return

        provider = get_embedding_provider(model_name)
        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start:start + _EMBED_BATCH_SIZE]
            vectors = await asyncio.to_thread(provider.embed, [c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                chunk.embedding = encode_embedding(vector)
            print(f"  embedded {min(start + _EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")

        space.embedding_model = provider.model_name
        space.embedding_dim = provider.dimension
        await sess.commit()
        print(f"Done. Space now on model={provider.model_name!r} dim={provider.dimension}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space_id", nargs="?", help="RAG space id to recompute")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help=f"Target embedding model (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--apply", action="store_true", help="Actually recompute (default: dry run)")
    parser.add_argument("--list", action="store_true", help="List all RAG spaces and their current model, then exit")
    args = parser.parse_args()

    if args.list:
        asyncio.run(_list_spaces())
        return

    if args.space_id is None:
        parser.error("space_id is required unless --list is given")

    asyncio.run(_recompute(args.space_id, args.model, args.apply))


if __name__ == "__main__":
    main()
