import uuid
from typing import TYPE_CHECKING

from .base import BaseTool, tool_error
from tool_result_types import RagSearchResult, ToolResult

if TYPE_CHECKING:
    from agent.agent import AgentSession

_TOP_K = 5


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "Semantic search over the RAG space indexed for the current workspace (see /rag-index). "
        "Phrase the query as a full natural-language sentence describing what you're looking for "
        "(e.g. 'launch the myapp C++ simulation as a subprocess'), not a keyword list — the "
        "embedding model is code-tuned and rewards being asked that way. Returns the most relevant "
        "indexed chunks, not whole files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language sentence describing what to find.",
            },
        },
        "required": ["query"],
    }
    requires_confirmation = False
    measured_delta = 342

    def label(self, args: dict) -> str:
        return f"RAG SEARCH {args.get('query', '')[:60]}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        if working_directory is None:
            return tool_error(self.name, "No workspace configured — rag_search is disabled.")

        query = args.get("query", "").strip()
        if query == "":
            return tool_error(self.name, "query is required.")

        # Lazy imports to avoid circular dependency (pipeline/tools import chain).
        from database import AsyncSessionLocal
        import tables as db
        from sqlalchemy import select
        from conv_helpers import _now
        from rag.embedding import get_embedding_provider
        from rag.store import search as rag_search_chunks

        async with AsyncSessionLocal() as sess:
            space = (await sess.execute(
                select(db.RagSpace).where(db.RagSpace.workspace_path == working_directory)
            )).scalars().first()

            if space is None:
                provider = get_embedding_provider()
                name = working_directory.rstrip("/\\").replace("\\", "/").split("/")[-1] or working_directory
                space = db.RagSpace(
                    id=str(uuid.uuid4()), name=name, description=None,
                    workspace_path=working_directory, embedding_model=provider.model_name,
                    embedding_dim=provider.dimension, created_at=_now(),
                )
                sess.add(space)
                await sess.flush()

            try:
                results = await rag_search_chunks(sess, space.id, query, top_k=_TOP_K)
            except ValueError as e:
                return tool_error(self.name, str(e))

            await sess.commit()

        return RagSearchResult(
            tool=self.name,
            status="success",
            query=query,
            space_name=space.name,
            results=[
                {
                    "source_title": r.source_title,
                    "origin_path": r.origin_path,
                    "text": r.text,
                    "score": r.score,
                }
                for r in results
            ],
        )
