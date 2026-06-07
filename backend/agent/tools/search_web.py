import asyncio
from .base import BaseTool, tool_error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class SearchWebTool(BaseTool):
    name = "search_web"
    description = "Search DuckDuckGo and extract page content. Use for external documentation, error messages, or API references."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
        },
        "required": ["query"],
    }
    requires_confirmation = True
    measured_delta = 282

    def make_validation_text_for_user_confirmation(self, args: dict) -> str:
        return f"SEARCH: {args.get('query', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        from ddgs import DDGS
        import trafilatura

        query = args.get("query", "")
        max_results = 5

        preview = self.make_validation_text_for_user_confirmation(args)
        approved, user_msg = await session.request_confirm(f"search-{preview}", self.name, args, preview)
        if not approved:
            return tool_error(self.name, "User aborted the search", user_message=user_msg)

        def _do_search() -> list:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    url = r["href"]
                    try:
                        downloaded = trafilatura.fetch_url(url)
                        content = trafilatura.extract(downloaded)
                    except Exception:
                        content = None
                    results.append({
                        "title": r.get("title"),
                        "url": url,
                        "snippet": r.get("body"),
                        "content": content,
                    })
            return results

        try:
            results = await asyncio.to_thread(_do_search)
            return {
                "tool": self.name,
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results),
            }
        except Exception as e:
            return tool_error(self.name, f"Search error: {e}")
