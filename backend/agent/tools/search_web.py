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
    requires_confirmation = False
    measured_delta = 282

    def validate(self, args: dict) -> str:
        return f"SEARCH: {args.get('query', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        from ddgs import DDGS
        import trafilatura

        query = args.get("query", "")
        max_results = 5
        results = []
        try:
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
            return {
                "tool": self.name,
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results),
            }
        except Exception as e:
            return tool_error(self.name, f"Search error: {e}")
