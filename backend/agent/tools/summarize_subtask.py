import aiohttp
from .base import BaseTool, tool_error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"


class SummarizeSubtaskTool(BaseTool):
    name = "summarize_subtask"
    description = "Summarize a large piece of content relative to a specific task using a fresh LLM call. Use this to compress large tool outputs (file reads, search results) that would overflow context."
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What you need to know from the content (guides what to keep).",
            },
            "content": {
                "type": "string",
                "description": "The large content to summarize.",
            },
        },
        "required": ["task", "content"],
    }
    requires_confirmation = False

    def validate(self, args: dict) -> str:
        return f"SUMMARIZE for: {args.get('task', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        task = args.get("task", "")
        content = args.get("content", "")

        if not task or not content:
            return tool_error(self.name, "Both 'task' and 'content' are required")

        prompt = f"Task: {task}\n\nContent:\n{content}\n\nProvide a concise summary focused on the task above."
        try:
            async with aiohttp.ClientSession() as http:
                async with http.post(
                    OLLAMA_CHAT_URL,
                    json={
                        "model": MODEL_NAME,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                ) as response:
                    result = await response.json()
                    summary = result.get("message", {}).get("content", "")
                    return {"tool": self.name, "status": "success", "summary": summary}
        except Exception as e:
            return tool_error(self.name, f"Summarization error: {e}")
