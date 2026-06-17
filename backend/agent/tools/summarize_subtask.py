from .base import BaseTool, tool_error
from tool_result_types import SummarizeSubtaskResult, ToolResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession


class SummarizeSubtaskTool(BaseTool):
    name = "summarize_subtask"
    description = "A subagent that summarize a large piece of content relative to a specific task using a fresh LLM call. Use this to compress large tool outputs (file reads, search results) that would overflow context."
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
    measured_delta = 337

    def label(self, args: dict) -> str:
        return f"SUMMARIZE for: {args.get('task', '')}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> ToolResult:
        from llm import backend

        task = args.get("task", "")
        content = args.get("content", "")

        if not task or not content:
            return tool_error(self.name, "Both 'task' and 'content' are required")

        messages = [
            {"role": "system", "content": f"Task: {task}\nProvide a concise summary of the user message focused on the task above."},
            {"role": "user", "content": content},
        ]
        try:
            summary = ""
            async for event in backend.stream_completion(messages, [], temperature=0.1):
                if event["type"] == "content":
                    summary += event["content"]
            return SummarizeSubtaskResult(
                tool=self.name,
                status="success",
                summary=summary,
            )
        except Exception as e:
            return tool_error(self.name, f"Summarization error: {e}")
