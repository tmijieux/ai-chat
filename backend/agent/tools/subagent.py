import asyncio
from .base import BaseTool, tool_error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentSession

# Read-only, non-confirmation tools the subagent may use.
# search_web is intentionally excluded: it requires_confirmation=True and the
# confirm future would be registered on the sub-session, but the WS client's
# response would resolve on the parent session — no match.
_SUBAGENT_TOOL_NAMES = {"list_directory", "glob_files", "grep_files", "read_file", "summarize_subtask"}


class SubAgentTool(BaseTool):
    name = "subagent"
    description = (
        "Delegate a research or analysis subtask to a sub-agent that runs its own "
        "tool-calling loop. The sub-agent has access to read-only tools: "
        "list_directory, glob_files, grep_files, read_file, summarize_subtask. "
        "Use this to explore a codebase or gather information without expanding the main context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Specific task or question for the sub-agent. Be precise about what to find or analyze.",
            },
        },
        "required": ["task"],
    }
    requires_confirmation = False
    measured_delta = 335

    def label(self, args: dict) -> str:
        return f"SUBAGENT: {args.get('task', '')[:80]}"

    async def execute(self, args: dict, session: "AgentSession", working_directory: str | None) -> dict:
        from agent.agent import AgentSession as _AgentSession, run_agent
        from agent.tools import TOOL_REGISTRY, get_ollama_tool_list

        task = args.get("task", "")
        if not task:
            return tool_error(self.name, "task is required")

        sub_tools = get_ollama_tool_list(list(_SUBAGENT_TOOL_NAMES))
        messages = [{"role": "user", "content": task}]
        sub_session = _AgentSession()

        agent_task = asyncio.create_task(
            run_agent(sub_session, messages, sub_tools, working_directory)
        )

        final_content = ""
        FORWARD = {"thinking", "content", "tool_call_start", "tool_call_chunk", "tool_call", "tool_result", "error"}

        while True:
            event = await sub_session.outbound.get()
            etype = event.get("type")

            if etype == "content":
                final_content += event.get("content", "")

            if etype in FORWARD:
                await session.emit({**event, "_subagent": True})

            if etype in ("done", "error"):
                break

        await agent_task

        if not final_content:
            return tool_error(self.name, "Sub-agent finished without producing a response")

        return {"tool": self.name, "status": "success", "result": final_content}
