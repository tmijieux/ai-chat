from .base import BaseTool
from .list_directory import ListDirectoryTool
from .glob_files import GlobFilesTool
from .grep_files import GrepFilesTool
from .read_file import ReadFileTool
from .write_file import WriteFileTool
from .edit_file import EditFileTool
from .run_shell import RunShellTool
from .search_web import SearchWebTool
from .summarize_subtask import SummarizeSubtaskTool
from .subagent import SubAgentTool
from .read_file_range import ReadFileRangeTool
from .explore_codebase import ExploreCodebaseTool
from .propose_plan import ProposePlanTool
from .ask_user_question import AskUserQuestionTool

TOOL_REGISTRY: dict[str, BaseTool] = {
    t.name: t
    for t in [
        ListDirectoryTool(),
        GlobFilesTool(),
        GrepFilesTool(),
        ReadFileTool(),
        ReadFileRangeTool(),
        ExploreCodebaseTool(),
        WriteFileTool(),
        EditFileTool(),
        RunShellTool(),
        SearchWebTool(),
        SummarizeSubtaskTool(),
        SubAgentTool(),
    ]
}

CONVERSATIONAL_TOOLS: dict[str, BaseTool] = {
    t.name: t for t in [AskUserQuestionTool()]
}

PLAN_MODE_TOOLS: dict[str, BaseTool] = {
    t.name: t for t in [ProposePlanTool()]
}


def get_ollama_tool_list(names: list[str]) -> list[dict]:
    return [
        {"type": "function", "function": TOOL_REGISTRY[n].to_ollama_schema()}
        for n in names
        if n in TOOL_REGISTRY
    ]
