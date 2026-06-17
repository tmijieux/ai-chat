from typing import TypedDict, NotRequired


class ToolResult(TypedDict):
    """Base envelope for every tool result. tool_call_id is added by the framework after execute() returns."""
    tool: str
    status: str
    tool_call_id: NotRequired[str]
    error: NotRequired[dict]
    reason: NotRequired[str]


class ReadFileResult(ToolResult):
    """Result of read_file: the file path and its full text content."""
    path: NotRequired[str]
    file_content: NotRequired[str]


class WriteFileResult(ToolResult):
    """Result of write_file: the path that was written."""
    path: NotRequired[str]


class EditFileResult(ToolResult):
    """Result of edit_file: the path that was edited."""
    path: NotRequired[str]


class ListDirectoryResult(ToolResult):
    """Result of list_directory: the path and a text listing of its entries."""
    path: NotRequired[str]
    content: NotRequired[str]


class GlobFilesResult(ToolResult):
    """Result of glob_files: matched file paths and their count."""
    pattern: NotRequired[str]
    path: NotRequired[str]
    files: NotRequired[list[str]]
    file_count: NotRequired[int]
    result_id: NotRequired[str]


class GrepFilesResult(ToolResult):
    """Result of grep_files: per-file match objects with metadata."""
    pattern: NotRequired[str]
    path: NotRequired[str]
    glob_pattern: NotRequired[str]
    result_id: NotRequired[str]
    matches: NotRequired[list[dict]]
    total: NotRequired[int]
    truncated: NotRequired[bool]


class RunShellResult(ToolResult):
    """Result of run_shell: the command, stdout, stderr, and exit code."""
    command: NotRequired[str]
    output: NotRequired[str]
    stderr: NotRequired[str]
    exit_code: NotRequired[int]


class SearchWebResult(ToolResult):
    """Result of search_web: the query, list of result objects, and total count."""
    query: NotRequired[str]
    results: NotRequired[list[dict]]
    total_results: NotRequired[int]


class ReadFileRangeResult(ToolResult):
    """Result of read_file_range: the file path, line range, and numbered content."""
    file_path: NotRequired[str]
    start_line: NotRequired[int]
    end_line: NotRequired[int]
    content: NotRequired[str]


class ExploreCodebaseResult(ToolResult):
    """Result of explore_codebase: a natural-language summary and matching code snippets."""
    summary: NotRequired[str]
    snippets: NotRequired[list[dict]]


class AskUserQuestionResult(ToolResult):
    """Result of ask_user_question: the user's free-text reply."""
    reply: NotRequired[str]


class ProposePlanResult(ToolResult):
    """Result of propose_plan: either accepted (with chosen_mode) or feedback from the user."""
    feedback: NotRequired[str]
    chosen_mode: NotRequired[str]
    comment: NotRequired[str]


class SummarizeSubtaskResult(ToolResult):
    """Result of summarize_subtask: the generated summary text."""
    summary: NotRequired[str]


class SubagentResult(ToolResult):
    """Result of subagent: the final content produced by the sub-agent."""
    result: NotRequired[str]
