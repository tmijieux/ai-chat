from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent.finish_tools import (
    BaseFinishTool,
    FinishAugmentation,
    FinishClassify,
    FinishCritique,
    FinishExplore,
    FinishPlan,
    FinishTask,
    FinishVerify,
)

_BUILTIN_FINISH_TOOL_CLASSES: dict[str, type[BaseFinishTool]] = {
    "finish_classify": FinishClassify,
    "finish_augmentation": FinishAugmentation,
    "finish_critique": FinishCritique,
    "finish_plan": FinishPlan,
    "finish_task": FinishTask,
    "finish_verify": FinishVerify,
    "finish_explore": FinishExplore,
}


@dataclass
class AgentDefinition:
    """A bundled system prompt + tool collection, callable as a tool from workflow stages.

    input_schema maps parameter names to JSON Schema fragments (type, description, etc.).
    The agent takes its inputs as its user message and returns its finish_tool result.
    """

    name: str
    description: str
    system_prompt: str
    tools: list[str]
    finish_tool_name: str
    input_schema: dict[str, dict]
    max_iterations: int | None = None
    inject_turn_reminders: bool = False


@dataclass
class WorkflowStageDefinition:
    """A single step in a workflow. The type field determines which other fields are used.

    Types:
      llm         — one LLM stage loop with tools/agents and a finish tool
      coordinator — deterministic action run by the orchestrator (no LLM)
      branch      — evaluate a condition and jump to a named stage
      loop        — iterate over a list or repeat until exit_condition, with inner stages
      agent       — run the plain agent loop on the full conversation history
    """

    name: str
    type: str = "llm"

    # llm fields
    system_prompt: str = ""
    user_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    finish_tool_name: str = "finish_task"
    max_iterations: int | None = None
    inject_turn_reminders: bool = False

    # coordinator fields
    action: str = ""
    action_input: dict[str, str] = field(default_factory=dict)
    action_output: str = ""

    # shared: skip this stage if condition is false (coordinator + loop inner stages)
    condition: str = ""

    # branch fields
    if_true: str = ""
    if_false: str = ""

    # agent type fields
    message_suffix: str = ""

    # loop fields
    over: str = ""                   # {{slot.field}} expression resolving to a list
    item_var: str = "item"           # slot name for the current loop item
    entry_condition: str = ""        # only enter the loop if this evaluates to true
    exit_condition: str = ""         # exit loop early when this evaluates to true
    on_max_retries: str = "continue" # "continue" or "abort_workflow"
    on_retry: dict[str, str] = field(default_factory=dict)  # slot_name → template expression
    max_retries: int = 3
    loop_output: str = ""            # slot name to store aggregated loop results
    inner_stages: list[WorkflowStageDefinition] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """A parsed workflow loaded from a YAML file."""

    name: str
    description: str
    stages: list[WorkflowStageDefinition]
    directory: Path = field(default_factory=Path)
    mode: str | None = None
    auto_safe_commands: list[str] = field(default_factory=list)
    _finish_tool_classes: dict[str, type[BaseFinishTool]] = field(default_factory=dict, repr=False)

    def make_finish_tool(self, name: str) -> BaseFinishTool:
        """Instantiate a finish tool by name (builtin or inline-defined)."""
        cls = self._finish_tool_classes.get(name)
        if cls is None:
            raise ValueError(f"Unknown finish tool '{name}' — not a builtin and not defined in finish_tools")
        return cls()


def _build_dynamic_finish_tool_class(tool_name: str, spec: dict) -> type[BaseFinishTool]:
    """Create a BaseFinishTool subclass dynamically from an inline YAML finish_tools entry."""
    description = spec.get("description") or ""
    params_spec: dict = spec.get("parameters") or {}
    parameters = {
        "type": "object",
        "properties": dict(params_spec),
        "required": list(params_spec.keys()),
    }
    return type(
        tool_name,
        (BaseFinishTool,),
        {"name": tool_name, "description": description, "parameters": parameters},
    )


def _parse_stage(data: dict, finish_tool_classes: dict[str, type[BaseFinishTool]]) -> WorkflowStageDefinition:
    """Parse one stage dict from YAML into a WorkflowStageDefinition."""
    stage_type = data.get("type") or "llm"
    name = data.get("name") or ""

    if stage_type == "llm":
        finish_tool_name = data.get("finish_tool") or "finish_task"
        if finish_tool_name not in finish_tool_classes:
            raise ValueError(f"Stage '{name}' references unknown finish tool '{finish_tool_name}'")
        return WorkflowStageDefinition(
            name=name,
            type="llm",
            system_prompt=data.get("system_prompt") or "",
            user_prompt=data.get("user_prompt") or "",
            tools=data.get("tools") or [],
            agents=data.get("agents") or [],
            finish_tool_name=finish_tool_name,
            max_iterations=data.get("max_iterations") if data.get("max_iterations") is not None else None,
            inject_turn_reminders=bool(data.get("inject_turn_reminders")),
            condition=data.get("condition") or "",
        )

    if stage_type == "coordinator":
        return WorkflowStageDefinition(
            name=name,
            type="coordinator",
            action=data.get("action") or "",
            condition=data.get("condition") or "",
            action_input=data.get("input") or {},
            action_output=data.get("output") or "",
        )

    if stage_type == "branch":
        return WorkflowStageDefinition(
            name=name,
            type="branch",
            condition=data.get("condition") or "",
            if_true=data.get("if_true") or "",
            if_false=data.get("if_false") or "",
        )

    if stage_type == "respond":
        return WorkflowStageDefinition(
            name=name,
            type="respond",
            message_suffix=data.get("message_suffix") or "",
        )

    if stage_type == "agent":
        return WorkflowStageDefinition(
            name=name,
            type="agent",
            workflow_ref=data.get("ref") or "",
            user_prompt=data.get("user_prompt") or "",
        )

    if stage_type == "loop":
        inner_stages = [_parse_stage(s, finish_tool_classes) for s in (data.get("stages") or [])]
        return WorkflowStageDefinition(
            name=name,
            type="loop",
            over=data.get("over") or "",
            item_var=data.get("item_var") or "item",
            entry_condition=data.get("entry_condition") or "",
            exit_condition=data.get("exit_condition") or "",
            max_retries=data.get("max_retries") or 3,
            on_max_retries=data.get("on_max_retries") or "continue",
            on_retry=data.get("on_retry") or {},
            loop_output=data.get("output") or "",
            inner_stages=inner_stages,
        )

    raise ValueError(f"Stage '{name}': unknown type '{stage_type}'")


def load_workflow(path: Path) -> WorkflowDefinition:
    """Parse a workflow YAML file or directory and return a WorkflowDefinition.

    Accepts either a .yaml file path or a directory containing workflow.yaml.
    The resolved directory is stored in WorkflowDefinition.directory so the
    orchestrator can resolve agent refs and other files relative to it.
    """
    if path.is_dir():
        directory = path
        yaml_file = path / "workflow.yaml"
    else:
        directory = path.parent
        yaml_file = path

    with open(yaml_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    name: str = data.get("name") or path.stem
    description: str = data.get("description") or ""

    finish_tool_classes: dict[str, type[BaseFinishTool]] = dict(_BUILTIN_FINISH_TOOL_CLASSES)
    for tool_name, tool_spec in (data.get("finish_tools") or {}).items():
        finish_tool_classes[tool_name] = _build_dynamic_finish_tool_class(tool_name, tool_spec)

    stages = [_parse_stage(s, finish_tool_classes) for s in (data.get("stages") or [])]

    return WorkflowDefinition(
        name=name,
        description=description,
        stages=stages,
        directory=directory,
        mode=data.get("mode") or None,
        auto_safe_commands=data.get("auto_safe_commands") or [],
        _finish_tool_classes=finish_tool_classes,
    )


def load_agent(path: Path) -> AgentDefinition:
    """Parse an agent YAML file and return an AgentDefinition."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AgentDefinition(
        name=data.get("name") or path.stem,
        description=data.get("description") or "",
        system_prompt=data.get("system_prompt") or "",
        tools=data.get("tools") or [],
        finish_tool_name=data.get("finish_tool") or "finish_task",
        input_schema=data.get("input") or {},
        max_iterations=data.get("max_iterations") if data.get("max_iterations") is not None else None,
        inject_turn_reminders=bool(data.get("inject_turn_reminders")),
    )
