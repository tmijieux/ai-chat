from __future__ import annotations

import importlib.util
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
class WorkflowStageDefinition:
    """Defines a single stage within a workflow."""

    name: str
    prompt: str
    tools: list[str]
    finish_tool_name: str
    max_iterations: int = 12
    inject_turn_reminders: bool = False


@dataclass
class WorkflowDefinition:
    """A parsed workflow loaded from a YAML file."""

    name: str
    description: str
    stages: list[WorkflowStageDefinition]
    _finish_tool_classes: dict[str, type[BaseFinishTool]] = field(default_factory=dict, repr=False)

    def make_finish_tool(self, name: str) -> BaseFinishTool:
        """Instantiate a finish tool by name (builtin or inline-defined)."""
        cls = self._finish_tool_classes.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown finish tool '{name}' — not a builtin and not defined in finish_tools"
            )
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


def load_workflow(path: Path) -> WorkflowDefinition:
    """Parse a workflow YAML file and return a WorkflowDefinition."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    name: str = data.get("name") or path.stem
    description: str = data.get("description") or ""

    finish_tool_classes: dict[str, type[BaseFinishTool]] = dict(_BUILTIN_FINISH_TOOL_CLASSES)
    for tool_name, tool_spec in (data.get("finish_tools") or {}).items():
        finish_tool_classes[tool_name] = _build_dynamic_finish_tool_class(tool_name, tool_spec)

    stages: list[WorkflowStageDefinition] = []
    for stage_data in (data.get("stages") or []):
        stage_name: str = stage_data.get("name") or ""
        finish_tool_name: str = stage_data.get("finish_tool") or "finish_task"
        if finish_tool_name not in finish_tool_classes:
            raise ValueError(
                f"Stage '{stage_name}' references unknown finish tool '{finish_tool_name}'"
            )
        stages.append(WorkflowStageDefinition(
            name=stage_name,
            prompt=stage_data.get("prompt") or "",
            tools=stage_data.get("tools") or [],
            finish_tool_name=finish_tool_name,
            max_iterations=stage_data.get("max_iterations") or 12,
            inject_turn_reminders=bool(stage_data.get("inject_turn_reminders")),
        ))

    return WorkflowDefinition(
        name=name,
        description=description,
        stages=stages,
        _finish_tool_classes=finish_tool_classes,
    )
