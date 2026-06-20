"""Agent definition file-based CRUD endpoints (backend/agents/*.yaml)."""
from pathlib import Path

from fastapi import APIRouter, HTTPException

import loaders as ld
from conv_helpers import _slugify

router = APIRouter()

_AGENTS_DIR = Path(__file__).parent.parent / "agents"


def _load_agent_file(path: Path) -> dict:
    """Parse an agent YAML file and return its data dict for the API."""
    import yaml as _yaml
    data = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "name": data.get("name") or path.stem,
        "description": data.get("description") or "",
        "system_prompt": data.get("system_prompt") or "",
        "tools": data.get("tools") or [],
        "finish_tool": data.get("finish_tool") or "finish_task",
        "max_iterations": data.get("max_iterations") if data.get("max_iterations") is not None else None,
        "inject_turn_reminders": bool(data.get("inject_turn_reminders")),
    }


def _write_agent_file(path: Path, data: dict) -> None:
    """Write an agent YAML file from the API data dict."""
    import yaml as _yaml
    path.write_text(_yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False), encoding="utf-8")


@router.get("/api/agents")
async def list_agents():
    """List all agent YAML files from backend/agents/."""
    _AGENTS_DIR.mkdir(exist_ok=True)
    return [_load_agent_file(f) for f in sorted(_AGENTS_DIR.glob("*.yaml"))]


@router.post("/api/agents")
async def create_agent(body: ld.NewAgent):
    """Create a new agent YAML file. Filename is derived from the name."""
    _AGENTS_DIR.mkdir(exist_ok=True)
    slug = _slugify(body.name)
    path = _AGENTS_DIR / f"{slug}.yaml"
    if path.exists():
        raise HTTPException(409, detail=f"Agent '{slug}' already exists")
    data = {
        "name": body.name,
        "description": body.description,
        "system_prompt": body.system_prompt,
        "tools": body.tools,
        "finish_tool": body.finish_tool,
        "max_iterations": body.max_iterations,
        "inject_turn_reminders": body.inject_turn_reminders,
    }
    _write_agent_file(path, data)
    return _load_agent_file(path)


@router.put("/api/agents/{name}")
async def update_agent(name: str, body: ld.UpdateAgent):
    """Update an existing agent YAML file."""
    path = _AGENTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(404)
    current = _load_agent_file(path)
    data = {
        "name": body.name if body.name is not None else current["name"],
        "description": body.description if body.description is not None else current["description"],
        "system_prompt": body.system_prompt if body.system_prompt is not None else current["system_prompt"],
        "tools": body.tools if body.tools is not None else current["tools"],
        "finish_tool": body.finish_tool if body.finish_tool is not None else current["finish_tool"],
        "max_iterations": body.max_iterations if body.max_iterations is not None else current["max_iterations"],
        "inject_turn_reminders": body.inject_turn_reminders if body.inject_turn_reminders is not None else current["inject_turn_reminders"],
    }
    _write_agent_file(path, data)
    return _load_agent_file(path)


@router.delete("/api/agents/{name}")
async def delete_agent(name: str):
    """Delete an agent YAML file."""
    path = _AGENTS_DIR / f"{name}.yaml"
    if path.exists():
        path.unlink()
    return ""


@router.get("/api/finish-tools")
async def list_finish_tools():
    """List builtin finish tool names available for agents."""
    from agent.workflow_loader import _BUILTIN_FINISH_TOOL_CLASSES
    return list(_BUILTIN_FINISH_TOOL_CLASSES.keys())
