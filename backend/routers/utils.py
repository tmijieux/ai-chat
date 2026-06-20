"""Utility endpoints: fuzzy file search, directory browsing, workflow listing, and agent tools list."""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from agent.tools import TOOL_REGISTRY, PLAN_MODE_TOOLS, CONVERSATIONAL_TOOLS
from agent.tools.base import TOOL_FRAMEWORK_OVERHEAD, STACKING_OVERHEAD_PER_ADDITIONAL_TOOL

router = APIRouter()


# ---------------------------------------------------------------------------
# Fuzzy file search helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(query: str, text: str) -> bool:
    """Returns True if every character of query appears in text in order. Requires at least 2 characters."""
    if len(query) < 2:
        return False
    it = iter(text)
    return all(c in it for c in query)


def _fuzzy_score(query: str, relative_path: str) -> tuple[int, int]:
    """Lower score = better match. Prefers matches in the filename, then tighter character spans."""
    filename = relative_path.split("/")[-1]
    filename_match = 0 if _fuzzy_match(query, filename) else 1
    positions = []
    idx = 0
    for c in query:
        while idx < len(relative_path) and relative_path[idx] != c:
            idx += 1
        positions.append(idx)
        idx += 1
    span = positions[-1] - positions[0] if len(positions) > 1 else 0
    return (filename_match, span)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/utils/search-files")
async def search_files(workspace: str, query: str = ""):
    """Recursively search for files in a workspace directory by filename. Skips common ignored directories."""
    workspace_path = Path(workspace).resolve()
    if not workspace_path.is_dir():
        raise HTTPException(400, detail=f"Not a directory: {workspace_path}")
    SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", ".angular", ".next", ".cache"}
    query_lower = query.lower()
    results = []
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        for filename in files:
            if filename.startswith("."):
                continue
            abs_path = Path(root) / filename
            relative_path = "/".join(abs_path.relative_to(workspace_path).parts)
            if query_lower and not _fuzzy_match(query_lower, relative_path.lower()):
                continue
            results.append({"name": filename, "path": str(abs_path), "relative_path": relative_path})
            if len(results) >= 50:
                break
        if len(results) >= 50:
            break
    results.sort(key=lambda r: r["relative_path"].lower())
    if query_lower:
        results.sort(key=lambda r: _fuzzy_score(query_lower, r["relative_path"].lower()))
    return {"results": results}


@router.get("/api/utils/browse-directory")
async def browse_directory(path: str | None = None):
    """List immediate subdirectories of a path, or cwd if not given."""
    current = Path(path).resolve() if path else Path.cwd()
    try:
        entries = sorted(
            [
                {"name": e.name, "path": str(e)}
                for e in current.iterdir()
                if e.is_dir() and not e.name.startswith(".")
            ],
            key=lambda e: e["name"].lower(),
        )
    except PermissionError:
        raise HTTPException(403, detail=f"Permission denied: {current}")
    parent = str(current.parent) if current != current.parent else None
    return {"path": str(current), "parent": parent, "entries": entries}


@router.get("/api/workflows")
async def list_workflows():
    """List available workflow definitions from the backend/workflows/ directory."""
    import yaml
    workflows_dir = Path(__file__).parent.parent / "workflows"
    if not workflows_dir.exists():
        return []
    results = []
    candidates: list[Path] = sorted(workflows_dir.glob("*.yaml"))
    candidates += sorted(
        sub / "workflow.yaml"
        for sub in workflows_dir.iterdir()
        if sub.is_dir() and (sub / "workflow.yaml").exists()
    )
    for yaml_file in candidates:
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            results.append({
                "name": data.get("name", yaml_file.parent.stem if yaml_file.name == "workflow.yaml" else yaml_file.stem),
                "description": data.get("description", ""),
            })
        except Exception:
            pass
    return results


@router.get("/api/agent/tools")
async def list_agent_tools():
    """List all available agent tools with their token costs."""
    always_active = [
        {
            "name": t.name,
            "description": t.description,
            "token_count": t.token_count,
            "mode_context": mode_context,
        }
        for t, mode_context in [
            (CONVERSATIONAL_TOOLS["ask_user_question"], "Included in Standard and Plan modes"),
            (PLAN_MODE_TOOLS["propose_plan"], "Included in Plan mode only"),
        ]
    ]
    return {
        "framework_overhead": TOOL_FRAMEWORK_OVERHEAD,
        "stacking_overhead_per_additional_tool": STACKING_OVERHEAD_PER_ADDITIONAL_TOOL,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "requires_confirmation": t.requires_confirmation,
                "token_count": t.token_count,
            }
            for t in TOOL_REGISTRY.values()
        ],
        "always_active_tools": always_active,
    }
