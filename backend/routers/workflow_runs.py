"""Read-only access to persisted workflow run history (ADR-0011).

Every workflow run writes its full execution tree to backend/workflow_runs/<workflow_name>/<run_id>/
as it happens (see agent/workflow_run_recorder.py). This router lets the run view fetch any node in
that tree on demand, one level at a time — the node's own recorded content plus a shallow list of
its direct children's status — rather than only ever showing whatever the frontend still happens to
have buffered live.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from agent.workflow_run_recorder import WORKFLOW_RUNS_DIR

router = APIRouter()


def _read_json(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def _resolve_node_dir(workflow_name: str, run_id: str, path: str) -> Path:
    """Resolve `path` (a "/"-separated on-disk address, as written by WorkflowRunRecorder — not
    the engine's dotted stage path, since it interleaves item numbers) to a directory under this
    run's root, rejecting anything that would escape it.
    """
    run_dir = WORKFLOW_RUNS_DIR / workflow_name / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, detail=f"Unknown workflow run: {workflow_name}/{run_id}")

    node_dir = run_dir
    if path != "":
        for segment in path.split("/"):
            if segment in ("", ".", "..") or "\\" in segment:
                raise HTTPException(400, detail=f"Invalid path segment: {segment!r}")
            node_dir = node_dir / segment

    node_dir = node_dir.resolve()
    if not node_dir.is_relative_to(run_dir.resolve()):
        raise HTTPException(400, detail="Invalid path")
    if not node_dir.is_dir():
        raise HTTPException(404, detail=f"Unknown node: {path!r}")
    return node_dir


@router.get("/api/workflow-runs/{workflow_name}/{run_id}/node")
async def get_workflow_run_node(workflow_name: str, run_id: str, path: str = ""):
    """Return one node's own persisted content (meta, every attempt, item result if it's a loop
    item) plus a shallow list of its direct children — one level only, matching how the run view
    drills down one click at a time.
    """
    node_dir = _resolve_node_dir(workflow_name, run_id, path)

    attempts = [
        _read_json(attempt_file)
        for attempt_file in sorted(node_dir.glob("attempt_*.json"), key=lambda p: p.name)
    ]

    # Directory listing order is alphabetical and has nothing to do with declared/execution order
    # ("chunk_notes_loop" sorts before "chunk_the_file" but runs after it) — sequence (stage
    # children) or item number (loop-item children) is the real order; see
    # WorkflowRunRecorder._sequence_for. entries: (sort_key, child_dict).
    entries: list[tuple[int, dict]] = []
    for child in node_dir.iterdir():
        if not child.is_dir():
            continue
        child_meta = _read_json(child / "meta.json")
        if child_meta is not None:
            entries.append((child_meta.get("sequence", 0), {
                "segment": child.name,
                "stage_type": child_meta.get("stage_type"),
                "status": child_meta.get("status"),
            }))
            continue
        # A child with no meta.json is a loop-item-number directory (e.g. "11"), not a stage —
        # its own status comes from item_result.json (written once, on loop_item_exit) rather
        # than meta.json, which only stage directories have. Sorted numerically by item number,
        # not the directory name string ("10" sorts before "2" as text).
        child_item_result = _read_json(child / "item_result.json")
        if child_item_result is None:
            child_status = None
        else:
            # "status" distinguishes a stopped item from a genuinely failed one (see
            # WorkflowRunRecorder.on_loop_item_exit); a run persisted before that field existed
            # falls back to the plain success/fail derivation.
            child_status = child_item_result.get("status")
            if child_status is None:
                child_status = "done" if child_item_result.get("success") else "failed"
        entries.append((int(child.name), {
            "segment": child.name,
            "stage_type": "loop_item",
            "status": child_status,
        }))
    entries.sort(key=lambda entry: entry[0])
    children = [child for _sort_key, child in entries]

    return {
        "path": path,
        "meta": _read_json(node_dir / "meta.json"),
        "attempts": attempts,
        "item_result": _read_json(node_dir / "item_result.json"),
        "children": children,
    }


@router.get("/api/workflow-runs/{workflow_name}/{run_id}")
async def get_workflow_run_status(workflow_name: str, run_id: str):
    """Return the run's own top-level status (running/done/failed, and which node failed if any)."""
    run_dir = WORKFLOW_RUNS_DIR / workflow_name / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, detail=f"Unknown workflow run: {workflow_name}/{run_id}")
    return _read_json(run_dir / "run.json")
