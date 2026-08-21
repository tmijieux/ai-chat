"""Pure, read-only reconstruction of a resumed workflow run's state from its persisted tree.

No LLM calls, no re-execution, no writes — everything here is disk reads plus the same slot-write
rules `CustomWorkflowOrchestrator` itself applies live (see ADR-0011's "Deferred: resumability").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.workflow_loader import WorkflowDefinition, WorkflowStageDefinition


def _read_json(path: Path) -> dict | None:
    """Read and parse one JSON file, or None if it doesn't exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_slots(
    run_root: Path, workflow: WorkflowDefinition, resume_address: list[str]
) -> dict[str, Any]:
    """Rebuild the slot registry a live run would have accumulated by `resume_address`.

    Walks stage-instance directories in `meta.json`'s `sequence` order, not YAML declaration
    order — `sequence` already reflects wherever a `branch` stage actually jumped at runtime, so
    replay never needs to re-evaluate branch conditions or reason about jump targets. Stops at the
    stage/item named by `resume_address`; everything from there on is left for live (re)execution.
    """
    run_meta = _read_json(run_root / "run.json") or {}
    slots: dict[str, Any] = {"user_message": run_meta.get("user_message", "")}
    _walk_stages(run_root, workflow.stages, resume_address, slots)
    return slots


def next_address_after(run_root: Path, address: list[str]) -> list[str] | None:
    """Given a completed node's on-disk address, find the address of whatever ran next.

    Next sibling by `sequence` (stage siblings) or item number (loop-item siblings), bubbling up
    to the enclosing level's own "next" once `address` is the last of its siblings. `None` once
    there's nothing left in the whole run. Sibling kind is read straight off the filesystem, not
    the workflow definition: a stage-instance directory always has its own `meta.json`; a loop
    item directory is a bare numbered directory holding `item_result.json`, no `meta.json`.
    """
    if len(address) == 0:
        return None
    parent_dir = run_root.joinpath(*address[:-1])
    this_name = address[-1]
    is_item_level = this_name.isdigit() and not (parent_dir / this_name / "meta.json").exists()

    if is_item_level:
        item_numbers = sorted(
            int(child.name) for child in parent_dir.iterdir() if child.is_dir() and child.name.isdigit()
        )
        index = item_numbers.index(int(this_name))
        if index + 1 < len(item_numbers):
            return address[:-1] + [str(item_numbers[index + 1])]
    else:
        siblings = sorted(
            (_read_json(meta_path)["sequence"], meta_path.parent.name)
            for meta_path in parent_dir.glob("*/meta.json")
        )
        names = [name for _, name in siblings]
        index = names.index(this_name)
        if index + 1 < len(names):
            return address[:-1] + [names[index + 1]]

    return next_address_after(run_root, address[:-1])


def _status_at(run_root: Path, address: list[str]) -> str | None:
    """Read one node's own persisted status — `meta.json` for a stage, `item_result.json` for a
    loop item (same distinction `next_address_after` makes) — or `None` if nothing is recorded
    there at all.
    """
    node_dir = run_root.joinpath(*address)
    meta = _read_json(node_dir / "meta.json")
    if meta is not None:
        return meta.get("status")
    item_result = _read_json(node_dir / "item_result.json")
    if item_result is not None:
        status = item_result.get("status")
        if status is not None:
            return status
        return "done" if item_result.get("success") else "failed"
    return None


def resolve_resume_address(run_root: Path, selected_address: list[str]) -> list[str] | None:
    """Translate a user-selected node's address into where dispatch should actually resume,
    based on that node's own persisted status (see ADR-0011's "Deferred: resumability" and the
    resumable-workflow-runs plan's Phase 4): a `failed`/`stopped` node — nothing trustworthy
    persisted there — is redone in place (its own address, unchanged); a `done`/`skipped` node
    resumes with whatever ran after it (`next_address_after`). Returns `None` if there's nothing
    left to resume (the selected node was the last thing in the whole run and it's done).
    """
    status = _status_at(run_root, selected_address)
    if status in ("failed", "stopped"):
        return selected_address
    return next_address_after(run_root, selected_address)


def _walk_stages(
    directory: Path,
    stage_defs: list[WorkflowStageDefinition],
    resume_address: list[str],
    slots: dict[str, Any],
) -> None:
    """Replay slot writes for every stage instance under `directory` that fully completed before
    `resume_address`, in `meta.json` sequence order.

    Stops without processing the stage matching `resume_address[0]` — that one is about to be
    (re)dispatched live. If `resume_address` has further segments, recurses into it first (only
    meaningful when that stage is a loop, since only a loop nests an item-number segment).
    """
    stage_by_name = {stage.name: stage for stage in stage_defs}
    entries: list[tuple[int, str, dict]] = []
    for meta_path in directory.glob("*/meta.json"):
        meta = _read_json(meta_path)
        entries.append((meta["sequence"], meta_path.parent.name, meta))
    entries.sort(key=lambda entry: entry[0])

    target_name = resume_address[0] if len(resume_address) > 0 else None
    for _, name, meta in entries:
        if name == target_name:
            if len(resume_address) > 1:
                stage = stage_by_name[name]
                _walk_loop_items(directory / name, stage, resume_address[1:], slots)
            return
        stage = stage_by_name.get(name)
        if stage is not None:
            _replay_stage(directory / name, stage, meta, slots)


def _walk_loop_items(
    loop_dir: Path,
    stage: WorkflowStageDefinition,
    resume_address: list[str],
    slots: dict[str, Any],
) -> None:
    """Replay a loop stage's aggregate output slot from its persisted `item_result.json` files.

    With an empty `resume_address`, replays every item (this whole loop is before the resume
    point). With a non-empty `resume_address`, replays items strictly before `resume_address[0]`
    and, if `resume_address` has more segments, recurses into that one item's own inner stages.
    """
    target_item_number = int(resume_address[0]) if len(resume_address) > 0 else None
    item_numbers = sorted(
        int(child.name) for child in loop_dir.iterdir() if child.is_dir() and child.name.isdigit()
    )
    aggregated: list[dict] = []
    for item_number in item_numbers:
        if item_number == target_item_number:
            if len(resume_address) > 1:
                item_payload = _read_json(loop_dir / str(item_number) / "item_result.json")
                if item_payload is not None:
                    slots[stage.item_var] = item_payload["item_result"]["item"]
                _walk_stages(loop_dir / str(item_number), stage.inner_stages, resume_address[1:], slots)
            break
        if target_item_number is not None and item_number > target_item_number:
            break
        item_payload = _read_json(loop_dir / str(item_number) / "item_result.json")
        if item_payload is not None:
            aggregated.append(item_payload["item_result"])

    # The live engine only ever writes slots[loop_output] once every item has run (see
    # custom_workflow.py's _run_loop, after its item loop). Reconstruction must match that: write
    # the aggregate only when this whole loop was replayed in full (target_item_number is None) —
    # a loop that's the resume target itself (or contains it) has not finished, live or replayed.
    if target_item_number is None and stage.loop_output != "" and stage.over != "":
        from agent.custom_workflow import _format_task_summary  # local: avoids a module-level import cycle
        slots[stage.loop_output] = {"items": aggregated, "task_summary": _format_task_summary(aggregated)}


def _replay_stage(
    stage_dir: Path, stage: WorkflowStageDefinition, meta: dict, slots: dict[str, Any]
) -> None:
    """Replay one fully-completed stage instance's slot write, matching custom_workflow.py's own
    write rule for that stage type exactly.
    """
    if stage.type == "loop":
        _walk_loop_items(stage_dir, stage, [], slots)
        return
    if stage.type in ("branch", "respond"):
        return

    attempt = _read_json(stage_dir / f"attempt_{meta['attempt_count']}.json")
    if attempt is None:
        return
    result = attempt["result"]
    if stage.type == "coordinator":
        if stage.action_output != "":
            slots[stage.action_output] = result
        return
    slots[stage.name] = result
