"""Persists a workflow run's full execution history to disk as it streams — see ADR-0011.

The on-disk tree mirrors the *runtime* call structure rather than the engine's flat, non-resetting
execution_id counter (ADR-0009/0010): a loop stage's directory holds one numbered subdirectory per
item, and each item's inner stages nest inside that — recursively, so a loop nested inside another
loop lives inside its parent item's own folder and item numbers can never collide across outer
items. Every stage directory holds `meta.json` (status, stage_type, attempt_count) plus one
`attempt_N.json` per dispatch (numbered from 1, item-local via the engine's own LoopContext), shaped
like the frontend's WorkflowStageState so the read side (backend/routers/workflow_runs.py) can hand
it back with no translation layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKFLOW_RUNS_DIR = Path(__file__).parent.parent / "workflow_runs"


@dataclass
class _InvocationRecord:
    """Accumulated state for one stage invocation (one execution_id) while it streams."""

    path: str
    execution_id: str
    stage_type: str
    invocation_number: int
    item_number: int | None
    item_total: int | None
    attempt_number: int | None
    attempt_total: int | None
    status: str = "running"
    result: Any = None
    duration_ms: int | None = None
    context_tokens: int | None = None
    response_tokens: int = 0
    iteration_count: int = 0
    activity: list[dict] = field(default_factory=list)


class WorkflowRunRecorder:
    """One instance per top-level run, shared by reference into any sub-workflow orchestrator
    (see CustomWorkflowOrchestrator._run_sub_workflow) so the whole call tree persists into the
    same run directory.
    """

    def __init__(self, workflow_name: str, run_id: str, user_message: str = "", nodes: list[dict] | None = None) -> None:
        self.run_id = run_id
        self._root = WORKFLOW_RUNS_DIR / workflow_name / run_id
        self._root.mkdir(parents=True, exist_ok=True)
        self._invocations: dict[str, _InvocationRecord] = {}
        self.last_failed_address: str | None = None
        # Persisted into run.json (see _write_run_meta) so a resumed run can reseed
        # slots["user_message"] from disk without the caller supplying it again.
        self._user_message = user_message
        # The declared stage tree (custom_workflow._serialize_stage_nodes), persisted alongside
        # status so a run reopened from disk (no live workflow_start event) can still draw the
        # whole plan up front, same as the live view does — see routers/workflow_runs.py's status
        # endpoint and workflow-run.service.ts's openPersistedRun.
        self._nodes = nodes if nodes is not None else []
        # First-seen order of each address among its own siblings — directory names sort
        # alphabetically ("chunk_notes_loop" before "chunk_the_file"), which has nothing to do
        # with declared/execution order, so meta.json carries this instead for the read side to
        # sort by (routers/workflow_runs.py). Assigned once per address, on its first write.
        self._sequence_by_address: dict[str, int] = {}
        self._next_sequence_by_parent: dict[str, int] = {}
        self._write_run_meta(status="running")

    @classmethod
    def load(cls, workflow_name: str, run_id: str) -> "WorkflowRunRecorder":
        """Reopen an existing run's directory to resume it.

        Reconstructs `_sequence_by_address`/`_next_sequence_by_parent` from every stage's
        meta.json so redispatched stages keep their original `sequence` and any genuinely new
        siblings still get correctly incrementing ones, and recovers the original `user_message`
        and declared `nodes` tree from run.json so the caller doesn't have to supply them again.
        """
        root = WORKFLOW_RUNS_DIR / workflow_name / run_id
        run_meta = json.loads((root / "run.json").read_text(encoding="utf-8"))
        instance = cls(workflow_name, run_id, user_message=run_meta.get("user_message", ""), nodes=run_meta.get("nodes"))
        for meta_path in root.rglob("meta.json"):
            address = meta_path.parent.relative_to(root).as_posix().split("/")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            key = "/".join(address)
            instance._sequence_by_address[key] = meta["sequence"]
            parent_key = "/".join(address[:-1])
            instance._next_sequence_by_parent[parent_key] = max(
                instance._next_sequence_by_parent.get(parent_key, 0), meta["sequence"] + 1
            )
        return instance

    # ------------------------------------------------------------------
    # Run-level lifecycle
    # ------------------------------------------------------------------

    def mark_done(self) -> None:
        self._write_run_meta(status="done")

    def mark_failed(self) -> None:
        self._write_run_meta(status="failed", failed_path=self.last_failed_address)

    def mark_stopped(self) -> None:
        """User-requested cancellation, distinct from `mark_failed` so a stopped run doesn't look
        like a crash. `last_failed_address` doubles as the resume point here too — see
        `on_stage_exit`, which now tracks it for a "stopped" stage the same way as a failed one.
        """
        self._write_run_meta(status="stopped", failed_path=self.last_failed_address)

    def _write_run_meta(self, status: str, failed_path: str | None = None) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "user_message": self._user_message,
            "nodes": self._nodes,
        }
        if failed_path is not None:
            payload["failed_path"] = failed_path
        (self._root / "run.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Addressing
    # ------------------------------------------------------------------

    def address_for(self, path: str, active_item_stack: list[tuple[str, int]]) -> list[str]:
        """Build the on-disk address for a dotted engine `path`, given the currently-active item
        ancestry (loop_path, item_number) pairs for every loop currently entered, outermost first.

        Interleaves each ancestor loop's item number right after that loop's own path segment,
        e.g. path="files_loop.chunk_the_file" with active_item_stack=[("files_loop", 11)] becomes
        ["files_loop", "11", "chunk_the_file"].
        """
        segments = path.split(".")
        item_by_loop_path = dict(active_item_stack)
        address: list[str] = []
        prefix = ""
        for segment in segments:
            prefix = segment if prefix == "" else f"{prefix}.{segment}"
            address.append(segment)
            if prefix in item_by_loop_path:
                address.append(str(item_by_loop_path[prefix]))
        return address

    def _dir_for(self, address: list[str]) -> Path:
        directory = self._root
        for segment in address:
            directory = directory / segment
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def read_item_result(self, address: list[str], item_number: int) -> dict | None:
        """Read one loop item's persisted item_result.json in full (item_number, item_total,
        success, attempts_used, item_result), or None if this item was never recorded.

        Used by resumability: both `workflow_resume.reconstruct_slots` and the loop's own
        disk-reuse path (`CustomWorkflowOrchestrator._run_loop`) need this to treat an
        already-completed item as done without redispatching it.
        """
        item_result_path = self._root.joinpath(*address, str(item_number), "item_result.json")
        if not item_result_path.exists():
            return None
        return json.loads(item_result_path.read_text(encoding="utf-8"))

    def _sequence_for(self, address: list[str]) -> int:
        key = "/".join(address)
        if key in self._sequence_by_address:
            return self._sequence_by_address[key]
        parent_key = "/".join(address[:-1])
        sequence = self._next_sequence_by_parent.get(parent_key, 0)
        self._next_sequence_by_parent[parent_key] = sequence + 1
        self._sequence_by_address[key] = sequence
        return sequence

    # ------------------------------------------------------------------
    # Event handlers — called from the recording session proxy in custom_workflow.py
    # ------------------------------------------------------------------

    def on_stage_enter(self, event: dict, active_item_stack: list[tuple[str, int]]) -> None:
        record = _InvocationRecord(
            path=event["path"],
            execution_id=event["execution_id"],
            stage_type=event["stage_type"],
            invocation_number=event["invocation_number"],
            item_number=event.get("item_number"),
            item_total=event.get("item_total"),
            attempt_number=event.get("attempt_number"),
            attempt_total=event.get("attempt_total"),
        )
        self._invocations[event["execution_id"]] = record
        address = self.address_for(event["path"], active_item_stack)
        directory = self._dir_for(address)
        self._write_meta(directory, address, record)

    def on_stage_exit(self, event: dict, active_item_stack: list[tuple[str, int]]) -> None:
        record = self._invocations.pop(event["execution_id"], None)
        if record is None:
            return
        record.status = event["status"]
        record.result = event["result"]
        record.duration_ms = event["duration_ms"]
        address = self.address_for(event["path"], active_item_stack)
        directory = self._dir_for(address)
        self._write_meta(directory, address, record)
        self._write_attempt(directory, record)
        if record.status in ("failed", "stopped"):
            self.last_failed_address = "/".join(address)

    def on_loop_item_exit(self, event: dict, active_item_stack: list[tuple[str, int]]) -> None:
        address = self.address_for(event["path"], active_item_stack) + [str(event["item_number"])]
        directory = self._dir_for(address)
        payload = {
            "item_number": event["item_number"],
            "item_total": event["item_total"],
            "success": event["success"],
            "status": event["status"],
            "attempts_used": event["attempts_used"],
            "item_result": event["item_result"],
        }
        (directory / "item_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    def on_activity(self, execution_id: str, event: dict) -> None:
        record = self._invocations.get(execution_id)
        if record is None:
            return
        _fold_activity(record, event)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_meta(self, directory: Path, address: list[str], record: _InvocationRecord) -> None:
        attempt_number = record.attempt_number if record.attempt_number is not None else 1
        meta = {
            "path": record.path,
            "stage_type": record.stage_type,
            "status": record.status,
            "attempt_count": attempt_number,
            "attempt_total": record.attempt_total,
            "sequence": self._sequence_for(address),
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_attempt(self, directory: Path, record: _InvocationRecord) -> None:
        attempt_number = record.attempt_number if record.attempt_number is not None else 1
        payload = {
            "execution_id": record.execution_id,
            "invocation_number": record.invocation_number,
            "status": record.status,
            "stage_type": record.stage_type,
            "item_number": record.item_number,
            "item_total": record.item_total,
            "attempt_number": record.attempt_number,
            "attempt_total": record.attempt_total,
            "iteration_count": record.iteration_count,
            "context_tokens": record.context_tokens,
            "response_tokens": record.response_tokens,
            "result": record.result,
            "duration_ms": record.duration_ms,
            "activity": record.activity,
        }
        (directory / f"attempt_{attempt_number}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


def _fold_activity(record: _InvocationRecord, event: dict) -> None:
    """Fold one streamed event into `record.activity`, mirroring the frontend's own grouping in
    workflow-run.service.ts's _onStageActivity — kept in lockstep so a persisted attempt_N.json is
    structurally identical to what the live view already builds.
    """
    event_type = event.get("type")

    if event_type == "iteration_end":
        record.iteration_count += 1
        record.context_tokens = event.get("prompt_tokens")
        record.response_tokens += event.get("response_tokens", 0)
        return

    if event_type in ("thinking", "content"):
        text = event.get("content")
        if text is None or text == "":
            return
        last = record.activity[-1] if len(record.activity) > 0 else None
        if last is not None and last.get("kind") == event_type:
            last["text"] += text
        else:
            record.activity.append({"kind": event_type, "text": text})
        return

    if event_type == "tool_call_start":
        tool_id = event.get("tool_id")
        entry = {"kind": "tool_call", "tool_id": tool_id, "tool_name": event.get("tool_name"), "args_text": ""}
        for index, existing in enumerate(record.activity):
            if existing.get("kind") == "tool_call" and existing.get("tool_id") == tool_id:
                record.activity[index] = entry
                return
        record.activity.append(entry)
        return

    if event_type == "tool_call_raw":
        if len(record.activity) == 0 or record.activity[-1].get("kind") != "tool_call":
            return
        record.activity[-1]["args_text"] += event.get("fragment", "")
        return

    if event_type == "tool_call_chunk":
        tool_id = event.get("tool_id")
        chunk = event.get("chunk", "")
        for entry in record.activity:
            if entry.get("kind") == "tool_call" and entry.get("tool_id") == tool_id:
                entry["args_text"] += chunk
        return

    if event_type == "tool_result":
        record.activity.append({
            "kind": "tool_result",
            "tool_id": event.get("tool_id"),
            "tool_name": event.get("tool_name") or "",
            "log_message": event.get("log_message"),
            "content": event.get("content") or "",
        })
        return

    if event_type == "error":
        record.activity.append({"kind": "error", "message": event.get("message")})
