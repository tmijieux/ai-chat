"""Merge one depth level's own files with the previous (deeper) loop iteration's synthesized
child-directory summaries, unioned by directory.

Specific to the map-codebase workflow's own bottom-up directory fold (see workflow.yaml, stage
merge_level inside fold_levels_loop) — not a generic coordinator primitive, so it lives here rather
than in the shared workflow_coordinator.py.

A directory that has both its own files and a subdirectory (e.g. backend/agent has pipeline.py
directly in it AND a backend/agent/tools subdirectory) needs both inputs merged into the SAME
entry so synthesize_directory can describe both in one summary — that's the whole reason this is a
separate merge step rather than just chaining own_groups and child_summaries independently.

Prints one JSON list to stdout:
    [{"directory": str, "own_files": [...], "child_summaries": [...]}, ...]
one entry per unique directory referenced by either input — a directory with only files, only
child subdirectories, or both, all produce exactly one merged entry.

Usage:
    python merge_fold_level.py <own_groups_json> <child_summaries_json_or_null>
"""
from __future__ import annotations

import json
import sys
from pathlib import PurePosixPath


def main() -> None:
    """Union-merge this level's own file groups with the previous iteration's child summaries."""
    if len(sys.argv) != 3:
        sys.exit("usage: merge_fold_level.py <own_groups_json> <child_summaries_json_or_null>")

    own_groups = json.loads(sys.argv[1]) or []
    child_items = json.loads(sys.argv[2]) or []

    merged: dict[str, dict] = {}

    for group in own_groups:
        directory = group.get("directory")
        if directory is None:
            continue
        entry = merged.setdefault(directory, {"directory": directory, "own_files": [], "child_summaries": []})
        entry["own_files"].extend(group.get("files") or [])

    for child in child_items:
        child_item = child.get("item") or {}
        child_directory = child_item.get("directory")
        synthesis = child.get("synthesize_directory") or {}
        if child_directory is None:
            continue
        parent_directory = str(PurePosixPath(child_directory).parent)
        entry = merged.setdefault(parent_directory, {"directory": parent_directory, "own_files": [], "child_summaries": []})
        entry["child_summaries"].append({
            "directory": child_directory,
            "purpose": synthesis.get("purpose", ""),
            "key_files": synthesis.get("key_files", []),
            "notable_dependencies": synthesis.get("notable_dependencies", []),
        })

    print(json.dumps(list(merged.values()), ensure_ascii=False))


if __name__ == "__main__":
    main()
