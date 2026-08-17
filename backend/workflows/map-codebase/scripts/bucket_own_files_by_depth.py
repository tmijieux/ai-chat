"""Group per-file summaries by their (possibly clamped) directory and depth, deepest first.

Specific to the map-codebase workflow's own bottom-up directory fold (see workflow.yaml, stage
bucket_by_depth) — not a generic coordinator primitive, so it lives here rather than in the shared
workflow_coordinator.py.

Depth is measured relative to target_directory: a file directly inside a subdirectory of
target_directory is depth 1, a grandchild is depth 2, etc. A file that sits directly inside
target_directory itself (no subdirectory at all) is treated as depth 1 too, grouped under
target_directory's own path — there's nothing shallower than depth 1 for it to belong to, and this
lets it fold into the root summary through the same mechanism as every other directory instead of
being a special case. A directory deeper than max_fold_depth is clamped to its ancestor at that
depth, so very deep chains degrade to a coarser bucket instead of being dropped.

Prints one JSON list to stdout, length == max_fold_depth, ordered DEEPEST FIRST:
    [{"depth": N, "own_groups": [{"directory": str, "files": [...]}, ...]}, ...]
A depth with no files at all still gets an entry with "own_groups": [] — the list length is always
exactly max_fold_depth so fold_levels_loop (an ordinary loop over this list) runs the right number
of iterations regardless of how deep the repo actually goes.

Usage:
    python bucket_own_files_by_depth.py <items_json> <target_directory> <max_fold_depth>
"""
from __future__ import annotations

import json
import sys
from pathlib import PurePosixPath


def main() -> None:
    """Bucket files_loop's aggregated per-file summaries by clamped depth and print the result."""
    if len(sys.argv) != 4:
        sys.exit("usage: bucket_own_files_by_depth.py <items_json> <target_directory> <max_fold_depth>")

    items = json.loads(sys.argv[1])
    target_directory = sys.argv[2]
    max_fold_depth = int(sys.argv[3])

    target_norm = "." if target_directory in ("", ".") else target_directory.rstrip("/")
    target_path = PurePosixPath(target_norm)

    # own_by_directory[directory] -> {"depth": N, "files": [...]}
    own_by_directory: dict[str, dict] = {}

    for entry in items:
        file_info = entry.get("item") or {}
        summary = entry.get("summarize_file") or {}
        path = file_info.get("path")
        if path is None:
            continue

        file_dir = PurePosixPath(path).parent
        if target_norm == ".":
            rel_parts = [] if str(file_dir) == "." else list(file_dir.parts)
        else:
            try:
                rel = file_dir.relative_to(target_path)
                rel_parts = [] if str(rel) == "." else list(rel.parts)
            except ValueError:
                # File isn't actually under target_directory — shouldn't happen since
                # enumerate_files only walks target_directory, but fail open rather than crash.
                rel_parts = list(file_dir.parts)

        actual_depth = len(rel_parts)
        depth = max(1, min(actual_depth, max_fold_depth))

        if actual_depth == 0:
            directory = target_norm
        else:
            clamped_parts = rel_parts[:depth]
            directory = "/".join(clamped_parts) if target_norm == "." else "/".join([target_norm, *clamped_parts])

        bucket = own_by_directory.setdefault(directory, {"depth": depth, "files": []})
        bucket["files"].append({
            "path": path,
            "role": summary.get("role", ""),
            "exports": summary.get("exports", []),
            "key_dependencies": summary.get("key_dependencies", []),
        })

    by_depth: dict[int, list[dict]] = {d: [] for d in range(1, max_fold_depth + 1)}
    for directory, bucket in own_by_directory.items():
        by_depth[bucket["depth"]].append({"directory": directory, "files": bucket["files"]})

    result = [
        {"depth": depth, "own_groups": by_depth[depth]}
        for depth in range(max_fold_depth, 0, -1)
    ]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
