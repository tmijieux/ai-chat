"""One-time migration: rewrite item_result.json files under a workflow-runs directory from the
old flat {item, success, <stage_name>: ...} shape to the new {input, success, result: {...}}
shape (see custom_workflow.py's _collect_inner_results). Idempotent - a file already carrying
"input" is left untouched, so a directory mixing old-format and new-format items (e.g. a run
resumed after this migration landed) is safe to re-run.

Prints only aggregate counts, never file paths or record content.

Usage:
    python migrate_item_result_format.py <root>            # dry run, prints counts only
    python migrate_item_result_format.py <root> --apply    # actually rewrite files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _migrate_one(payload: dict) -> tuple[dict, bool]:
    """Return (possibly-migrated payload, True if migration was applied).

    payload is the on-disk item_result.json top-level dict: {item_number, item_total, success,
    status, attempts_used, item_result}. Only the inner item_result field's shape changed.
    """
    inner = payload.get("item_result")
    if not isinstance(inner, dict):
        return payload, False
    if "input" in inner:
        return payload, False  # already migrated
    if "item" not in inner:
        return payload, False  # unrecognized shape, leave alone

    migrated_inner = {
        "input": inner.get("item"),
        "success": inner.get("success"),
        "result": {k: v for k, v in inner.items() if k not in ("item", "success")},
    }
    return {**payload, "item_result": migrated_inner}, True


def migrate_directory(root: Path, apply: bool) -> dict[str, int]:
    """Walk root for item_result.json files and migrate old-format ones in place.

    apply=False (default) is a dry run: counts what would change, writes nothing.
    """
    counts = {"scanned": 0, "already_new": 0, "migrated": 0, "unrecognized": 0, "errors": 0}
    for path in root.rglob("item_result.json"):
        counts["scanned"] += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counts["errors"] += 1
            continue

        new_payload, changed = _migrate_one(payload)
        if not changed:
            inner = payload.get("item_result")
            if isinstance(inner, dict) and "input" in inner:
                counts["already_new"] += 1
            else:
                counts["unrecognized"] += 1
            continue

        counts["migrated"] += 1
        if apply:
            try:
                path.write_text(
                    json.dumps(new_payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                counts["errors"] += 1
                counts["migrated"] -= 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Directory to scan (e.g. backend/workflow_runs)")
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry run)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    counts = migrate_directory(root, apply=args.apply)
    mode = "APPLY" if args.apply else "DRY RUN"
    print(
        f"[{mode}] scanned={counts['scanned']} already_new={counts['already_new']} "
        f"migrated={counts['migrated']} unrecognized={counts['unrecognized']} errors={counts['errors']}"
    )


if __name__ == "__main__":
    main()
