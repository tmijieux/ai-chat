"""Merge a translated delta file into a main locale file: update existing keys in place, append
new keys at the end, preserving the main file's existing key order otherwise. Makes a .bak backup
of the main file before writing.

Dry-run is the default: nothing is written unless --apply is passed. Prints one JSON summary line
to stdout either way: {"updated", "added", "path"}.

Usage:
    python merge_locale_entries.py <main_path> <delta_path>
    python merge_locale_entries.py <main_path> <delta_path> --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from locale_utils import load_json_pairs


def main() -> None:
    """Parse arguments, merge delta pairs into the main file's pairs, report, and optionally apply."""
    parser = argparse.ArgumentParser(
        description="Merge a translated delta file into a main locale file.",
    )
    parser.add_argument("main_path", type=Path, help="Locale file to merge into.")
    parser.add_argument("delta_path", type=Path, help="Translated delta file to merge from.")
    parser.add_argument(
        "--apply", action="store_true", help="Write the merge. Without this flag, only reports (dry run)."
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="Indentation used when rewriting the JSON. Default: 2"
    )
    args = parser.parse_args()

    main_file = load_json_pairs(args.main_path)
    delta_file = load_json_pairs(args.delta_path)

    result_pairs = list(main_file.pairs)
    key_to_index = {key: index for index, (key, _) in enumerate(result_pairs)}
    updated = 0
    added = 0
    for key, value in delta_file.pairs:
        if key in key_to_index:
            result_pairs[key_to_index[key]] = (key, value)
            updated += 1
        else:
            key_to_index[key] = len(result_pairs)
            result_pairs.append((key, value))
            added += 1

    print(json.dumps({"updated": updated, "added": added, "path": str(args.main_path)}))

    if not args.apply:
        return

    backup_path = args.main_path.with_suffix(args.main_path.suffix + ".bak")
    shutil.copy2(args.main_path, backup_path)

    payload = json.dumps(dict(result_pairs), ensure_ascii=False, indent=args.indent)
    args.main_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
