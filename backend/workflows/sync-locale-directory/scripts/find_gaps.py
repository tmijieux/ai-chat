"""Find every locale entry a translation delta needs to cover: missing keys plus untranslated-
looking values (reusing find_missing_source_keys and collect_suspects from the existing audit
scripts — same detection logic, nothing reimplemented here). Writes ONE combined delta file
(source-shaped, ready to feed the translate-locale workflow) into a .locale-sync-tmp/ scratch
directory next to the translated file, and clears/creates a fresh, empty file for the translation
to be appended into.

Prints one JSON object to stdout: {"count", "path", "translated_path"} — path/translated_path are
null when there is nothing to translate. Nothing else goes to stdout/stderr on the success path.

Usage:
    python find_gaps.py <source_path> <translated_path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from find_untranslated_values import collect_suspects
from locale_utils import find_missing_source_keys, load_json_pairs

_UNTRANSLATED_MIN_LENGTH = 3
_MISSING_KEY_THRESHOLD = 0.90


def main() -> None:
    """Union missing-key and untranslated-value gaps for one file pair, write the delta, print JSON."""
    if len(sys.argv) != 3:
        sys.exit("usage: find_gaps.py <source_path> <translated_path>")
    source_path = Path(sys.argv[1])
    translated_path = Path(sys.argv[2])

    source = load_json_pairs(source_path)
    translated = load_json_pairs(translated_path)

    missing = find_missing_source_keys(source, translated, _MISSING_KEY_THRESHOLD)

    source_values = source.as_dict()
    suspects = collect_suspects(translated, source)
    untranslated = [
        (suspect.key, source_values.get(suspect.key, suspect.value))
        for suspect in suspects
        if not suspect.is_short(_UNTRANSLATED_MIN_LENGTH)
    ]

    combined: dict[str, object] = {}
    for key, value in missing:
        combined[key] = value
    for key, value in untranslated:
        combined[key] = value

    scratch_dir = translated_path.parent / ".locale-sync-tmp"
    delta_path = scratch_dir / f"{translated_path.stem}.delta.json"
    translated_out_path = scratch_dir / f"{translated_path.stem}.delta.translated.json"

    if len(combined) == 0:
        if delta_path.exists():
            delta_path.unlink()
        if translated_out_path.exists():
            translated_out_path.unlink()
        print(json.dumps({"count": 0, "path": None, "translated_path": None}))
        return

    scratch_dir.mkdir(parents=True, exist_ok=True)
    delta_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Fresh, empty file — the translate-locale sub-workflow appends to this chunk by chunk, so any
    # stale content left over from a previous run must be gone before it starts.
    translated_out_path.write_text("", encoding="utf-8")

    print(json.dumps({
        "count": len(combined),
        "path": str(delta_path.resolve()),
        "translated_path": str(translated_out_path.resolve()),
    }))


if __name__ == "__main__":
    main()
