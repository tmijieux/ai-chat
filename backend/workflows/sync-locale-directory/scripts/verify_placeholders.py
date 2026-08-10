"""Find translated entries that broke a placeholder or leaked a key's context annotation.

Two failure modes show up after a translate-locale run, even though the model is told to leave
placeholders alone and never translate a key's bracketed context hint:

  placeholder mismatch — the value contains a `{tag}` / `%s` / `%d` / `%1$s` / HTML tag that never
                          appeared in the key. The model rewrote or invented a substitution marker,
                          which breaks the app code that fills it in. A value MAY omit tags the key
                          has (some languages don't need every placeholder), it just may never add
                          or rename one.
  context leaked        — the key carries a `[ctx: ...]` prefix (disambiguation for the model and
                          the app, e.g. "[ctx: short form of September] Sept": "Sept"), and that
                          bracketed text — or any `[ctx: ...]`-shaped text at all — shows up in the
                          translated value, where it must never appear.

Writes the offending entries as ONE source-shaped JSON file (values taken from the delta's source
file, so retranslation starts from the original source text rather than the broken output) into a
.locale-sync-tmp/ scratch directory next to the translated file, ready to feed back into the
translate-locale workflow.

Prints one JSON object to stdout: {"count", "path"} — path is null when there are no violations.
Nothing else goes to stdout/stderr on the success path.

Usage:
    python verify_placeholders.py <delta_source.json> <delta_translated.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from locale_utils import LoadedFile, load_json_pairs

CONTEXT_PATTERN = re.compile(r"^\[ctx:[^\]]*\]\s*")
CONTEXT_ANYWHERE_PATTERN = re.compile(r"\[ctx:[^\]]*\]")
TAG_PATTERN = re.compile(r"\{[^{}]*\}|%\d*\$?[sdf]|<[^<>]+>")


def extract_tags(text: str) -> set[str]:
    """Return the set of placeholder/HTML-tag substrings found in text."""
    return set(TAG_PATTERN.findall(text))


def find_violations(source: LoadedFile, translated: LoadedFile) -> list[tuple[str, str]]:
    """Return (key, value) for every translated entry whose value adds a placeholder its key
    doesn't have, or repeats the key's [ctx: ...] annotation."""
    violations: list[tuple[str, str]] = []

    for key, value in translated.pairs:
        if not isinstance(value, str):
            continue

        bare_key = CONTEXT_PATTERN.sub("", key)
        key_tags = extract_tags(bare_key)
        value_tags = extract_tags(value)
        if not value_tags.issubset(key_tags):
            violations.append((key, value))
            continue

        if CONTEXT_ANYWHERE_PATTERN.search(value):
            violations.append((key, value))

    return violations


def main() -> None:
    """Check every translated entry, write violators for retranslation, print the JSON summary."""
    if len(sys.argv) != 3:
        sys.exit("usage: verify_placeholders.py <delta_source_path> <delta_translated_path>")
    source_path = Path(sys.argv[1])
    translated_path = Path(sys.argv[2])

    source = load_json_pairs(source_path)
    translated = load_json_pairs(translated_path)
    source_values = source.as_dict()

    violations = find_violations(source, translated)

    scratch_dir = translated_path.parent / ".locale-sync-tmp"
    violators_path = scratch_dir / f"{translated_path.stem}.violations.json"

    if len(violations) == 0:
        if violators_path.exists():
            violators_path.unlink()
        print(json.dumps({"count": 0, "path": None}))
        return

    export: dict[str, object] = {}
    for key, value in violations:
        export[key] = source_values.get(key, value)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    violators_path.write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"count": len(violations), "path": str(violators_path.resolve())}))


if __name__ == "__main__":
    main()
