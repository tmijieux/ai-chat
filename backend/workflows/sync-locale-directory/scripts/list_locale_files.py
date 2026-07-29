"""List the locale files in a directory that are translation targets for a given source file.

Prints one JSON object to stdout: {"files": [...], "skipped": [...]}. Nothing else goes to
stdout/stderr on the success path — the sync-locale-directory workflow parses this output
directly as the loop's item list.

Usage:
    python list_locale_files.py <directory> <source_filename>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ISO 639-1 code (locale filename stem) -> language name. A file whose stem isn't listed here is
# reported as skipped rather than guessed, so an unattended directory sync never invents a language.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "fr": "French", "de": "German", "nl": "Dutch", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "ro": "Romanian", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "el": "Greek", "cs": "Czech",
    "sk": "Slovak", "hu": "Hungarian", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "sl": "Slovenian", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian", "tr": "Turkish",
    "ru": "Russian", "uk": "Ukrainian", "ar": "Arabic", "he": "Hebrew", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian",
}


def main() -> None:
    """Glob the directory for locale files, map each to a language, and print the result as JSON."""
    if len(sys.argv) != 3:
        sys.exit("usage: list_locale_files.py <directory> <source_filename>")
    directory = Path(sys.argv[1])
    source_filename = sys.argv[2]

    if not directory.is_dir():
        sys.exit(f"ERROR: not a directory: {directory}")

    files = []
    skipped = []
    for candidate in sorted(directory.glob("*.json")):
        if candidate.name == source_filename:
            continue
        language_name = LANGUAGE_NAMES.get(candidate.stem.lower())
        if language_name is None:
            skipped.append(candidate.name)
            continue
        files.append({
            "path": str(candidate.resolve()),
            "filename": candidate.name,
            "language_code": candidate.stem.lower(),
            "language_name": language_name,
        })

    print(json.dumps({"files": files, "skipped": skipped}))


if __name__ == "__main__":
    main()
