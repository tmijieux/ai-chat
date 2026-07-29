"""List the source keys a translated locale file never received, and optionally write them out.

When the translate-locale workflow gives up on a chunk (max_retries exhausted with
on_max_retries: continue) that chunk is silently absent from the output. This script finds those
gaps by matching every source key against the translated file — the same tiered exact/normalized/
fuzzy matching fix_locale_keys.py uses, so a key that is merely *misspelled* in the output counts
as present, not missing.

With --out it writes the missing entries as a source-shaped JSON file, which can be fed straight
back into the translate-locale workflow to fill the holes, then merged into the main file.

Dry-run is the default: nothing is written unless --out is passed.

Usage:
    python scripts/extract_missing_keys.py <source.json> <translated.json>
    python scripts/extract_missing_keys.py <source.json> <translated.json> --out missing-nl.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from locale_utils import find_missing_source_keys, load_json_pairs, write_pairs_file


def report(source_key_count: int, translated_key_count: int, missing: list[tuple[str, object]]) -> None:
    """Print how many keys are present versus missing, then list every missing key and its source value."""
    present = source_key_count - len(missing)
    coverage = 100.0 * present / source_key_count if source_key_count > 0 else 0.0

    print("SUMMARY")
    print(f"  keys in source ................ {source_key_count}")
    print(f"  keys in translation ........... {translated_key_count}")
    print(f"  source keys present in output . {present}")
    print(f"  source keys MISSING ........... {len(missing)}")
    print(f"  coverage ...................... {coverage:.1f}%")
    print()

    if len(missing) == 0:
        print("Every source key is accounted for.")
        return

    print("-" * 72)
    print(f"MISSING — {len(missing)} source key(s) with no counterpart in the translation")
    print("-" * 72)
    for key, value in missing:
        print(f"  {key!r}")
        print(f"      source value: {value!r}")
    print()


def main() -> None:
    """Parse arguments, find the missing source keys, report them, and optionally write them to a file."""
    parser = argparse.ArgumentParser(
        description="Find the source keys that never made it into a translated locale file.",
    )
    parser.add_argument("source", type=Path, help="Source locale file whose keys are authoritative.")
    parser.add_argument("translated", type=Path, help="Translated locale file to check for gaps.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the missing entries to this JSON file (same shape as the source). Omit for a dry run.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Allow --out to overwrite an existing file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Minimum similarity (0-1) below which a source key counts as missing. Default: 0.90",
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="Indentation used when writing --out. Default: 2"
    )
    args = parser.parse_args()

    source = load_json_pairs(args.source)
    translated = load_json_pairs(args.translated)

    print("=" * 72)
    print(f"source:     {source.path}")
    print(f"translated: {translated.path}")
    print(f"mode:       {'WRITE ' + str(args.out) if args.out is not None else 'DRY RUN (nothing written)'}")
    print(f"fuzzy threshold: {args.threshold}")
    print("=" * 72)
    print()

    missing = find_missing_source_keys(source, translated, args.threshold)
    report(len(source.keys()), len(translated.keys()), missing)

    if args.out is None:
        if len(missing) > 0:
            print(f"DRY RUN — nothing was written. Re-run with --out <file> to export {len(missing)} entr"
                  f"{'y' if len(missing) == 1 else 'ies'}.")
        return

    if len(missing) == 0:
        print("Nothing to export — no file written.")
        return
    write_pairs_file(args.out, missing, args.indent, args.force)


if __name__ == "__main__":
    main()
