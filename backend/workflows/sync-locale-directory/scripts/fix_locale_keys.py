"""Repair localization keys that a translation pass altered.

The translate-locale workflow is supposed to copy each JSON key byte-for-byte and only rewrite
the value. In practice the model normalizes "smart" punctuation in the keys — most commonly the
French curly apostrophe (U+2019) becoming a straight ASCII apostrophe (U+0027). Keys are lookup
identifiers, so any such change silently breaks the translation at runtime.

This script matches every key in the translated file back to a key in the source file and
restores the source spelling exactly. Matching runs in tiers, most trustworthy first:

  exact       — byte-for-byte identical, nothing to do
  normalized  — identical once smart punctuation / unicode form / whitespace are normalized
  fuzzy       — close enough by character similarity, above --threshold
  unmatched   — no candidate found; left untouched and reported

Dry-run is the default: nothing is written unless --apply is passed.

Usage:
    python scripts/fix_locale_keys.py <source.json> <translated.json>
    python scripts/fix_locale_keys.py <source.json> <translated.json> --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from locale_utils import (
    TIER_EXACT,
    TIER_FUZZY,
    TIER_NORMALIZED,
    TIER_UNMATCHED,
    KeyMatch,
    LoadedFile,
    describe_character_differences,
    load_json_pairs,
    match_keys,
)


def report(
    source: LoadedFile,
    translated: LoadedFile,
    matches: list[KeyMatch],
    threshold: float,
    apply_changes: bool,
) -> None:
    """Print the summary counts and the per-key detail of what differs and what would change."""
    tier_counts = {
        TIER_EXACT: 0,
        TIER_NORMALIZED: 0,
        TIER_FUZZY: 0,
        TIER_UNMATCHED: 0,
    }
    for match in matches:
        tier_counts[match.tier] += 1

    rewrites = [m for m in matches if m.needs_rewrite()]
    matched_source_keys = {m.source_key for m in matches if m.source_key is not None}
    missing_from_translation = [k for k in source.keys() if k not in matched_source_keys]

    source_key_list = source.keys()
    translated_key_list = translated.keys()
    duplicate_source_keys = len(source_key_list) - len(set(source_key_list))
    duplicate_translated_keys = len(translated_key_list) - len(set(translated_key_list))

    print("=" * 72)
    print(f"source:     {source.path}  ({len(source_key_list)} keys)")
    print(f"translated: {translated.path}  ({len(translated_key_list)} keys)")
    print(f"mode:       {'APPLY (file will be rewritten)' if apply_changes else 'DRY RUN (nothing written)'}")
    print(f"fuzzy threshold: {threshold}")
    print("=" * 72)
    print()
    print("SUMMARY")
    print(f"  exact matches ................. {tier_counts[TIER_EXACT]}")
    print(f"  normalized matches ............ {tier_counts[TIER_NORMALIZED]}   (punctuation/whitespace only)")
    print(f"  fuzzy matches ................. {tier_counts[TIER_FUZZY]}   (review these)")
    print(f"  unmatched ..................... {tier_counts[TIER_UNMATCHED]}   (left untouched)")
    print(f"  keys to rewrite ............... {len(rewrites)}")
    print(f"  source keys missing in output . {len(missing_from_translation)}")
    print(f"  duplicate keys in source ...... {duplicate_source_keys}")
    print(f"  duplicate keys in translation . {duplicate_translated_keys}")
    print()

    normalized_rewrites = [m for m in rewrites if m.tier == TIER_NORMALIZED]
    if len(normalized_rewrites) > 0:
        print("-" * 72)
        print(f"NORMALIZED MATCHES — {len(normalized_rewrites)} key(s) would be restored to the source spelling")
        print("-" * 72)
        for match in normalized_rewrites:
            print(f"  current: {match.translated_key!r}")
            print(f"  source:  {match.source_key!r}")
            print(f"  diff:    {describe_character_differences(match.translated_key, match.source_key)}")
            print()

    fuzzy_rewrites = [m for m in matches if m.tier == TIER_FUZZY]
    if len(fuzzy_rewrites) > 0:
        print("-" * 72)
        print(f"FUZZY MATCHES — {len(fuzzy_rewrites)} key(s), NOT identical after normalization; check each one")
        print("-" * 72)
        for match in sorted(fuzzy_rewrites, key=lambda m: m.similarity):
            print(f"  similarity: {match.similarity:.3f}")
            print(f"  current:    {match.translated_key!r}")
            print(f"  source:     {match.source_key!r}")
            print(f"  diff:       {describe_character_differences(match.translated_key, match.source_key)}")
            print()

    unmatched = [m for m in matches if m.tier == TIER_UNMATCHED]
    if len(unmatched) > 0:
        print("-" * 72)
        print(f"UNMATCHED — {len(unmatched)} key(s) have no source counterpart above the threshold")
        print("-" * 72)
        for match in unmatched:
            print(f"  {match.translated_key!r}")
        print()

    if len(missing_from_translation) > 0:
        print("-" * 72)
        print(f"MISSING FROM TRANSLATION — {len(missing_from_translation)} source key(s) never appear in the output")
        print("-" * 72)
        for key in missing_from_translation:
            print(f"  {key!r}")
        print()
        print("  Use scripts/extract_missing_keys.py to write these out as a file you can re-translate.")
        print()


def write_fixed_file(translated: LoadedFile, matches: list[KeyMatch], indent: int) -> int:
    """Rewrite the translated file with source key spellings restored. Returns the number of keys changed.

    A .bak copy of the original is made first. Key order and values are preserved exactly; only the
    key strings change, so the result stays diffable against the previous version.
    """
    rewritten_pairs: list[tuple[str, object]] = []
    changed = 0
    for (original_key, value), match in zip(translated.pairs, matches):
        if match.needs_rewrite():
            rewritten_pairs.append((match.source_key, value))
            changed += 1
        else:
            rewritten_pairs.append((original_key, value))

    backup_path = translated.path.with_suffix(translated.path.suffix + ".bak")
    shutil.copy2(translated.path, backup_path)

    payload = json.dumps(dict(rewritten_pairs), ensure_ascii=False, indent=indent)
    translated.path.write_text(payload + "\n", encoding="utf-8")

    print(f"backup written to: {backup_path}")
    print(f"rewrote {changed} key(s) in: {translated.path}")
    return changed


def main() -> None:
    """Parse arguments, match keys, report, and optionally apply the fixes."""
    parser = argparse.ArgumentParser(
        description="Restore localization keys that a translation pass altered (e.g. ’ turned into ').",
    )
    parser.add_argument("source", type=Path, help="Source locale file whose keys are authoritative.")
    parser.add_argument("translated", type=Path, help="Translated locale file whose keys may be altered.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the fixes. Without this flag the script only reports (dry run).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Minimum similarity (0-1) for a fuzzy match to be accepted. Default: 0.90",
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="Indentation used when rewriting the JSON. Default: 2"
    )
    args = parser.parse_args()

    source = load_json_pairs(args.source)
    translated = load_json_pairs(args.translated)

    matches = match_keys(source.keys(), translated.keys(), args.threshold)
    report(source, translated, matches, args.threshold, args.apply)

    rewrite_count = len([m for m in matches if m.needs_rewrite()])
    if not args.apply:
        if rewrite_count > 0:
            print(f"DRY RUN — nothing was written. Re-run with --apply to rewrite {rewrite_count} key(s).")
        else:
            print("DRY RUN — no changes needed.")
        return

    if rewrite_count == 0:
        print("No changes needed — file left untouched.")
        return
    write_fixed_file(translated, matches, args.indent)


if __name__ == "__main__":
    main()
