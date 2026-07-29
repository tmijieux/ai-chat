"""Find locale entries whose value was never actually translated.

Two failure modes show up after a translate-locale run:

  copied key   — the value is the key itself, i.e. the model echoed the French line instead of
                 rewriting it into the target language.
  copied source — the value is byte-identical to the source file's value, i.e. the English text was
                 left in place. Only checked when --source is given.

Only exact, character-for-character identity counts. A value that differs from the key or the source
even slightly has been touched by the translator, so it is treated as translated and never reported.

Short values (--min-length, default 3) are counted but listed apart, because strings like "OK",
"Email" or "1" are legitimately identical across languages.

This script never modifies the file it inspects. With --out it writes the suspect entries as a
source-shaped JSON file, ready to feed back into the translate-locale workflow.

Usage:
    python scripts/find_untranslated_values.py <translated.json>
    python scripts/find_untranslated_values.py <translated.json> --source en.json
    python scripts/find_untranslated_values.py <translated.json> --source en.json --out redo-nl.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from locale_utils import LoadedFile, load_json_pairs, write_pairs_file

REASON_COPIED_KEY = "value is the key"
REASON_COPIED_SOURCE = "value is the source value"
REASON_EMPTY = "value is empty"


@dataclass
class Suspect:
    """One entry that looks untranslated, with the comparison that flagged it."""

    key: str
    value: str
    reason: str

    def is_short(self, min_length: int) -> bool:
        """True when the value is too short for identity to mean anything (e.g. "OK", "1").

        An empty value is never "short" — it is a defect at any length, so it always stays in the
        main report.
        """
        if self.reason == REASON_EMPTY:
            return False
        return len(self.value.strip()) < min_length


def collect_suspects(translated: LoadedFile, source: LoadedFile | None) -> list[Suspect]:
    """Scan every entry for a value left exactly identical to its key or its source-file counterpart.

    Each entry yields at most one suspect: the key comparison wins over the source comparison, so the
    reported reason is always the most direct one. Any difference at all, however small, means the
    entry was touched by the translator and is not reported.
    """
    source_values = source.as_dict() if source is not None else {}
    suspects: list[Suspect] = []

    for key, value in translated.pairs:
        if not isinstance(value, str):
            continue

        if value.strip() == "":
            suspects.append(Suspect(key, value, REASON_EMPTY))
            continue

        if value == key:
            suspects.append(Suspect(key, value, REASON_COPIED_KEY))
            continue

        source_value = source_values.get(key)
        if isinstance(source_value, str) and value == source_value:
            suspects.append(Suspect(key, value, REASON_COPIED_SOURCE))

    return suspects


def print_group(title: str, suspects: list[Suspect]) -> None:
    """Print one titled block of suspects."""
    if len(suspects) == 0:
        return
    print("-" * 72)
    print(f"{title} — {len(suspects)} entr{'y' if len(suspects) == 1 else 'ies'}")
    print("-" * 72)
    for suspect in suspects:
        print(f"  key:   {suspect.key!r}")
        print(f"  value: {suspect.value!r}")
        print()


def report(
    translated: LoadedFile,
    source: LoadedFile | None,
    suspects: list[Suspect],
    min_length: int,
) -> list[Suspect]:
    """Print the counts and every group of suspects. Returns the long-enough suspects worth re-translating."""
    total = len(translated.pairs)
    long_suspects = [s for s in suspects if not s.is_short(min_length)]
    short_suspects = [s for s in suspects if s.is_short(min_length)]

    by_reason: dict[str, list[Suspect]] = {}
    for suspect in long_suspects:
        by_reason.setdefault(suspect.reason, []).append(suspect)

    clean = total - len(suspects)
    print("SUMMARY")
    print(f"  entries in file ............... {total}")
    print(f"  look translated ............... {clean}")
    print(f"  value is the key .............. {len(by_reason.get(REASON_COPIED_KEY, []))}")
    if source is not None:
        print(f"  value is the source value ..... {len(by_reason.get(REASON_COPIED_SOURCE, []))}")
    else:
        print("  (source-value checks skipped — pass --source <en.json> to enable them)")
    print(f"  empty values .................. {len(by_reason.get(REASON_EMPTY, []))}")
    print(f"  short, ignored (< {min_length} chars) .... {len(short_suspects)}")
    print()

    print_group("VALUE IS THE KEY — never translated", by_reason.get(REASON_COPIED_KEY, []))
    print_group("VALUE IS THE SOURCE VALUE — left in the source language",
                by_reason.get(REASON_COPIED_SOURCE, []))
    print_group("EMPTY VALUES", by_reason.get(REASON_EMPTY, []))

    if len(short_suspects) > 0:
        print("-" * 72)
        print(f"SHORT VALUES — {len(short_suspects)} entr{'y' if len(short_suspects) == 1 else 'ies'}"
              f" identical but under {min_length} characters; usually fine")
        print("-" * 72)
        for suspect in short_suspects:
            print(f"  {suspect.key!r}: {suspect.value!r}   ({suspect.reason})")
        print()

    return long_suspects


def main() -> None:
    """Parse arguments, scan for untranslated values, report them, and optionally export them."""
    parser = argparse.ArgumentParser(
        description="Find locale entries whose value is exactly identical to the key or the source value.",
    )
    parser.add_argument("translated", type=Path, help="Translated locale file to inspect.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source locale file, to also detect values left in the source language.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the suspect entries to this JSON file, for re-translation. Omit to only report.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Allow --out to overwrite an existing file."
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Values shorter than this are listed apart, not counted as failures. Default: 3",
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="Indentation used when writing --out. Default: 2"
    )
    args = parser.parse_args()

    translated = load_json_pairs(args.translated)
    source = load_json_pairs(args.source) if args.source is not None else None

    print("=" * 72)
    print(f"translated: {translated.path}  ({len(translated.pairs)} entries)")
    print(f"source:     {source.path if source is not None else '(not given)'}")
    print(f"exact matches only   min length: {args.min_length}")
    print("=" * 72)
    print()

    suspects = collect_suspects(translated, source)
    reportable = report(translated, source, suspects, args.min_length)

    if args.out is None:
        if len(reportable) > 0:
            print(f"Re-run with --out <file> to export {len(reportable)} entr"
                  f"{'y' if len(reportable) == 1 else 'ies'} for re-translation.")
        return

    if len(reportable) == 0:
        print("Nothing to export — no file written.")
        return

    # Export the SOURCE value where we have one, so the re-translation input looks like the original
    # file rather than like the broken output.
    source_values = source.as_dict() if source is not None else {}
    export_pairs: list[tuple[str, object]] = []
    keys_absent_from_source = 0
    for suspect in reportable:
        source_value = source_values.get(suspect.key)
        if source_value is None:
            keys_absent_from_source += 1
            export_pairs.append((suspect.key, suspect.value))
        else:
            export_pairs.append((suspect.key, source_value))
    write_pairs_file(args.out, export_pairs, args.indent, args.force)

    if source is not None and keys_absent_from_source > 0:
        print(f"NOTE: {keys_absent_from_source} exported key(s) were not found verbatim in the source, so "
              "their current value was exported instead.")
        print("      Run scripts/fix_locale_keys.py --apply first to repair altered keys, then re-run this.")


if __name__ == "__main__":
    main()
