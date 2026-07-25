"""Shared helpers for the locale-file repair and audit scripts.

Loading, unicode normalization, character-level diffing and key matching all behave identically
across fix_locale_keys.py, extract_missing_keys.py and find_untranslated_values.py, so they live
here rather than being duplicated per script.
"""
from __future__ import annotations

import difflib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Smart punctuation the model tends to rewrite, mapped to a plain-ASCII stand-in. Used only to
# decide whether two strings are "the same"; the source spelling is always what gets written.
_PUNCTUATION_FOLDING = {
    "’": "'",  # right single quotation mark — the usual French apostrophe
    "‘": "'",  # left single quotation mark
    "‚": "'",  # single low-9 quotation mark
    "‛": "'",  # single high-reversed-9 quotation mark
    "ʼ": "'",  # modifier letter apostrophe
    "´": "'",  # acute accent used as apostrophe
    "`": "'",  # grave accent used as apostrophe
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "„": '"',  # double low-9 quotation mark
    "«": '"',  # left-pointing double angle quotation mark
    "»": '"',  # right-pointing double angle quotation mark
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‒": "-",  # figure dash
    "−": "-",  # minus sign
    "…": "...",  # horizontal ellipsis
    " ": " ",  # no-break space
    " ": " ",  # narrow no-break space
    " ": " ",  # thin space
    "​": "",   # zero-width space
    "﻿": "",   # zero-width no-break space / BOM
}

TIER_EXACT = "exact"
TIER_NORMALIZED = "normalized"
TIER_FUZZY = "fuzzy"
TIER_UNMATCHED = "unmatched"


@dataclass
class KeyMatch:
    """One translated key resolved (or not) against the source key list."""

    translated_key: str
    source_key: str | None
    tier: str
    similarity: float

    def needs_rewrite(self) -> bool:
        """True when the source spelling differs from what the translated file currently holds."""
        return self.source_key is not None and self.source_key != self.translated_key


@dataclass
class LoadedFile:
    """A JSON object loaded as ordered pairs, so duplicate keys stay visible."""

    path: Path
    pairs: list[tuple[str, object]]

    def keys(self) -> list[str]:
        """Return the keys in file order, including any duplicates."""
        return [key for key, _ in self.pairs]

    def as_dict(self) -> dict[str, object]:
        """Return the pairs as a dict; on duplicate keys the last one wins, as json.load would."""
        return dict(self.pairs)


def normalize_key(key: str) -> str:
    """Fold a string to a comparison form: NFC unicode, smart punctuation flattened, whitespace collapsed."""
    folded = unicodedata.normalize("NFC", key)
    for original, replacement in _PUNCTUATION_FOLDING.items():
        folded = folded.replace(original, replacement)
    return " ".join(folded.split())


def describe_run(text: str) -> str:
    """Render a short run of characters with their unicode codepoints, for readable diffs."""
    if text == "":
        return "(nothing)"
    parts = []
    for character in text:
        try:
            name = unicodedata.name(character)
        except ValueError:
            name = "UNNAMED"
        parts.append(f"{character!r} U+{ord(character):04X} {name}")
    return " + ".join(parts)


def describe_character_differences(left: str, right: str, max_differences: int = 4) -> str:
    """Describe the character runs that differ between two strings, most significant first."""
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    differences = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        differences.append(
            f"{describe_run(left[left_start:left_end])}  ->  {describe_run(right[right_start:right_end])}"
        )
        if len(differences) >= max_differences:
            differences.append("...")
            break
    if len(differences) == 0:
        return "(no character difference)"
    return "; ".join(differences)


def similarity(left: str, right: str) -> float:
    """Character-level similarity of two strings after normalization, in the range 0-1."""
    return difflib.SequenceMatcher(None, normalize_key(left), normalize_key(right), autojunk=False).ratio()


def load_json_pairs(path: Path) -> LoadedFile:
    """Load a JSON object preserving key order and duplicates. Exits with a clear message on failure."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        sys.exit(f"ERROR: cannot read {path}: {error}")

    try:
        pairs = json.loads(raw, object_pairs_hook=lambda items: items)
    except json.JSONDecodeError as error:
        sys.exit(
            f"ERROR: {path} is not valid JSON (line {error.lineno}, column {error.colno}): {error.msg}\n"
            "       Fix the syntax first — this script cannot match keys in a malformed file."
        )

    if not isinstance(pairs, list):
        sys.exit(f"ERROR: {path} must contain a JSON object at the top level.")
    return LoadedFile(path=path, pairs=pairs)


def match_keys(source_keys: list[str], translated_keys: list[str], threshold: float) -> list[KeyMatch]:
    """Resolve each translated key to a source key using exact, then normalized, then fuzzy matching.

    Each source key is consumed at most once so two translated keys can never claim the same source
    key. Fuzzy matching only runs on whatever is left over, keeping it cheap on large files.
    """
    available_source_keys = list(source_keys)

    exact_lookup: dict[str, int] = {}
    for index, key in enumerate(available_source_keys):
        exact_lookup.setdefault(key, index)

    normalized_lookup: dict[str, list[int]] = {}
    for index, key in enumerate(available_source_keys):
        normalized_lookup.setdefault(normalize_key(key), []).append(index)

    consumed: set[int] = set()
    matches: list[KeyMatch] = []
    deferred: list[int] = []  # positions in `matches` still needing a fuzzy pass

    for translated_key in translated_keys:
        exact_index = exact_lookup.get(translated_key)
        if exact_index is not None and exact_index not in consumed:
            consumed.add(exact_index)
            matches.append(KeyMatch(translated_key, available_source_keys[exact_index], TIER_EXACT, 1.0))
            continue

        normalized_candidates = normalized_lookup.get(normalize_key(translated_key), [])
        normalized_index = next((i for i in normalized_candidates if i not in consumed), None)
        if normalized_index is not None:
            consumed.add(normalized_index)
            matches.append(
                KeyMatch(translated_key, available_source_keys[normalized_index], TIER_NORMALIZED, 1.0)
            )
            continue

        deferred.append(len(matches))
        matches.append(KeyMatch(translated_key, None, TIER_UNMATCHED, 0.0))

    if len(deferred) > 0:
        leftover_indices = [i for i in range(len(available_source_keys)) if i not in consumed]
        for match_position in deferred:
            translated_key = matches[match_position].translated_key
            normalized_translated = normalize_key(translated_key)
            best_index: int | None = None
            best_ratio = 0.0
            for source_index in leftover_indices:
                if source_index in consumed:
                    continue
                ratio = difflib.SequenceMatcher(
                    None, normalized_translated, normalize_key(available_source_keys[source_index]), autojunk=False
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_index = source_index
            if best_index is not None and best_ratio >= threshold:
                consumed.add(best_index)
                matches[match_position] = KeyMatch(
                    translated_key, available_source_keys[best_index], TIER_FUZZY, best_ratio
                )

    return matches


def write_pairs_file(path: Path, pairs: list[tuple[str, object]], indent: int, force: bool) -> None:
    """Write pairs as a JSON object, preserving order. Refuses to clobber an existing file unless forced."""
    if path.exists() and not force:
        sys.exit(
            f"ERROR: {path} already exists. Pass --force to overwrite it, or choose another --out path."
        )
    payload = json.dumps(dict(pairs), ensure_ascii=False, indent=indent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    print(f"wrote {len(pairs)} entr{'y' if len(pairs) == 1 else 'ies'} to: {path}")


def find_missing_source_keys(
    source: LoadedFile, translated: LoadedFile, threshold: float
) -> list[tuple[str, object]]:
    """Return the source pairs whose key never matched anything in the translated file, in source order."""
    matches = match_keys(source.keys(), translated.keys(), threshold)
    matched_source_keys = {m.source_key for m in matches if m.source_key is not None}
    return [(key, value) for key, value in source.pairs if key not in matched_source_keys]
