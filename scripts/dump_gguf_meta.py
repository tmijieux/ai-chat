"""Dump GGUF KV metadata for one or more model files using _read_gguf_kv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from tokenizer import _read_gguf_kv


MODELS = [
    ("Ollama qwen3.5:9b", Path.home() / ".ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"),
    ("Unsloth Q3_K_XL",   Path.home() / "ai/models/unsloth/Qwen3.5-9B-UD-Q3_K_XL.gguf"),
    ("Unsloth Q4_K_M",    Path.home() / "ai/models/unsloth/Qwen3.5-9B-Q4_K_M.gguf"),
]


def dump_model(label, path, out):
    out.write(f"{'='*70}\n")
    out.write(f"  {label}\n")
    out.write(f"  {path}\n")
    out.write(f"{'='*70}\n\n")

    if not path.exists():
        out.write("  FILE NOT FOUND\n\n")
        return

    kv = _read_gguf_kv(str(path))
    for key, val in sorted(kv.items()):
        # Truncate very long lists (e.g. token lists)
        if isinstance(val, list) and len(val) > 20:
            preview = val[:5]
            out.write(f"  {key} = {preview} ... ({len(val)} items)\n")
        else:
            out.write(f"  {key} = {val!r}\n")
    out.write("\n")


def main():
    out_path = Path(__file__).parent / "gguf_meta_dump.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for label, path in MODELS:
            print(f"Reading {label} ...")
            dump_model(label, path, out)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
