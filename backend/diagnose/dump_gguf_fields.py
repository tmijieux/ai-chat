"""One-shot script to dump tokenizer-related fields from the GGUF file."""
from gguf import GGUFReader

GGUF_PATH = (
    "C:\\Users\\tmijieux\\.ollama\\models\\blobs\\"
    "sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
)

reader = GGUFReader(GGUF_PATH, "r")

for name, field in reader.fields.items():
    if not name.startswith("tokenizer"):
        continue
    # For large arrays (vocab, merges) just show the count and first few entries
    if field.types and str(field.types[0]) == "GGUFValueType.ARRAY":
        sample = field.contents(slice(0, 5))
        print(f"{name}: [{len(field.data)} items] first 5: {sample}")
    else:
        value = field.contents()
        preview = str(value)[:120]
        print(f"{name}: {preview}")
