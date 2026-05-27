from gguf import GGUFReader

GGUF_PATH = (
    "C:\\Users\\tmijieux\\.ollama\\models\\blobs\\"
    "sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
)

reader = GGUFReader(GGUF_PATH, "r")
fields = reader.fields
tokens = fields["tokenizer.ggml.tokens"].contents()
token_types = fields["tokenizer.ggml.token_type"].contents()

needles = [
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<|im_start|>", "<|im_end|>",
]

for needle in needles:
    if needle in tokens:
        rank = tokens.index(needle)
        print(f"{needle!r:30s} rank={rank:7d}  type={token_types[rank]}")
    else:
        print(f"{needle!r:30s} NOT FOUND (multi-token BPE)")

print()
print("All CONTROL tokens (type 3):")
for rank, (tok, tok_type) in enumerate(zip(tokens, token_types)):
    if tok_type == 3:
        print(f"  rank={rank:7d}  {tok!r}")

print()
print("All USER_DEFINED tokens (type 4):")
for rank, (tok, tok_type) in enumerate(zip(tokens, token_types)):
    if tok_type == 4:
        print(f"  rank={rank:7d}  {tok!r}")
