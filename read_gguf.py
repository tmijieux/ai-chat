from gguf import GGUFReader

reader = GGUFReader(r"C:\Users\tmijieux\.ollama\models\blobs\sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c")

tokens_field = reader.fields["tokenizer.ggml.tokens"]
types_field  = reader.fields["tokenizer.ggml.token_type"]

tokens = [str(bytes(tokens_field.parts[i]), encoding="utf-8", errors="replace") for i in tokens_field.data]
types  = [int(types_field.parts[i][0]) for i in types_field.data]

# find special-looking tokens
for i, (tok, typ) in enumerate(zip(tokens, types)):
    if tok in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
        print(f"{i:6d}  type={typ}  {tok!r}")


