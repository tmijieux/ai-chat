# Handoff: llama.cpp per-part `parse_special` tokenization

## Context

When using llama-server's `/v1/chat/completions` (OpenAI-compatible) endpoint, the chat template is rendered to a **flat string** and tokenized in one call with `parse_special=true`. This means any special token strings that appear inside tool results or user content — e.g. `<tool_call>`, `</tool_call>`, `<|im_start|>`, `<|im_end|>` — get tokenized as those special tokens, corrupting the context structure entirely.

This is a real vulnerability for agentic use: if a tool reads a file or returns grep output that contains these strings, the model's context is silently broken.

## Prior research done this session

### Qwen3.5 tool format (from reading the local GGUF dump + template)

- Template is at `C:/Users/tmijieux/ai/my_ai_chat/qwen3.5_template_jinja.txt` (extracted from `qwen3.5_dump.json` metadata key `tokenizer.chat_template`).
- Tool results sent as `role: "tool"` messages. The chat template wraps them automatically:
  ```
  <|im_start|>user
  <tool_response>
  {message.content}
  </tool_response>
  <|im_end|>
  ```
- Multiple consecutive `role: "tool"` messages are grouped into one `<|im_start|>user` block with multiple `<tool_response>` tags.
- Content inside `<tool_response>` is `message.content` verbatim (trimmed). We use `json.dumps(result_dict)` — a compact JSON string with default separators (spaces after `:` and `,`). This matches official Qwen docs examples. No change needed.
- Tool call format emitted by model:
  ```xml
  <tool_call>
  <function=name>
  <parameter=key>
  value
  </parameter>
  </function>
  </tool_call>
  ```
- Training data: Qwen3 technical report (arxiv 2505.09388) states training used diverse formats (JSON, XML, Python-style, TypeScript). No canonical "best" format for tool response content — JSON is fine.

### llama.cpp internals (from reading source at `C:/Users/tmijieux/ai/llama.cpp/`)

**Template rendering:**
- `common_chat_templates_init` loads the template from GGUF metadata via `llama_model_chat_template()`.
- Rendered by llama.cpp's own jinja engine (`common/jinja/`). The Qwen3.5 jinja template runs as-is — no hardcoded Qwen logic.
- The jinja engine internally tracks `is_input` flags on string parts (see `common/jinja/string.h:17`): template-emitted text vs. user/variable-substituted text. Propagation rules are defined (one-to-one preserves flag, many-to-one requires all-input, etc.).

**Tokenization path (chat API):**
- `common_chat_template_direct_apply_impl` calls `parts->as_string().str()` — **flattens all parts to a single string**, discarding `is_input` tags.
- That string becomes `chat_params.prompt` (a `std::string`), assigned to `llama_params["prompt"]` at `server-common.cpp:1092`.
- Tokenized via `tokenize_input_prompts → tokenize_mixed → common_tokenize(vocab, s, add_special=true, parse_special=true)` — one call, all special tokens active.

**`is_input` is tracked but never used downstream** — confirmed by grep: only referenced in `tests/test-chat-template.cpp:272` for debug printing.

**Response parsing:**
- Decoded to flat string, then parsed by a PEG parser (`common/chat-peg-parser.cpp`).
- The auto-parser (`common/chat-diff-analyzer.cpp`) inspects the template source to detect markers (`<think>`, `</think>`, `<tool_call>`, etc.) and builds grammar rules. No special token IDs involved — pure string matching.

**The per-part path that exists but isn't used:**
- `tokenize_mixed` (`server-common.cpp:641`) already handles JSON arrays: plain strings get `common_tokenize(..., parse_special)`, raw token IDs are passed through directly.
- If `chat_params.prompt` were a JSON array instead of a flat string, the infrastructure is already in place to tokenize parts differently.

---

## Implementation plan

Wire the existing `is_input` tracking in the jinja engine through to the tokenization call. **4-file change.**

### File 1: `common/chat.cpp` — `common_chat_template_direct_apply_impl`

Instead of flattening to a string, produce a JSON array of parts:

```cpp
// Currently:
std::string result = parts->as_string().str();

// Change to:
json result = json::array();
for (const auto & part : parts->val_str.parts) {
    if (part.is_input) {
        result.push_back({{"text", part.val}, {"parse_special", false}});
    } else {
        result.push_back(part.val);  // plain string = parse_special=true
    }
}
```

The return type of this function (and callers that use it for `.prompt`) needs to change from `std::string` to `json`.

### File 2: `common/chat-auto-parser.h` — `common_chat_params`

Change `prompt` field type from `std::string` to `json`. Already assigned into a `json` value at `server-common.cpp:1092` — callers unaffected.

Note: `generation_prompt` is also a `std::string` and used similarly — may need the same treatment if it can contain user content (check: it's the suffix added after the last message, typically pure template text, so probably fine to leave as string).

### File 3: `tools/server/server-common.cpp` — `tokenize_mixed`

Add handling for the new object format in the array branch:

```cpp
} else if (p.is_object() && p.contains("text")) {
    bool ps = p.value("parse_special", true);
    auto s   = p["text"].get<std::string>();
    llama_tokens toks = common_tokenize(vocab, s, /* add_special= */ false, ps);
    prompt_tokens.insert(prompt_tokens.end(), toks.begin(), toks.end());
}
```

### File 4: nothing else
`llama_params["prompt"] = chat_params.prompt` at `server-common.cpp:1092` already assigns a `json` value — if `prompt` becomes a JSON array it passes through unchanged. `tokenize_input_prompts` → `tokenize_input_subprompt` → `tokenize_mixed` already handles arrays via `json_is_array_of_mixed_numbers_strings` check — may need to widen that check to also accept arrays containing objects.

### Edge cases to verify

- `common_chat_format_single` (`chat.cpp:555`) uses `.prompt` for diff-based single-message formatting — needs to handle json array.
- `common_chat_verify_template` (`chat.cpp:529`) — uses apply result, check if it compares strings.
- The `generation_prompt` field (the `<|im_start|>assistant\n<think>\n` suffix) is pure template text, no user input — safe to keep as `std::string`.
- Parts from `tojson` filter (used for tool schemas in system prompt): these are template-generated JSON strings, not user input, so `is_input=false` — correct, they tokenize with `parse_special=true`.
- `trim` filter on `render_content` output: `string.h` says one-to-many preserves `is_input` — so trimming a user-input string keeps it marked as input. Good.

---

## Suggested skills

- `/plan` — before touching `common_chat_params` type, map all callers of `.prompt` across the codebase (it's used in tests, common/, tools/server/).
- `/tdd` — write a test in `tests/test-chat-template.cpp` that injects `<tool_call>` into a tool result and asserts the token sequence does NOT contain the special `<tool_call>` token ID.
- `/code-review` — after implementation, review for any place that stringifies `chat_params.prompt` assuming it's a flat string.
