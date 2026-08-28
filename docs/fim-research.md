# FIM (Fill-in-the-Middle) research

Exploration of whether the local model supports fill-in-the-middle code completion well
enough to build a feature on. Tested 2026-08-28 against **Ornith-1.5-9B-Q4_K_M** (the
currently active local model — see `backend/llm/llama_server.py` `ACTIVE_MODEL`), driven
through llama-server's raw generation endpoints. No FIM feature exists in the app yet; this
is groundwork.

## TL;DR

- **Ornith-1.5-9B has full FIM support.** All six Qwen2.5-Coder-style FIM tokens are in the
  vocab as `CONTROL` tokens, llama.cpp auto-detects them, and the `/infill` endpoint works
  with zero configuration.
- **The model genuinely understands FIM** — single-line completions in real code are
  frequently exact; it reads cross-file context passed via `<|file_sep|>`.
- **Not production-ready as-is.** Two recurring problems: greedy decoding falls into
  repetition loops on short snippets, and every multi-line fill over-generates past the
  "middle" into re-deriving the suffix. Whole-function synthesis introduces subtle bugs
  (it is a 9B Q4 general model, not a dedicated code model).
- **Usable today** for short (1–few line) completions with first-few-lines truncation, a
  repetition penalty, and suffix-overlap trimming.

## FIM tokens in the vocab

Probed both GGUFs directly with the `gguf` Python package (`tokenizer.ggml.tokens` +
`tokenizer.ggml.token_type`).

| token | id | Ornith-1.5-9B type | Qwen3.5-9B type |
|---|---|---|---|
| `<|fim_prefix|>` | 248060 | 3 = CONTROL | 4 = USER_DEFINED |
| `<|fim_middle|>` | 248061 | 3 = CONTROL | 4 = USER_DEFINED |
| `<|fim_suffix|>` | 248062 | 3 = CONTROL | 4 = USER_DEFINED |
| `<|fim_pad|>`    | 248063 | 3 = CONTROL | 4 = USER_DEFINED |
| `<|repo_name|>`  | 248064 | 3 = CONTROL | 4 = USER_DEFINED |
| `<|file_sep|>`   | 248065 | 3 = CONTROL | 4 = USER_DEFINED |

Other relevant tokens (Ornith): EOS = `<|im_end|>` (248046), `<|endoftext|>` = 248044
(pad token).

**Ornith marks the FIM tokens `CONTROL`; Qwen3.5-9B marks them `USER_DEFINED`.** This
matters:

- On Ornith they tokenize as single special tokens *regardless* of the `special` flag
  (verified: `/tokenize` of `<|fim_prefix|>A<|fim_suffix|>B<|fim_middle|>` returns
  `[248060, 32, 248062, 33, 248061]` with `special` both true and false).
- llama.cpp's FIM auto-detection (by known token string name) picks up `CONTROL` tokens
  cleanly. Neither GGUF carries the explicit `tokenizer.ggml.fim_*_token_id` metadata keys,
  and neither has an infill chat template — but Ornith's `/infill` endpoint still works
  out of the box because of the string-name + type detection.

Qwen3.5-9B's `USER_DEFINED` typing was not re-tested for `/infill` auto-detection in this
session; the prior finding was only that the tokens exist in the vocab.

## Endpoints

Everything was driven through llama-server started plainly:

```
llama-server -m Ornith-1.5-9B-Q4_K_M.gguf -c 8192 -ngl 99 --port 8080 --host 127.0.0.1
```

(No `--mmproj`, smaller context than the app's 32768 — this was a throwaway text-only
probe server.)

### `/infill`

Works with zero config. Request:

```json
{ "input_prefix": "...", "input_suffix": "...", "n_predict": 64, "temperature": 0 }
```

llama-server assembles the prompt internally as:

```
<|repo_name|>myproject
<|file_sep|>filename
<|fim_prefix|>{input_prefix}<|fim_suffix|>{input_suffix}<|fim_middle|>
```

and handles EOT/EOG stopping automatically. Downside: the `repo_name`/`file_sep` values it
injects are placeholders — no way to pass real ones without `input_extra`.

### Raw `/completion` (what we want to build on)

llama-server's `/completion` parses special-token strings in the `prompt` field by default
(same as the existing think-gated path in `llama_server.py` relies on for `<think>`), so
the FIM prompt can be handed over as a plain string:

```
<|repo_name|>{repo}
<|file_sep|>{sibling1_path}
{sibling1_content}
<|file_sep|>{sibling2_path}
{sibling2_content}
<|file_sep|>{target_path}
<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
```

Recommended stop list for the raw path:
`["<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>", "<|im_end|>"]`.

The `<|repo_name|>` / `<|file_sep|>` scaffold measurably improves stop behaviour vs. a bare
`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>` — with the scaffold the model hits EOS at
the right point more often instead of running on.

## Test results

### 1. Synthetic single-line completions (greedy, temp 0, top_k 1)

All contextually correct:

| context | completion |
|---|---|
| `def factorial(n): … return ` | `n * factorial(n - 1)` |
| `for x in nums:` (sum-positives body) | `total += x` |
| `self.x = x` → `def dist` (fill `__init__` tail) | `self.y = y` |
| `requests.get(url, timeout=` … `)` | `10` |
| `count = signal(` … `)` (Angular) | `0` |
| `def set(self, key, value):` (Cache method body) | `self._data[key] = value` |
| `if ` … `: continue` (loop filter) | `item is None` |

Repetition failures on greedy: some completions that should have stopped kept emitting
degenerate loops — `print(resp.status_code)\nprint(resp.text)\n…` spam, `}\n }\n }\n` spam.
Raising temperature to 0.3 fixed some but caused a hard derail on one case (Python int gap
→ HTML/XML garbage). The DRY sampler is on by llama-server default and was not enough alone.

### 2. Cross-file context via `<|file_sep|>` (greedy, temp 0)

The model demonstrably reads sibling files:

| sibling file provides | fill produced | correct? |
|---|---|---|
| `haversine_km(lat1, lon1, lat2, lon2)` in `geometry.py` | `haversine_km(user_lat, user_lon, lat, lon)` | ✅ name + arg order, merged with local loop vars |
| `frobnicate_widget(widget, *, sparkle_level=3)` (invented name) | `frobnicate_widget(w)` | ✅ recovered a name it could not have guessed |
| `API_BASE` constant in `config.py` | `API_BASE + path` | ✅ correct symbol (then ran on) |
| `CounterService` with `inject()` in a `.service.ts` | `inject(CounterService)` | ✅ correct call (then ran on) |

One run leaked a training-data artifact after the correct fill:
`// Here are some relevant code fragments from other files of the repo:` — the
natural-language preamble Qwen2.5-Coder's repo-level FIM training data puts before
`<|file_sep|>` blocks. Confirms the model saw this exact repo-FIM format in training. Harmless
with output truncation.

### 3. README.md / TASK.md as a "prompt" via `<|file_sep|>`

Feeding a `README.md` (project conventions) and a `TASK.md` (what to implement in this gap)
as `<|file_sep|>` blocks *before* the target file does steer the fill:

- **Constraint following ✅** — `TASK.md` = "MUST be iterative, recursion forbidden, O(1)
  space" for `fib(n)` → produced the textbook iterative `a, b = 0, 1; for _ in range(n):
  a, b = b, a + b; return a`, clean EOS.
- **Convention following ✅** — `README.md` = "all public functions raise `InvoiceError` on
  bad input" → the fill led with `raise InvoiceError(...)` validation using exactly the dict
  keys named in `TASK.md`.
- **Complex algorithm logic ⚠️** — "generator, iterative, yield each ancestor up to root,
  don't yield path itself" → got the shape (generator over `.parent`) but botched loop
  termination, then degenerated into a repeated comment.

Guidance: put `TASK.md` as the **last** `<|file_sep|>` block before the target (closest to
the cursor — proximity matters in FIM). It reliably picks up constraints and naming, less
reliably complex logic. A long `TASK.md` far from the cursor risks being diluted by FIM's
pull to just continue local code.

### 4. Real file — `backend/rag/chunking.py`

With a `CONTEXT.md` glossary excerpt as the README-style context block.

**Single-line gap (the `for piece in _split_oversized_line(...)` loop header):**
reproduced character-for-character, clean EOS. This is the realistic IDE-autocomplete case
and it was exact.

**2-line gap (`_overlap_tail` call + assignment):** the two real lines are correct, but it
inserted a stray comment between them and then over-generated into the suffix (kept
re-deriving `current_chars` / `current_start_line`, which literally follow in the file).

**Whole function body of `_split_oversized_line` from its docstring:** right identifiers
(`max_chars`, `overlap_chars`), core loop *equivalent* to the real one — but dropped the
`len(line) == 0` guard, added an off-style `else` branch, and the trailing
`yield line[start:]` is a bug (empty/duplicate slice).

| gap size | result |
|---|---|
| one line | ✅ frequently exact |
| few lines | ✅ correct intent, over-generates into the suffix |
| whole function body | ⚠️ right shape & identifiers, subtle bugs, off-style |

Notable: with ~1000+ tokens of real file context, greedy decoding was **stable** — zero
repetition loops across the real-file runs, unlike the tiny synthetic snippets. Bigger real
context helps.

## Known weaknesses

1. **Repetition loops** on greedy decoding for short completions. Mitigation: repetition
   penalty (`repeat_penalty` ~1.1, `repeat_last_n` ~256), keep DRY on, small temperature —
   but temp > 0 occasionally derails hard. Large real context also suppresses it.
2. **Over-generation past the middle.** Every multi-line fill starts re-emitting the suffix.
   Mitigation: stop as soon as generated output starts overlapping the known suffix; for
   single-line contexts, hard-truncate at the first newline.
3. **Whole-function synthesis is buggy.** Fine for scaffolding, not for trusting output
   verbatim. It is a 9B Q4 general model.
4. **`/infill` scaffold values are placeholders.** Use raw `/completion` to pass a real repo
   name / file path / sibling files.

## Recommended recipe (if a FIM feature gets built)

- Raw `/completion`, prompt assembled as the repo-level format above.
- Context: 1–3 most-relevant sibling files (or the rest of the current file split around the
  cursor), kept small.
- If using README/TASK guidance: `TASK.md` as the last `<|file_sep|>` before the target.
- Sampling: `temperature` 0–0.2, `repeat_penalty` ~1.1, `repeat_last_n` ~256, DRY on.
- Stop: `["<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>", "<|im_end|>"]`.
- Post-process: trim generated text where it begins to match the suffix; for single-line
  use, cut at the first newline.
- `n_predict`: generous (150–200) if the task expects validation + logic, small if it is a
  line completion.

## Open questions

- Does Qwen3.5-9B's `USER_DEFINED` FIM-token typing break llama.cpp `/infill` auto-detection,
  or does the string-name detection still catch it? (Not tested — Ornith is the active model.)
- Is there a real use case in this app? Candidate ideas: inline code completion in a future
  editor surface; `edit_file` speed-ups by having the model emit only the changed span;
  filling templated scaffolds. None currently planned.
