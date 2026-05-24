# ADR-0004: Token counting via a 1-token inference call

**Date:** 2026-05-23  
**Status:** Accepted

## Context

Accurate token counts are central to the app's value proposition — the user needs to know exactly how many tokens are in context after each message, including system prompt and tool schema overhead. The count must match what Ollama actually sends to the model, not an approximation.

Ollama's HTTP API does not expose a standalone tokenizer endpoint. The only way to get the model's own `prompt_eval_count` is to run an inference call and read the field from the response.

## Decision

Use a `num_predict: 1` Ollama inference call to count tokens. The request sends the full message list (system prompt + history + tools) with `num_predict` capped at 1, so the model generates at most one token. The response's `prompt_eval_count` field is the exact token count for the submitted context.

This is implemented in `backend/agent/count_token.py` and called:
- After each non-agentic assistant response (via `POST /api/conversations/{id}/count-tokens`)
- After each agentic run completes (for the final message)
- On `SystemPromptTemplate` create/edit (to store `token_count` on the template)

## Alternatives considered

**Local tokenizer library (tiktoken, sentencepiece)** — no network call, fast. Rejected: the tokenizer must exactly match what Ollama uses for the loaded model. Ollama's tokenizer config is model-specific and not easily replicated without reading the model's internals. A mismatch would give wrong counts, undermining the feature's purpose.

**Ollama source / llama.cpp direct** — would give direct tokenizer access without inference overhead. Not rejected permanently — a future migration to llama.cpp direct API would enable this. For now Ollama is the runtime and its API is the only interface available.

**Accept estimates** — use a rough heuristic (e.g. chars / 4). Rejected: the whole point of the feature is accuracy; an estimate is no better than what any generic chat UI provides.

## Consequences

- Token counts are always exact — they match what the model actually received.
- Each count fires a real Ollama network call (fast, generates 1 token, but still a round-trip).
- If Ollama is unavailable, token counts silently fail (nullable `token_count` column).
- Future migration to llama.cpp or direct model access would allow replacing this with a pure tokenizer call.
