# ADR-0008: Working Memory Compression (Stage 3)

**Date:** 2026-06-18
**Status:** Accepted (disabled by default pending validation)

After a long exploratory run, Stage 1/2 compresses individual tool-role messages but the conversation still grows: every assistant thinking block and stub accumulates. The real fix is a single structured summary that replaces an entire exploration sequence.

## Decision

Introduce a `context_summary` message role that synthesises a span of conversation into a compact JSON structure (`goal`, `learned`, `done`, `rejected`, `dead_ends`). It is inserted into the message tree between the last covered message and the most recent user message. All covered messages are marked excluded. When compression runs again the old summary's JSON is folded into the new one, then excluded.

## Why `context_summary` is injected as `user` role into the LLM

OpenAI-compatible APIs only accept `user`, `assistant`, `tool`, and `system` roles. There is no neutral `context` role. We use `user` rather than `system` because a second `system` message would reactivate system-prompt handling and may confuse smaller models. The `[Context: …]` markdown header makes the source of the message unambiguous to the model.

## Digest truncation: proportional over LLM summarisation

When collected pieces exceed the 6 000-token digest budget we proportionally truncate each piece rather than making additional LLM summarisation calls. Considered calling the LLM to summarise oversized pieces, but: (1) with typical per-piece caps (300 chars for thinking excerpts, 800 chars for responses, compressed stubs for tools) a 20-iteration session uses ~4 250 tokens — well under budget without any extra calls; (2) each extra LLM call adds latency and a failure mode. Proportional truncation is synchronous and never fails. LLM summarisation can be layered in later if real sessions consistently exceed the budget.

## Tree insertion updates `parent_id` of the first live message

Rather than re-parenting a subtree of messages, only the single `parent_id` of the first live message (the last user message) is updated to point to the new `context_summary` message. The `context_summary` message itself gets `parent_id` = the last covered message. This keeps the mutation minimal: one insert + one parent update, and the active branch remains structurally valid.

## Disabled by default

Stage 3 runs independently of Stages 1/2 and is controlled by a single toggle. It is left off until validated against a real long conversation, so that Stage 1/2 behaviour is never affected by an untested path.
