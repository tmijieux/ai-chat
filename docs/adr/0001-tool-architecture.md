# ADR-0001: One file per tool with a shared base class

**Date:** 2026-05-20  
**Status:** Accepted

## Context

Agent tools were split across two files: `tools_definition.py` (JSON schema) and `tool_implementation.py` (execution handlers). Adding or modifying a tool required edits in two places, and the frontend had no way to fetch tool metadata from the backend.

## Decision

Refactor to one Python file per tool under `backend/agent/tools/`. Each file subclasses a common `BaseTool` abstract class that requires:

- `name: str`
- `description: str`
- `parameters: dict` (JSON schema)
- `requires_confirmation: bool`
- `validate(args) → str` — validates arguments and returns a human-readable preview for the confirmation UI
- `execute(args) → str` — performs the actual operation

Tools are explicitly registered by importing them in `backend/agent/tools/__init__.py`. A new `GET /api/agent/tools` endpoint serializes the registered tools for the frontend.

## Alternatives considered

**Auto-discovery by glob** — scan `tools/*.py` and import subclasses automatically. Rejected: silent failures when a tool fails to load are hard to debug, and explicit imports keep the full tool inventory visible in one place.

**Keep the split** — continue with `tools_definition.py` + `tool_implementation.py`. Rejected: adding a tool requires edits in two files with no enforced contract between them.

## Consequences

- Adding a tool = one new file + one import line in `__init__.py`.
- `validate` / `execute` split enables the confirmation step to slide in between without duplicating logic.
- `GET /api/agent/tools` gives the frontend a single source of truth for available tool names and metadata.
