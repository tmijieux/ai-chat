# ADR-0021: uv as the Python Package Manager

**Date:** 2026-08-28
**Status:** Accepted

## Context

The backend and the whisper STT tooling were each managed with a hand-maintained pair of
files: a pinned `requirements.txt` of direct dependencies and a fully-resolved
`requirements-lock.txt` produced separately. Keeping the lock file in sync was a manual
step, it was platform-specific, and the Python version was only implied by whatever
interpreter happened to create the `venv/`. The whisper side had no lock file at all — its
`requirements.txt` was five unpinned names.

## Decision

### Native uv projects (`pyproject.toml` + `uv.lock`), not `uv pip` over the old files

Both `backend/` and `whisper/` become uv projects. Dependencies live in `pyproject.toml`,
resolution is captured in a committed cross-platform `uv.lock`, and `uv sync` is the install
flow. `uv run` is how Python is invoked. The old `requirements*.txt` files are removed.

### Two separate projects, not one

`backend` and `whisper` cannot share an environment — they need conflicting major versions of
`transformers` (5.x vs 4.x) and pull in disjoint heavy stacks (fastembed/ONNX vs torch/optimum).
Each keeps its own `pyproject.toml`, `uv.lock`, and `.venv`.

### Applications, not packages

Both set `[tool.uv] package = false`. Neither is an importable distribution — they are flat
collections of modules run in place — so there is no build backend and no src layout to
maintain.

### Exact `==` pins kept in the manifest

The direct dependencies stay pinned to exact versions in `pyproject.toml` rather than being
loosened to `>=` floors. `uv.lock` already guarantees reproducibility; keeping the hard pins in
the manifest preserves the existing "nothing moves unless I move it" posture and keeps the
manifest readable as the actual set of versions in use. Whisper's previously-unpinned direct
deps were pinned to their installed versions during the migration.

### Environment directory renamed `venv/` → `.venv/`

uv's default. The three references to the old path (`package.json` scripts, `CLAUDE.md`,
the `recount-tool-tokens` skill) were updated to `uv run`, and `.venv/` was added to
`.gitignore`.

## Consequences

- Adding/removing deps is `uv add` / `uv remove`; `uv.lock` must never be hand-edited.
- Transitive versions in whisper's fresh resolution drift from whatever was in its old venv
  (e.g. torch, scikit-learn) — acceptable since those were entirely unpinned before.
- A future from-scratch installation guide (`todo.md`) should be written against uv.

## Verification

`uv sync` in each project produced a clean `.venv`; `uv run python` import smoke tests pass
for the backend (`serve`, `fastembed`, `tiktoken`, `gguf`, `trafilatura`, `openvino`) and for
whisper (`optimum.intel.OVModelForSpeechSeq2Seq`, `openvino`, `sounddevice`, `soundfile`).
