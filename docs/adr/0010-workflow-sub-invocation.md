# ADR-0010: Workflows Can Invoke Other Workflows and Bundled Scripts

**Date:** 2026-07-28
**Status:** Accepted

The locale translation tooling was two disconnected pieces: a workflow that translates one whole file per chat invocation, and standalone CLI scripts (run by hand) that audit a file pair for missing/untranslated entries and repair corrupted keys. Building a directory-wide sync needed both — translate a delta the way the existing workflow already does, and run the same kind of audit/repair logic — but there was no way for a workflow to call another workflow, and no way for a workflow to invoke a script deterministically (only an LLM stage choosing to call a shell tool).

## Decision

Two additions to the workflow system, both deliberately generic rather than locale-specific:

1. A `workflow`-typed stage runs another named workflow definition as an isolated sub-step. It seeds the sub-workflow's own state from the caller (so the sub-workflow doesn't need to re-derive values the caller already knows) and can skip whichever of the sub-workflow's own stages that seeding makes redundant.
2. A `run_script`-style deterministic stage executes a script bundled in the calling workflow's own directory (mirroring how a workflow can already bundle its own reusable agent definitions) and captures its output, parsed as structured data when the script prints JSON.

The locale sync workflow is built entirely from these two primitives plus its own bundled scripts — no locale-specific code was added to the backend itself.

## Sub-workflows get isolated state, not shared state

A loop's inner stages share the loop's own state on purpose — later stages need to see what earlier ones in the same iteration produced. A called sub-workflow does not: it gets a fresh, empty state seeded only with what the caller explicitly passes in. Sharing the caller's full state would let the sub-workflow's internal working values collide with the caller's own (both a translation workflow and its caller can easily end up using the same generic names for "the file being processed"), and would make a sub-workflow's behavior depend on which workflow happened to call it — the opposite of a reusable building block.

## Skipping stages is explicit, not inferred

When the caller already knows a value a sub-workflow's own early stage would normally produce (asking the user which file to work on, when the caller already picked the file), that stage is bypassed entirely rather than run and its result discarded. The caller names exactly which stages to skip. An implicit rule ("skip a stage if its value is already known") was considered and rejected — it would make a sub-workflow's behavior depend on incidental naming overlaps between caller and callee, silently, instead of being visible in the calling workflow's own definition.

One consequence worth naming: in the locale sync workflow, three of the translation workflow's five stages end up skipped when it's called this way (asking which file, creating the output file, and reporting back to the user) — only the two stages that actually do the chunk-by-chunk translation run. That's the cost of reuse over duplication here; the alternative was copying those two stages' prompt directly into the calling workflow instead of referencing the existing one.

## Run-view invocation identity must be shared across the whole call tree, not per sub-run

Each stage invocation needs a run-wide-unique identity so the run view can tell repeated invocations of the same stage apart (the existing per-run counters this already relied on — see ADR-0009). A sub-workflow call constructs a fresh orchestrator for the sub-definition; if that fresh instance also started fresh identity counters, every sub-workflow call would restart numbering from the beginning, and two different calls' identically-named stages would collide under the same identity. The counters are threaded through to the sub-invocation instead of reset, so identity stays unique across the entire call tree regardless of how many workflows deep it goes. This also means the known nested-loop gap from ADR-0009 (item numbering tracked in a single slot, so a loop nested inside another loop in the *same* running workflow would overwrite its parent's item numbers) still doesn't apply here: a loop inside a called sub-workflow tracks its own item numbers on the sub-invocation's own instance, never the caller's.

## Script execution is unsandboxed, matching an existing precedent

Reading or writing a file through a workflow's regular file-manipulation stages is restricted to the configured workspace directory. Running a bundled script is not — the script receives whatever arguments the workflow gives it and can do anything a script that Python invocation is capable of, the same trust level the existing arbitrary-shell-command stage already has. That existing stage earns that trust with an interactive confirmation before every run; running a script does not ask for one, because the point is to run unattended across many files without a confirmation per file. The risk this accepts: a workflow author who lets the directory a script operates on come from unreviewed input controls where that script reads and writes. Mitigated in practice by scripts being reviewed, checked-in code (not user-supplied at request time) and by every locale sync running with the directory named directly in the invoking chat message.
