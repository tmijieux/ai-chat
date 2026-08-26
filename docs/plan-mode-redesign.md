# Plan Mode Redesign — Design Exploration

Status: **exploratory, not decided.** This captures an in-progress design discussion, not an
architecture to build. Follow-up ADRs should be split out per decision once each part actually
settles — do not treat this file as a spec. Despite the title, the discussion outgrew "Plan mode"
specifically — see the unifying thesis below; Plan mode (violet/blue) is one instance of it, not
the whole idea.

## The unifying thesis: hierarchical context management, not blanket auto-compaction

Every idea in this document — colors, barriers, skills — is really one underlying goal: get tasks
done that need far more total tokens than the 32K context limit, by managing context deliberately
at meaningful semantic boundaries, instead of relying on generic periodic/threshold-based
compaction (today's Working Memory: always after a run, or at a 50-60% token threshold, or every N
iterations — see `CONTEXT.md`'s Post-Iteration Sub-Agent). That generic approach doesn't know
*why* something is or isn't still needed; it can only react to size.

**Workflows already prove the bounded/hierarchical approach works.** `sync-locale-directory`,
`translate-locale`, and `map-codebase` already run far longer and consume far more cumulative
tokens than the 32K limit, and still produce coherent, useful results — because each stage and each
loop item gets its own small, isolated `run_stage` context (never holding the whole run's history
at once), and results roll up structurally (JSONL records, folded directory summaries) instead of
accumulating in one flat, ever-growing conversation. `map-codebase`'s bottom-up depth-ordered fold
is a clean example: no single call ever sees more than one file or one directory level at a time,
yet the final output covers an entire repo.

The colors/barriers/skills ideas are an attempt to bring that same proven bounded-and-hierarchical
structuring into the more natural, conversational, model-driven experience users actually want —
rather than forcing every task that needs it into a rigid YAML workflow DAG. Where a workflow's
stage boundaries are fixed at authoring time by a human, a color/skill boundary is meant to be
something the model itself can enter, exit, and reason about mid-conversation.

## Motivation

One of the model's worst habits is jumping straight into code without designing the change with
the human first. `PipelineOrchestrator` (`agent/pipeline.py`, now deprecated in favor of YAML
workflows — see `todo.md` and ADR-0009) was an earlier attempt at forcing a classify → augment →
critique → plan → execute → verify → compile_fix structure for exactly this reason. This
exploration is a spiritual successor: make **Plan mode** the default for a new conversation, and
make the planning phase itself richer — grounded in real retrieval, allowed to genuinely converse
with the user, and structured enough that the eventual execution phase can lean on it instead of
re-discovering everything from scratch.

## Grounded facts (verified against the current code, not assumptions)

These are load-bearing for the design below — re-verify before relying on them if this doc is
read much later:

- **The main agent loop is already free-form.** `run_agent`/`chat_with_tools` (`agent/agent.py`)
  never forces `tool_choice`. The model can already respond in plain prose or call a tool at will,
  in every existing mode including Plan. Nothing new is needed to let the model "just chat" instead
  of being forced through a tool call every turn.
- **`run_stage` (`agent/pipeline.py`) is the existing forced-subagent mechanism** — used by
  `explore_codebase` and every `PipelineOrchestrator` stage. It runs in an isolated `AgentSession`
  (separate message list, separate LLM context) hidden from the user; only the finish-tool's
  structured result surfaces to the caller as one collapsed tool-result message. Forcing only
  happens mechanically on a stage's last allowed turn (grammar-constrained to the single remaining
  finish-tool schema); earlier turns are only instructed, not constrained.
- **`propose_plan` is a plain tool, not a `run_stage` finish-tool.** Its `execute()`
  (`agent/tools/propose_plan.py`) calls `session.request_plan_confirm(...)` directly — emits
  `plan_proposal`, suspends for the user's accept/feedback, and on accept emits `mode_changed`.
  It's called from the free outer loop today, in Plan mode.
- **Plan mode's tool set today** (`routers/ws.py::_build_tool_set`) is the conversation's own
  `active_tool_names` minus `{write_file, edit_file, run_shell}`, plus injected
  `ask_user_question` + `propose_plan`. It is *not* a fixed curated set independent of what the
  user enabled for the conversation.
- **Rejected-plan feedback already survives in context** — `propose_plan`'s feedback result is
  just a normal tool-result message the model sees next turn, so refining a rejected plan already
  works without new plumbing.
- **`write_file` already refuses to overwrite an existing file** (checked before the confirmation
  prompt even fires) — the problem reported ("small models think it can overwrite") is a model
  naming-intuition/prompt-comprehension issue, not a permissiveness bug.
- **There is an existing but apparently-unused declarative single-stage mechanism**:
  `backend/agents/*.yaml` + `routers/agents.py`, full CRUD (`name`, `system_prompt`, `tools`,
  `finish_tool`, `max_iterations`, `inject_turn_reminders`). `backend/agents/explore_codebase.yaml`
  mirrors `ExploreCodebaseTool`'s hardcoded Python prompt almost field-for-field, but
  `explore_codebase.py` does not actually load it — it hardcodes its own copy. Not documented in
  `CONTEXT.md`. Worth understanding before building new subagent machinery — this may already be
  scaffolding for it, abandoned mid-way, or may be dead and irrelevant. Needs investigation, not
  assumption.
- **`map-codebase` (`backend/workflows/map-codebase/`)** produces, per run:
  `{output_directory}/files.jsonl` (per-file role/exports/key_dependencies),
  `{output_directory}/directories.jsonl` (per-directory purpose/key_files/notable_dependencies),
  and `{output_directory}/overview.md` (mission/architecture/glossary candidates). Default
  `output_directory` is `docs/codebase-map`. It is a workflow the user runs explicitly — this
  design should only ever *read* its output if present, never trigger the workflow itself.
- **`FinishPlan` (`agent/finish_tools.py`) already has a "one file per task" rule** — but it's
  enforced only via prompt instruction over a free-text `description` field. There is no structural
  `file_path` field on a task today, so nothing downstream can read it without re-parsing prose.
- **The think-gated two-phase generation mechanism already exists**
  (`llm/llama_server.py::_stream_completion_think_gated`, `_force_tool_call`). Phase 1 runs
  unconstrained thinking and hard-stops at `</think>`; phase 2 replays the same KV cache
  (`cache_prompt`) against the raw `/completion` endpoint — not the chat-completion endpoint — with
  either a grammar or, potentially, literal prefilled text. This is real, working infrastructure,
  not something to build from scratch.
- **Auto mode's always-safe tool allowlist** (`agent/auto_safety.py::_ALWAYS_SAFE_TOOLS`) already
  exists as the place read-only tools get registered as auto-approved.

## Why this needs so much engineering (meta-note)

A frontier model with a large context window doesn't need engineered phase boundaries: it can hold
more at once, judge relevance itself, and fall back on an occasional wholesale compaction instead
of many small hand-built ones. This whole color/barrier apparatus exists specifically to compensate
for what a 32K-context small local model cannot reliably do on its own. Worth remembering as a
sanity check on scope: any piece of this that a bigger/smarter local model would eventually make
unnecessary is a piece worth keeping simple rather than over-engineering now.

## Emerging design: modes as "colors" with entry/exit rules

Working metaphor (not final terminology): a conversation moves through colored zones, each with
its own tool set / system prompt, entry conditions, a possible forced kickoff action, and an exit
rule governing what context survives into the next zone. This generalizes today's flat
`standard/auto/plan/yolo` mode switch (currently just a tool-set filter,
`routers/ws.py::_build_tool_set`) into something with actual entry/exit hooks.

Concretely discussed so far:

- **Green (initial/default)** — today's Standard-shaped mode, but Plan is intended to become the
  *default* starting color for a new conversation (currently the frontend hardcodes
  `mode: 'standard'` in two spots in `chat.service.ts`).
- **Violet (planning session)** — entered one of two ways: (a) automatically, via a classifier on
  the conversation's initial message (same idea as `PipelineOrchestrator`'s existing
  `classify`/`FinishClassify` stage — simple vs. complex), where a complex-classified message
  immediately becomes the query that kicks off violet; or (b) manually, via the existing `/plan`
  slash command (`CONTEXT.md`'s Slash Command Palette already has `/plan`, but it currently just
  flips `ConversationSettings.mode` with no argument) extended to take an optional query argument,
  usable both to enter immediately and to re-enter violet later in an already-running conversation
  — entry is not a one-time start-of-conversation event only. Entry forces a deterministic kickoff:
  `rag_search` against the workspace's RAG space, plus a read of pre-existing `map-codebase` output
  if it happens to already exist on disk (never triggers that workflow itself). After the forced
  kickoff, violet continues as an ordinary free agent turn — same conversation, same visible
  transcript — with a tool set along the lines of `read_file_range`, `grep_files`, `glob_files`,
  `list_directory`, `search_web`, `rag_search`, `ask_user_question`, `propose_plan`. The model can
  converse, ask questions, explore more, or propose a plan whenever it judges it has enough.
- **Blue (execution)** — entered when `propose_plan` is accepted. Gets `edit_file` and a
  write-tool (see naming note below), plus whatever the conversation's own `active_tool_names`
  already grant (this part already matches how `chosen_mode` works today — accepting a plan already
  switches to the user's chosen full mode).

### The "surfaced subagent" idea (superseded understanding — see below)

Initially explored: run exploration as an isolated `run_stage`-style hidden subagent (matching
today's `explore_codebase`). Corrected mid-discussion: the intent is closer to **no isolation at
all** — exploration happens as real, visible turns in the same conversation and the same
`AgentSession`, not a hidden nested LLM call under a different system prompt. What today's
`run_stage` calls a "stage" would instead just be a phase of the *same* session, distinguished by
which tools/system-prompt are active, with a phase-ending ("barrier") tool call.

**The barrier**: when a phase concludes (e.g. a `finish_explore`-shaped signal, or `propose_plan`
being accepted), the just-finished phase's transcript gets reorganized — not simply left as raw
multi-turn noise, and not simply thrown away. Some structured result (analogous to today's
`ExploreCodebaseResult` snippets/summary, or the accepted plan text) plus specifically the tool
results still judged useful survive, marked to stay in context; everything else from that phase is
excluded (`context_excluded`-style, same flag Working Memory already uses) rather than kept
verbatim forever. This is explicitly **not** assumed to reuse Working Memory's existing 3-pass
compression as-is — it may need its own barrier-specific logic — but it may also turn out to want
the same `context_excluded` plumbing. Not decided. Do not build the "no isolated session, phases
share the main transcript" mechanism yet — this needs more design work and possibly a prototype
before committing; it's a materially different shape from every existing subagent mechanism in the
codebase (`run_stage`, `CustomWorkflowOrchestrator`, `SubAgentTool`) and changing it has wide blast
radius.

### Skills — a third construct, model-as-framework instead of code-as-framework

`CONTEXT.md`'s existing "Workflow/skills" glossary entry already contrasts workflows with "simple
skills (prompt injection)," but today that's just a passing phrase — there is no distinct Skill
construct in the app. This discussion gives it real shape:

- **Workflow**: code is the framework. `CustomWorkflowOrchestrator` deterministically drives stage
  sequence, tool availability, and forced finish tools per stage — the model has no say over what
  happens next structurally.
- **Skill**: the model is the framework. Loaded instructions plus a scoped tool set, run inside an
  ordinary free conversational turn (same shape as Claude Code's own skills) — the model itself
  decides what to do next, not an external orchestrator.

A skill invocation is its own barrier-bounded phase, but unlike green/violet/blue (a small fixed
palette), **each skill instance is its own separate, ad-hoc phase** — there can be arbitrarily many
over a conversation's lifetime, one per invocation. It stays visible/natural in the transcript, same
as violet's exploration turns — never a hidden `run_stage`-style call. The barrier here is
explicitly **asymmetric**: not only might what happened *inside* the skill fail to fully leak
forward afterward (same idea as violet→blue), but what feeds *into* the skill on entry also isn't
necessarily the whole preceding conversation — a skill instance may start from a curated/filtered
slice of prior context rather than everything said so far.

### Signatures — named arguments for skills/agents/workflows, and richer keyboard-driven entry

Today every slash command (`/plan`, `/rag-search`, a workflow name) and every subagent invocation
takes exactly one opaque free-text string (`chat-client/.../slash-command-palette.component.ts`'s
`paramHint`, `ChatService.startAgentRun`/`runRagCommand`). Separately, `AgentDefinition.input_schema`
(`agent/workflow_loader.py`) already exists as a typed `{name: {type, description}}` convention
parsed from agent YAMLs' `input:` field — but nothing downstream reads it. `WorkflowDefinition` has
no equivalent field at all.

Working name for this concept: a **signature** — the named, typed parameters a skill/agent/workflow
declares it accepts, in the same sense as a function signature. Proposed direction:

- Extend the existing `input_schema` convention onto `WorkflowDefinition` too, so agents, skills, and
  workflows all declare arguments the same way instead of each having their own ad-hoc single-string
  convention.
- Slash-command syntax becomes `name:value` pairs (quoted for spaces), backward-compatible: if a
  signature has exactly one field, a bare string with no `name:` still maps to it, so existing
  single-arg commands (`/rag-search foo bar`) keep working unchanged.
- Fields left unset are not errors — they mean "use default," same as leaving a function argument
  unpassed.

**Richer input UX**, explicitly keyboard-first (mouse/click-driven interaction — drag-to-reorder,
right-click, hover-preview panes — is out of scope for now):

- The existing `@`-mention picker (`FileMentionPickerComponent`) and `SlashCommandPalette` are
  already proof that an overlay-driven, trigger-token → filtered-list → keyboard-select pattern works
  in this input. Generalize it: once a signature-bearing command is selected, each field can offer
  its own picker of the same shape, keyed off the field's declared type (a `file`-typed field reuses
  `FileMentionPickerComponent` directly; an `enum`-typed field gets a small static-list variant;
  plain `string`/`number` stays inline text).
- The signature's field hints should render *in* the input bar once a command is selected — not as
  literal prefilled text the user has to delete/type over, but as visually distinct ghost
  placeholders layered over the real value, replaced once a field is actually filled.
- Keyboard navigation between fields: `Tab`/`Shift+Tab` to the next/previous empty field, plus a
  vimium-style jump — a modifier key reveals a short label on every still-empty field, pressing that
  label jumps straight there — so filling several named fields never requires reaching for the mouse.
- Wilder, unresolved: nested composition — a field whose value is itself another command's
  invocation (e.g. passing `/explore_codebase`'s result as an argument to another command),
  collapsed to a sub-chip that can be expanded/edited in place. Not designed, flagged only as a
  direction worth keeping in mind so the field/picker model doesn't accidentally foreclose it.

**Natural-language entry stays a valid alternative, not something the signature/picker idea
replaces.** Workflows already have a proven pattern of a first LLM-generation stage that parses free
text into the structured fields a later stage needs. That path should coexist with structured
field-by-field entry, not be displaced by it — a user who'd rather just type/speak a sentence should
be able to, with an LLM stage doing the same parsing-into-signature job the picker UI does
interactively. Same signature, two ways to fill it in.

### Plan structure

The plan should stop being pure free text. Concretely proposed: `propose_plan`'s task-like
structure should carry, per task, an explicit `file_path` field (alongside `id`, `description`,
`verification_method` — mirroring `FinishPlan`'s existing shape in `pipeline.py`, but making the
existing "one file per task" *prompt* rule a real structural field instead of prose discipline).

### Execution phase (blue) — open, contested ideas

Empirically, giving a task-execution turn *only* its own task description (no surrounding plan
context) produced worse results — the model didn't understand the larger purpose and executed its
slice badly. Current leaning: the execution turn should see the **whole plan** and the **whole
context gathered during planning**, with a per-task marker (current task highlighted, done tasks
marked done, upcoming tasks visible but marked "not yet"), and rely on prompting discipline to keep
it from touching more than its assigned task — not on hiding information from it.

Wilder, explicitly unresolved ideas floated in the same breath:

- Forbidding all read/explore tools during a task's execution turn, allowing only `edit_file` —
  flagged by the user themselves as risky ("what if it needs to look something up and can't").
- **Prefill-based argument forcing**: once a task's `file_path` is a real structured field (see
  above), it's known *before generation starts* — no need to infer it from the model's own
  thinking. The existing raw-`/completion`-endpoint two-phase mechanism
  (`_stream_completion_think_gated` / `_force_tool_call`) already assembles a literal prompt string
  and replays it past `</think>` — instead of (or in addition to) a grammar, that prompt could have
  the tool-call's `file_path` argument already spelled out as literal prefilled text, letting
  generation continue unconstrained for the parts that actually need the model's judgment (the edit
  content itself). This sidesteps needing a grammar that pins a literal string value, and sidesteps
  parsing thinking text for a guessed file path — because the file path is already known data, not
  something extracted from generation.

All of the above execution-phase ideas are explicitly "not sure, still trying things" — the user
was explicit that small local models may simply not be able to reliably pull this off, and that
this is worth attempting anyway since small models keep improving.

### Naming: `write_file` → clearer name

Models keep misreading `write_file` as capable of overwriting, despite the tool already refusing to
touch an existing file. Discussed direction: rename to something like `create_new_file` for
clarity. Scope if pursued: `agent/tools/write_file.py`, `tool_result_types.py`,
`agent/auto_safety.py::_FILE_WRITE_TOOLS`, `routers/ws.py::_PLAN_EXCLUDED_TOOLS`,
`agent/tools/__init__.py`, `chat-client/.../tool-result.component.ts` (`r.tool === 'write_file'`
checks), `CONTEXT.md`'s Tool Result Display section, and every workflow YAML referencing
`write_file` by name (`map-codebase`, `translate-locale`, `coding.yaml`, `create-workflow.yaml`).
Not yet decided whether this is worth the blast radius versus just improving the tool's
description text first and re-testing.

## Open questions (explicitly unresolved)

- The classifier-based auto-entry path needs its own behavior spec: same prompt/shape as
  `FinishClassify`'s simple/complex split, or a new dedicated classifier tuned for "does this want
  planning"? What happens on a borderline/wrong classification — can the user easily back out of
  violet if it triggered unwantedly?
- Does the barrier/phase-surfacing idea replace `run_stage` generally (affecting `pipeline.py` and
  any future subagent-shaped tool), or is it scoped specifically to this Plan-mode redesign,
  leaving `run_stage`'s isolated-hidden-subagent pattern intact elsewhere?
- How is a skill instance actually invoked and bounded — a slash command like workflows, a tool
  call the model makes on its own, something else? What marks its start and its end?
- Who/what decides the curated slice of prior context that feeds *into* a skill on entry? Is that
  filtering itself model-driven (consistent with "skill = model is the framework"), or does the
  surrounding framework still make that one call?
- Is `backend/agents/*.yaml` + `routers/agents.py` a natural fit for authoring skills specifically
  (as opposed to violet/blue's larger, more special-cased phases), given it's already a
  declarative single-stage definition format sitting mostly unused?
- Is `backend/agents/*.yaml` + `routers/agents.py` meant to become the actual authoring mechanism
  for phases/subagents here, or is it unrelated scaffolding that predates this idea and should stay
  untouched?
- Exact re-fetch/context margin when crossing the plan→execution barrier (how much surrounding code
  per referenced file, if the isolated-session model is kept after all).
- Whether `map-codebase` output should only be read opportunistically (if present) or should also
  get folded into the RAG space itself as indexable sources.
- Whether the four existing modes (`standard/auto/plan/yolo`) get subsumed into the color/phase
  model or continue to coexist alongside it.

## Related existing material

- ADR-0006 — Post-Iteration Sub-Agent / compression passes (`context_excluded`, Working Memory).
- ADR-0009 — Workflow Run View (motivates retiring `PipelineOrchestrator` in favor of YAML
  workflows).
- ADR-0011 / ADR-0012 — resumable workflow runs.
- ADR-0015/0016/0018 — RAG infrastructure, slash commands, embedding model choice.
- ADR-0019 — `rag_search` agent tool (the one piece of this broader idea already shipped).
- `docs/compression-research.md` — prior research doc in the same spirit as this one, for the
  compression/Working Memory system.
- `CONTEXT.md`: Conversation Mode, Working Memory, Context Eviction, Workflow/skills, RAG Space.
- `todo.md`: "Retire `PipelineOrchestrator`" item.
