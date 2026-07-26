# Compression Research

## Test Scenarios

Three concrete scenarios that drive the working memory / compression redesign. Use these to evaluate whether any new architecture works end-to-end.

### SHORT — SQL anonymization

User sends a SQL query, agent produces anonymized SQL (no tools), user says "nice work", compression runs.

**Context usage:** ~10% after congratulations.

**Failure mode:** verbatim SQL query and accepted output are paraphrased away. If user asks to refine with an extra condition, agent has lost the artifact.

**Requirement:** input data payload and accepted answer survive compression verbatim.

---

### MEDIUM — stats script

User asks for a JSONL stats extractor. Agent writes and runs it over several iterations. User refines:
- add CLI switching between stats modes
- add descriptive stats: p10, p90, median, mean

Multiple agent iterations. May hit context limit after several exchanges, often completed successfully by the agent.

**Failure mode:** agent loses track of current script state, implemented stats, CLI flags across compression cycles.

**Requirement:** artifact state (file path, what's built, what remains) survives multiple compression cycles without drift.

---

### HARD — architecture refactoring design

User asks about a difficult refactoring problem: paradigm shift from frontend to backend of some APIs for performance reasons. Agent must help design how to call the new API and when to refresh data.

The long run is tool-heavy: the agent must explore a large codebase extensively (many reads, greps, globs) just to understand the problem well enough to reason about it. The context fills during this exploration phase — before the agent has even proposed anything.

**Context usage:** fills or almost fills 32k on the first run due to exploration volume.

**Failure mode:** compression fires mid-exploration. Agent loses track of what it already read, re-explores files it already saw, or loses the thread of why it was reading them.

**Requirement:** after compression, agent knows what it already explored, what it learned from each file, and what question it was still trying to answer. Must not restart from scratch.

---

## Compression Pipeline Design

Compression is **context management for performance**, not just a safety valve for overflow. Post-run compression always runs regardless of context size. The mid-run threshold moves from overflow to 50-60% of CTX_LIMIT (32768) to leave headroom for the compression LLM calls themselves.

### On first user message

- Classify intent: complex task vs casual small talk
- Generate a conversation summary → feeds into conversation title

### On each compression trigger

Three sequential passes, operating on the delta since the last working memory message (first compression sees the full conversation):

**Pass 1 — Tool result classifier**
Classifies each tool result message as `drop`, `1-line-summary`, `summarize`, or `keep`. Tools `write_file`, `edit_file`, `ask_user_question`, and `propose_plan` are excluded from classification. User and assistant messages are included in the classifier prompt as context alongside the tool results to improve signal quality.

**Pass 2 — User+assistant message classifier**
Looks at all user and assistant messages at once. Can use surrounding context:
- "nice work" after an assistant response → tag that response as `accepted`, drop the "nice work"
- A later refinement that supersedes an earlier answer → tag the earlier answer as `stale`
- Stale messages are dropped; a brief mention goes into working memory (e.g. "first version before CLI refactoring — superseded")

**Pass 3 — Working memory writer**
Focused on:
- What didn't work: rejected tools, failed approaches, dead ends
- Current accepted answer state (from Pass 2 tagging)
- Files explored but deemed not useful

Previous working memory is folded forward into the new one on each subsequent compression.
