# Domain Glossary

## Workspace
A directory path scoped to a conversation. Restricts all file-access tools (`list_directory`, `glob_files`, `grep_files`, `read_file`, `write_file`, `edit_file`) to operate within that path, and sets the CWD for `run_shell` execution. When `null`, file tools and `run_shell` are disabled entirely. Stored as `working_directory` in `ConversationSettings`.

## SystemPromptTemplate
A named, versioned system prompt stored in the DB (`system_prompt_templates` table). Has a `name`, `category`, `content`, `token_count` (computed on create/edit), and `is_default` flag. The `token_count` is evaluated by calling `count_token` with `[{"role": "system", "content": content}]`.

## is_default
Flag on `SystemPromptTemplate`. Marks the single template that is pre-selected when a new conversation is created. Not "always-on" — can be changed or cleared per conversation.

## active_prompt_id
Per-conversation setting (`ConversationSettings.active_prompt_id: string | null`). Points to the active `SystemPromptTemplate` for that conversation. `null` means no system prompt is active.

## ConversationSettings
Per-conversation configuration stored as JSON in `Conversation.settings`. Fields: `active_prompt_id`, `active_tool_names`, `agentic_mode`, `working_directory`. `active_tool_names` is the sole authority for tool availability — empty list means no tools, full list means all tools.

## Tool
A self-contained agent capability implemented as a single Python file in `backend/agent/tools/`. Each tool subclasses a common base class that defines: `name`, `description`, `parameters` (JSON schema), `requires_confirmation`, `validate(args) → preview`, and `execute(args) → result`. The `validate` phase runs before user confirmation; `execute` runs after. All tools are explicitly registered in `backend/agent/tools/__init__.py`.

## Prompt History
A git-tracked directory (`prompts/`) of `.md` files with YAML frontmatter recording past system prompt versions. Each file carries: `name`, `date`, `is_current`, `category`, `effectiveness_notes`. On backend startup, the file marked `is_current: true` is seeded into the DB as a `SystemPromptTemplate` (idempotent — skipped if a row with that `name` already exists).

## Branch
A linear path from a root message to a leaf in the conversation message tree. The active branch is determined by `Conversation.active_message_id` (always a leaf); the full path is reconstructed by walking `parent_id` chains up to the root.

## Sibling Navigator
Inline `← N/M →` arrows rendered on any message that has siblings (multiple children of the same parent). Switching siblings calls `PUT /api/conversations/{id}/active-branch` and auto-advances to the deepest leaf of the selected sibling.

## Subtree Delete
Deletes a message and all its descendants. The existing `DELETE /api/conversations/{conv_id}/messages/{msg_id}` endpoint with `?subtree=true`.

## Single-Message Delete
Deletes only one message, re-parenting its direct children to the deleted message's parent. If the deleted message was a root (`parent_id = null`), children become new roots (siblings at the top of the tree). Same endpoint with `?subtree=false`.

## Tool Result Envelope
The structured JSON dict that every tool's `execute()` returns. Contains at minimum `tool` (tool name) and any relevant paths or identifiers (e.g. `path` for file tools). Serialized to JSON by the framework layer, never by the tool itself. Allows deterministic parsing of tool-role messages by the context manager.

## Active File
The file most recently written to via `write_file` or `edit_file` in the current agent session. Identified by parsing tool-role message history. Always kept at full content in context — never compressed.

## Reference File
A file read into context but not actively being modified. Identified by parsing tool-role message history. Compressed to API surface (signatures, types, exports) by the post-iteration sub-agent when context pressure requires it.

## Post-Iteration Sub-Agent
A fixed-prompt LLM call fired by the framework after each main agent iteration (not triggered by the main agent). Receives a digest of: tools called, changes made, last user message summary, and reference file contents to compress. Returns a JSON state object (`current_goal`, `current_assumption`, etc.) and API summaries for reference files. Enables context eviction without relying on the main agent to cooperate.

## Context Eviction
Framework-level pruning of tool-role messages before the next main iteration. Two deterministic rules: (1) duplicate file reads — evict older reads of the same path, keep only the most recent; (2) reference file compression — replace full file content with API summary produced by the post-iteration sub-agent.
