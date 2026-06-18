## UI / UX

- **@-mention file picker in chat input**: typing `@` in the message input opens a quick picker to reference files from the project/workspace, inserting a file reference into the message (like Claude's `@` mention UI).

- **Copy raw message to clipboard**: add button to the ⋮ action menu on each message (currently only code blocks have copy via Prism)

- **Conversation title update**: compute title update from one sentence generation based on user first message
(could be done by extracting information from working memory compression!!)

## Pipeline / Agent

- **Pipeline stage event visibility**: all events tagged `_pipeline_stage` are silently dropped by the frontend (`chat.service.ts:524`) — tool calls, results, and thinking inside workflow LLM stages and sub-agents are invisible to the user. Need to design a way to surface this (collapsible stage activity section, live log panel, etc.). Also covers sub-agent visibility.

