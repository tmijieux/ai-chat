PLAN THOUROUGHLY even when not asked. Create PRDs and ADR to remember details of plan

DONT READ catchall.py it currently unused and irrelevant.

ALWAYS READ `CONTEXT.md` at the start of every conversation. It contains the app mission, domain glossary, feature intent, known bugs, and planned changes. Features documented there must not be removed or broken when implementing other features.

# AI Chat Application

A local AI chat app using Ollama, with an Angular frontend and FastAPI backend. Includes an autonomous coding agent subsystem.

## Architecture

```
├── backend/
│   ├── main.py                    # FastAPI app, all API endpoints + WebSocket agent
│   ├── database.py                # SQLAlchemy async session setup
│   ├── tables.py                  # ORM models (Conversation, Message, SystemPromptTemplate)
│   ├── loaders.py                 # Pydantic request/response schemas
│   ├── requirements.txt
│   └── agent/
│       ├── agent.py               # Tool-calling agent loop (Ollama-backed, AgentSession)
│       ├── file_utils.py          # Path security checks
│       ├── count_token.py         # Token counting: fires 1-token Ollama call, reads prompt_eval_count
│       ├── count_tool_tokens.py   # Dev script: measures per-tool token cost via Ollama (see below)
│       └── tools/
│           ├── __init__.py        # TOOL_REGISTRY dict + get_ollama_tool_list()
│           ├── base.py            # BaseTool ABC, tool_error(), TOOL_FRAMEWORK_OVERHEAD
│           ├── list_directory.py  # ListDirectoryTool  (measured_delta=375)
│           ├── glob_files.py      # GlobFilesTool      (measured_delta=344)
│           ├── grep_files.py      # GrepFilesTool      (measured_delta=446)
│           ├── read_file.py       # ReadFileTool       (measured_delta=343)
│           ├── write_file.py      # WriteFileTool      (measured_delta=360, confirm)
│           ├── edit_file.py       # EditFileTool       (measured_delta=413, confirm)
│           ├── run_shell.py       # RunShellTool       (measured_delta=316, confirm)
│           ├── search_web.py      # SearchWebTool      (measured_delta=282)
│           └── summarize_subtask.py # SummarizeSubtaskTool (measured_delta=336)
│
├── chat-client/                   # Angular 20 frontend
│   └── src/
│       ├── app/app.ts             # Root component, routing config
│       ├── types/message-types.ts # TypeScript types (Message, Conversation, AgentEvent, etc.)
│       ├── services/
│       │   ├── chat.service.ts    # REST calls, conversation/message state, streaming
│       │   └── agent.service.ts   # WebSocket agent client, AgentUiMessage signals
│       └── components/chat/
│           ├── chat.component.ts
│           └── chat.component.html
│
├── .env                           # PYTHONPATH=backend
└── backend/chat_db.sqlite         # SQLite DB (auto-created)
```

## Backend API

All endpoints at `http://localhost:8000`.

### Conversations
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/conversations` | List all conversations |
| POST | `/api/conversations` | Create conversation (`{title: string}`) |
| GET | `/api/conversations/{id}/messages` | Fetch messages for the active branch (root→leaf order) |
| DELETE | `/api/conversations/{id}` | Delete conversation and all its messages |
| PUT | `/api/conversations/{id}/settings` | Update per-conversation settings (JSON: ConversationSettings) |
| PUT | `/api/conversations/{id}/active-branch` | Switch branch; auto-advances to deepest single-child leaf |
| GET | `/api/conversations/{id}/tree` | Full message tree with sibling counts |

### Messages
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/messages?conversationId={id}` | Append a message; uses active_message_id as parent if parent_id omitted |
| PUT | `/api/messages/{id}/branch` | Create sibling with new content (branch), update active branch |
| DELETE | `/api/conversations/{conv_id}/messages/{msg_id}` | Delete message subtree; adjusts active_message_id |

### System Prompts
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/system-prompts` | List all system prompt templates |
| POST | `/api/system-prompts` | Create new prompt template |
| PUT | `/api/system-prompts/{id}` | Update prompt fields |
| DELETE | `/api/system-prompts/{id}` | Delete prompt template |

### Chat (streaming)
`POST /api/chat`

Request:
```json
{
  "messages": [{"role": "user|assistant|system", "content": "...", "thinking": "..."}]
}
```

Response: NDJSON stream (one JSON object per line). Final chunk includes `prompt_eval_count` and `eval_count` for token usage.

### Token Counting
`POST /api/conversations/{id}/count-tokens` — calls Ollama with `num_predict:1` to read `prompt_eval_count`, persists on the last branch message.

### Agent WebSocket
`WS /api/agent/ws`

First client message: `{"message": "...", "conversation_id": "optional"}`.
Server events: `thinking`, `content`, `tool_call`, `tool_confirm`, `tool_result`, `iteration_end`, `done`, `error`.
Client can send: `{"type":"confirm","tool_id":"...","approved":bool}` or `{"type":"abort"}`.

### Ollama Proxy
`GET|POST|PUT /{full_path}` — proxies requests directly to the Ollama API at `http://localhost:11434/`.

## Database Models

**Conversation** (`tables.py`)
- `id`: String PK
- `title`: String (indexed)
- `settings`: Text (nullable) — JSON: `{active_prompt_ids, active_tool_names, tools_enabled, agentic_mode}`
- `created_at`: String (ISO timestamp)
- `active_message_id`: String (nullable) — logical FK to `messages.id`

**Message** (`tables.py`)
- `id`: String PK
- `conversation_id`: FK → Conversation
- `parent_id`: FK → messages.id (nullable — null for conversation root)
- `role`: String — `user`, `assistant`, `system`, or `tool`
- `content`: Text
- `thinking`: Text (model reasoning, nullable)
- `created_at`: String (ISO timestamp)
- `token_count`: Integer (nullable) — cumulative context size at this message

**SystemPromptTemplate** (`tables.py`)
- `id`: String PK
- `name`: String
- `category`: String — `general | code | summarization | context_compaction | state_storage`
- `content`: Text
- `is_global`: Integer (1=true, 0=false)
- `created_at`: String (ISO timestamp)

## Frontend (Angular 20)

- **Zoneless** change detection (`provideZonelessChangeDetection()`)
- Standalone components, no NgModules
- RxJS `BehaviorSubject` + Angular `signal` for state
- Custom utility CSS in `chat-client/src/styles.scss` (Tailwind-like naming, no Tailwind dependency — see CONTEXT.md for conventions)

### Key state in `chat.service.ts`
- `history$` (BehaviorSubject) — current conversation message list (active branch)
- `isLoading$` (BehaviorSubject) — whether a non-agentic response is in-flight
- `contextTokens$` (BehaviorSubject) — latest cumulative token count
- `_conversationId` (signal) — currently selected conversation ID
- `_conversationSettings` (signal) — `{agentic_mode, ...}` from DB or pending

### Key state in `agent.service.ts`
- `messages` (signal) — `AgentUiMessage[]` for the current live agent run
- `running` (signal) — whether a WS run is active
- `promptTokens` (signal) — prompt_tokens from the latest `iteration_end`
- `done$` (Subject) — fires when agent finishes (triggers `saveAgentMessages`)
- `phase` (computed) — `'thinking' | 'responding' | null`

### Streaming (non-agentic)
`sendMessage()` uses `HttpClient` with `observe: 'events'` + `reportProgress: true` to receive the NDJSON stream. Chunks are parsed and accumulated into the assistant message in real-time. Thinking and content are tracked separately.

### Agentic flow
1. `chatSvc.prepareAgentConversation(input)` saves the user message to DB.
2. `agentSvc.startWithUserMessage(input, convId)` opens a WebSocket.
3. Backend `agent_websocket` builds context from DB history, runs `run_agent`.
4. `iteration_end` from backend → frontend updates `promptTokens`; backend also persists `token_count` on the last DB user message (iteration 1 only).
5. On `done`, frontend calls `saveAgentMessages()` to persist all agentic messages to DB (content, tool results; skips iteration_end/user); tool result messages get `token_count` from the following iteration's `prompt_tokens`.
6. Then `countTokensForCurrentConversation()` is called (redundant but harmless).

## Token Counting

Token counting uses an Ollama `num_predict:1` trick (`agent/count_token.py`): send the full message list and read `prompt_eval_count` from the response — no real tokens generated. The tool-list schema is included in every count call (matching what the agent actually sends).

Per-message `token_count` stores the **cumulative context size** at that message:
- Non-agentic: fired via `POST /api/conversations/{id}/count-tokens` after each assistant response.
- Agentic: user message gets `prompt_tokens` from iteration 1 (persisted by backend on `iteration_end`); tool result messages get `prompt_tokens` from the next iteration (assigned by frontend in `saveAgentMessages`).
- Displayed as a hover tooltip (ⓘ) on each message: cumulative count, % of 16 384, and delta vs. previous counted message.

## Agent Subsystem (`backend/agent/`)

An autonomous coding agent that runs a tool-calling loop against Ollama.

### Tools
| Tool | Description |
|------|-------------|
| `list_directory` | Browse filesystem (with depth control) |
| `glob_files` | Find files by glob pattern |
| `grep_files` | Search file contents with regex |
| `read_file` | Read file contents (optional tail-N lines) |
| `write_file` | Create/overwrite files (user confirmation required) |
| `edit_file` | Replace text in a file (user confirmation required) |
| `run_shell` | Execute shell commands (user confirmation required) |
| `search_web` | DuckDuckGo search + content extraction via Trafilatura |
| `summarize_subtask` | Compress large content via a fresh LLM call |

File operations are validated against the working directory (`file_utils.py`) to prevent path traversal.

### Agent loop (`agent.py`)
`AgentSession` manages bidirectional comms (outbound queue + confirm futures). `chat_with_tools` does one Ollama call, streams thinking/content, executes tool calls, feeds results back. `run_agent` loops until no tool calls are emitted, then emits `done`.

### Tool token cost (`count_tool_tokens.py`)

Every tool class carries a `measured_delta` field — the empirically measured `prompt_eval_count` increase when that tool is included (1 tool, dummy message, no system prompt).

**Formula:**
```
total_tool_tokens = TOOL_FRAMEWORK_OVERHEAD + sum(t.token_count for enabled tools)
t.token_count     = t.measured_delta - TOOL_FRAMEWORK_OVERHEAD
TOOL_FRAMEWORK_OVERHEAD = 223   # defined in tools/base.py
```

**When to re-run:** any time you add a new tool or change an existing tool's `name`, `description`, or `parameters`. Run from `backend/`:
```bash
python -m agent.count_tool_tokens
```
The script prints each tool's total count and delta, then verifies `token_count` matches the measured value. Copy the printed `measured_delta` back into the tool class.

**Rule:** always update `measured_delta` in the tool class after modifying its schema.

## Known Gaps / TODOs

- `countTokensForCurrentConversation` still fires after agentic runs (line ~65 in `chat.component.ts`) — redundant since per-message token counts are accurate; harmless.
- `_ws_send_events` helper in `main.py` (line ~530) is unused — superseded by `send_events_with_token_update`; can be deleted.
- Final assistant message (no tool calls) from the agentic run does not yet get a `token_count` from the `iteration_end` path.
- Thinking-only assistant messages (content = "") are still saved to DB by `saveAgentMessages` — correctly filtered from Ollama context but add noise.
- Phase 3 (system prompt & tool library UI) is next — see Project Handoff document.

## Development

### Backend
```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # Git Bash on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd chat-client
npm start
```

Requires Ollama running with a model loaded (default: `qwen3.5:9b`).

## Configuration

`.env` sets `PYTHONPATH=backend`. Ollama URL and model are hardcoded in `main.py` and `agent/agent.py`.

Database: `backend/chat_db.sqlite` (created automatically on first run).

CORS is configured for `http://localhost:4200` (Angular dev server).
