# AI Chat Application with Token Counting

## Project Overview

An AI chat application built with:
- **Frontend**: Angular 20 (zoneless, standalone components)
- **Backend**: FastAPI with SQLAlchemy (SQLite)
- **AI Model**: Ollama API integration
- **Token Counting**: Hybrid approach using llama.cpp/python-tokenizers

## Architecture

```
├── backend/           # FastAPI backend
│   ├── main.py        # API endpoints
│   ├── database.py    # SQLAlchemy async session setup
│   ├── models.py      # Conversation/Message ORM models
│   └── requirements.txt
│
├── chat-client/       # Angular frontend
│   └── src/
│       ├── components/chat/
│       ├── services/
│       └── message-types.ts
└── .env               # Environment variables (API keys, etc.)
```

## Token Counting Implementation (Approach 2 - Hybrid)

### Design Decision
- **Initial context tokens**: Backend calculates before streaming
- **Streaming tokens**: Frontend tracks via response stream
- **Total/current tokens**: Optional endpoint uses llama.cpp/tokenizers for accurate count

### Frontend State Flow
1. `contextTokens$` BehaviorSubject exposes current token count
2. On message send: initial tokens set, streaming updates history
3. Token counter updates as messages accumulate

### Backend Endpoints
- `GET /api/conversations` - List all conversations
- `GET /api/conversations/create` - Create new conversation
- `POST /api/chat` - Chat with streaming, returns token info
- `POST /api/token-count` - Get current total token count (hybrid)

## Tech Stack

### Backend (FastAPI)
- `fastapi` - Web framework
- `sqlalchemy` - ORM with async support
- `uvicorn[standard]` - ASGI server
- `pydantic` - Data validation

### Frontend (Angular 20)
- Standalone components (non-ngm modules)
- Zoneless change detection
- RxJS Observables for state management
- TailwindCSS for styling

## File Structure

### Backend
- `database.py:22-38` - Async session setup and helper functions
- `models.py:9-28` - Conversation and Message ORM models
- `main.py:35-78` - API endpoints (conversations, chat, token-count)

### Frontend
- `chat-client/src/message-types.ts:1-20` - TypeScript interfaces (Message, StreamChunk, ChatResponse)
- `chat-client/src/services/chat.service.ts:1-113` - State management with BehaviorSubject
- `chat-client/src/components/chat/chat.component.ts:1-81` - Chat component logic
- `chat-client/src/components/chat/chat.component.html:1-86` - Chat UI template

## Key Patterns

### State Management
- Service-level state using `BehaviorSubject`
- Frontend subscribes to `history$`, `contextTokens$`, `isLoading$`
- New chat resets state via `startNewChat(title)`

### Token Counting
- Backend calculates context tokens on chat endpoint
- Frontend Observable tracks cumulative token count
- Optional dedicated endpoint for current token count (llama.cpp integration)

## Environment Variables (`.env`)
```
OLLAMA_BASE_URL=http://localhost:11434
TOKENIZER_MODULE=llama.cpp  # or python-tokenizers
SQLALCHEMY_DATABASE_URL=sqlite+aiosqlite:///./chat_db.sqlite
```

## Development

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd chat-client
ng serve
```

## Notes
- Angular uses zoneless change detection (`provideZonelessChangeDetection()`)
- SQLite database stored in `backend/chat_db.sqlite`
- CORS configured for Angular dev server (localhost:4200)

## Implementation Status ✅

### Backend Endpoints (Complete)
- `GET /api/conversations` - Lists all conversations from DB
- `GET /api/conversations/create` - Creates new conversation
- `POST /api/chat` - Chats with Ollama, streams response, saves to DB

### Features Implemented
- Conversations persistence in SQLite
- Messages saved/retrieved from database
- Ollama API integration with streaming
- Token counting (simplified: character-based estimate)
- Error handling for Ollama API calls

### Frontend (Complete)
- `chat.service.ts` now calls real backend API via HttpClient
- Stream chunks processed and displayed in real-time
- Token count updates as chat progresses

## Running the App

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend  
```bash
cd chat-client
ng serve
```

### Environment Variables
- `OLLAMA_BASE_URL=http://localhost:11434`
- Ensure Ollama is running with a model loaded (e.g., `qwen3.5:9b` or `llama2`)
