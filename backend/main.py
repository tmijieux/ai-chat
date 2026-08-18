"""FastAPI application entry point: app creation, lifespan startup, and /api/status."""
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import init_db
from llm import backend
import whisper_pipeline
from routers import conversations, prompts, agents, utils, tokens, ws, stt, token_visualizer, workflow_runs

logger = logging.getLogger(__name__)


def _disable_sqlalchemy_logging() -> None:
    """Suppress verbose SQLAlchemy engine logs."""
    for name in ["sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine", "aiosqlite"]:
        log = logging.getLogger(name)
        log.disabled = True
        log.propagate = False


_disable_sqlalchemy_logging()
logging.basicConfig(level=logging.DEBUG)


_whisper: whisper_pipeline.WhisperPipeline | None = None
_llm_ready: bool = False


def _load_whisper_bg() -> None:
    global _whisper
    try:
        _whisper = whisper_pipeline.load_pipeline()
    except Exception:
        logger.exception("Whisper pipeline failed to load — /api/transcribe will be unavailable")


def _load_llm_bg() -> None:
    global _llm_ready
    try:
        import asyncio
        asyncio.run(backend.ensure_running())
        _llm_ready = True
        logger.info("LLM backend ready.")
    except Exception:
        logger.exception("LLM backend failed to start.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("Database initialized successfully.")
    threading.Thread(target=_load_llm_bg, daemon=True).start()
    threading.Thread(target=_load_whisper_bg, daemon=True).start()
    yield


app = FastAPI(title="LLM Chat Backend", lifespan=lifespan)

app.include_router(conversations.router)
app.include_router(prompts.router)
app.include_router(agents.router)
app.include_router(utils.router)
app.include_router(tokens.router)
app.include_router(ws.router)
app.include_router(stt.router)
app.include_router(token_visualizer.router)
app.include_router(workflow_runs.router)


@app.get("/api/status")
async def get_status():
    return {"llm": _llm_ready, "whisper": _whisper is not None}


async def check_llm() -> None:
    """Raise HTTP 503 if the LLM backend is not reachable."""
    await backend.check_or_raise()
