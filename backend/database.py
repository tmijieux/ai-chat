from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event
import sqlalchemy.ext.asyncio
async_sessionmaker = sqlalchemy.ext.asyncio.async_sessionmaker

# SQLite connection string
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./chat_db.sqlite"

# Asynchronous engine setup
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,
    connect_args={"check_same_thread": False, "timeout": 30},
)

@event.listens_for(engine.sync_engine, "connect")
def set_wal_mode(dbapi_connection, connection_record):
    dbapi_connection.execute("PRAGMA journal_mode=WAL")

# --- 2. Base Definition (The core fix) ---
class Base(DeclarativeBase):
    """Base class which provides automated table name and declarative attributes."""
    pass

# Create the session factory
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Function to create tables in the database
async def init_db():
    """Creates all defined tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Helper function to get an async session
async def get_db_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise