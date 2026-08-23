from sqlalchemy import Integer, String, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column as column
from sqlalchemy.types import TypeEngine
from database import Base


def nullable_column[T](type_: type[TypeEngine[T]]) -> Mapped[T | None]:
    """Fills a real gap in SQLAlchemy's own stubs: mapped_column(SomeType, nullable=True) has no
    overload producing Mapped[T | None] from a bare (non-Optional-parameterized) TypeEngine
    subclass like String or Text — verified directly against SQLAlchemy's stubs, not a usage
    mistake (mapped_column(String, nullable=True) alone always resolves to type[TypeEngine[str]],
    which never structurally satisfies the Optional target their stubs expect).

    pyright accepts this fix cleanly. mypy has a separate, unrelated solver limitation for this
    generic shape (a TypeVar nested inside a type[Generic[T]] parameter, combined with T | None
    in the return) that reproduces no matter how the overload is written — confirmed with several
    variants, including ones with no external annotation at all to rule out bidirectional
    inference from the caller. mypy still reports a false positive at each call site below; left
    unsuppressed since it's a known, verified false positive rather than a real issue.
    """
    return column(type_, nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"
    id = column(String, primary_key=True, index=True)
    title = column(String, index=True, nullable=False)
    # JSON: {active_prompt_id, active_tool_names, working_directory}
    settings = column(Text, nullable=True)
    created_at = column(String, nullable=False)
    # Logical FK to messages.id — not declared as FK to avoid circular constraint on SQLite
    active_message_id: Mapped[str | None] = nullable_column(String)


class Message(Base):
    __tablename__ = "messages"
    id = column(String, primary_key=True, index=True)
    conversation_id = column(String, ForeignKey("conversations.id"), index=True)
    # Null for the first message in a conversation; set to the parent message id otherwise
    parent_id = column(String, ForeignKey("messages.id"), nullable=True, index=True)
    role = column(String, index=True)  # user | assistant | system | tool
    content = column(Text)
    thinking = column(Text, nullable=True)
    created_at = column(String, nullable=False)
    token_count = column(Integer, nullable=True)
    token_delta = column(Integer, nullable=True)
    context_excluded = column(Boolean, nullable=False, default=False)
    exclusion_reason = column(String, nullable=True)
    compressed_summary = column(Text, nullable=True)
    compression_label = column(String, nullable=True)  # drop | 1-line-summary | summarize | keep
    compressed_token_count = column(Integer, nullable=True)
    log_message = column(String, nullable=True)
    # JSON array of {id, name, args} — populated on assistant messages that made tool calls
    tool_calls = column(Text, nullable=True)
    # True when the agent stopped without producing a response (thinking-only, no content/tool calls)
    is_degenerate = column(Boolean, nullable=False, default=False)
    # JSON blob for working memory messages (role=context_summary): structured sections persisted for folding
    working_memory_json = column(Text, nullable=True)
    # Set on the user message that started a workflow run, once the engine's run_id is known —
    # lets the run view be reopened from disk after a page reload (ADR-0011's "Deferred:
    # resumability"). Both null for a message that isn't a workflow invocation.
    workflow_name = column(String, nullable=True)
    workflow_run_id = column(String, nullable=True)


class Image(Base):
    __tablename__ = "images"
    id = column(String, primary_key=True)
    mime_type = column(String, nullable=False)
    data = column(Text, nullable=False)  # base64-encoded blob
    width = column(Integer, nullable=True)
    height = column(Integer, nullable=True)
    created_at = column(String, nullable=False)


class MessageImageAttachment(Base):
    __tablename__ = "message_image_attachments"
    message_id = column(String, ForeignKey("messages.id"), primary_key=True)
    image_id = column(String, ForeignKey("images.id"), primary_key=True)
    position = column(Integer, nullable=False)


class AppSettings(Base):
    __tablename__ = "app_settings"
    key = column(String, primary_key=True)
    value: Mapped[str | None] = nullable_column(Text)


class SystemPromptTemplate(Base):
    __tablename__ = "system_prompt_templates"
    id = column(String, primary_key=True, index=True)
    name = column(String, nullable=False)
    # general | code | summarization | context_compaction | state_storage
    category = column(String, nullable=False)
    content = column(Text, nullable=False)
    is_default = column(Boolean, nullable=False, default=False)
    token_count = column(Integer, nullable=True)
    created_at = column(String, nullable=False)


class RagSpace(Base):
    __tablename__ = "rag_spaces"
    id = column(String, primary_key=True, index=True)
    name = column(String, nullable=False)
    description: Mapped[str | None] = nullable_column(Text)
    # None = global space; otherwise the workspace this space is pinned to
    workspace_path: Mapped[str | None] = nullable_column(String)
    embedding_model = column(String, nullable=False)
    embedding_dim = column(Integer, nullable=False)
    created_at = column(String, nullable=False)


class RagSource(Base):
    __tablename__ = "rag_sources"
    id = column(String, primary_key=True, index=True)
    space_id = column(String, ForeignKey("rag_spaces.id"), index=True, nullable=False)
    source_type = column(String, nullable=False)  # paste | upload | workspace_path
    title = column(String, nullable=False)
    # Filename or workspace-relative path; None for pasted text
    origin_path: Mapped[str | None] = nullable_column(String)
    content_hash = column(String, nullable=False)
    status = column(String, nullable=False)  # pending | indexed | error
    error_message: Mapped[str | None] = nullable_column(Text)
    created_at = column(String, nullable=False)
    updated_at = column(String, nullable=False)


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    id = column(String, primary_key=True, index=True)
    source_id = column(String, ForeignKey("rag_sources.id"), index=True, nullable=False)
    space_id = column(String, ForeignKey("rag_spaces.id"), index=True, nullable=False)  # denormalized for query
    chunk_index = column(Integer, nullable=False)
    start_line: Mapped[int | None] = nullable_column(Integer)
    end_line: Mapped[int | None] = nullable_column(Integer)
    text = column(Text, nullable=False)
    embedding = column(Text, nullable=False)  # base64-encoded float32 bytes
    created_at = column(String, nullable=False)
