from sqlalchemy import Integer, String, ForeignKey, Text, Boolean
from sqlalchemy.orm import mapped_column as column
from database import Base


class Conversation(Base):
    __tablename__ = "conversations"
    id = column(String, primary_key=True, index=True)
    title = column(String, index=True, nullable=False)
    # JSON: {active_prompt_id, active_tool_names, working_directory}
    settings = column(Text, nullable=True)
    created_at = column(String, nullable=False)
    # Logical FK to messages.id — not declared as FK to avoid circular constraint on SQLite
    active_message_id = column(String, nullable=True)


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
    value = column(Text, nullable=True)


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
