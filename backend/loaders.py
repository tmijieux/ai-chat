from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    thinking: str | None = None


class NewMessage(BaseModel):
    role: str
    content: str
    thinking: str | None = None
    parent_id: str | None = None
    token_count: int | None = None


class Conversation(BaseModel):
    messages: list[Message]


class NewConversation(BaseModel):
    title: str


class EditMessageContent(BaseModel):
    content: str


class BranchNavigation(BaseModel):
    message_id: str


class ConversationSettings(BaseModel):
    active_prompt_id: str | None = None
    active_tool_names: list[str] = []
    agentic_mode: bool = True
    working_directory: str | None = None


class NewSystemPrompt(BaseModel):
    name: str
    category: str
    content: str
    is_default: bool = False


class UpdateSystemPrompt(BaseModel):
    name: str | None = None
    category: str | None = None
    content: str | None = None
    is_default: bool | None = None
