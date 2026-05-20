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
    active_prompt_ids: list[str] = []
    active_tool_names: list[str] = []
    tools_enabled: bool = True
    agentic_mode: bool = True


class NewSystemPrompt(BaseModel):
    name: str
    category: str
    content: str
    is_global: bool = True


class UpdateSystemPrompt(BaseModel):
    name: str | None = None
    category: str | None = None
    content: str | None = None
    is_global: bool | None = None
