from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    thinking: str | None = None


class NewMessage(BaseModel):
    id: str
    role: str
    content: str
    thinking: str | None = None
    parent_id: str | None = None
    token_count: int | None = None
    token_delta: int | None = None
    log_message: str | None = None
    image_ids: list[str] = []
    # [{id, name, args}] — assistant messages that invoked tools
    tool_calls: list | None = None
    is_degenerate: bool = False


class UpdateTokenCount(BaseModel):
    token_count: int | None = None
    token_delta: int | None = None


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
    working_directory: str | None = None
    mode: str = "standard"


class AppSettingUpdate(BaseModel):
    value: str | None = None


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


class NewAgent(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    tools: list[str] = []
    finish_tool: str = "finish_task"
    max_iterations: int | None = None
    inject_turn_reminders: bool = False


class UpdateAgent(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    finish_tool: str | None = None
    max_iterations: int | None = None
    inject_turn_reminders: bool | None = None


class CorrectRequest(BaseModel):
    text: str
    language: str | None = "fr"
