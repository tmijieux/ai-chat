from typing import NotRequired
from pydantic import BaseModel

# Chat response model
class ChatResponse(BaseModel):
    status: str
    initial_context_tokens: int
    stream: list[dict]
    total_tokens: int

class Message(BaseModel):
    role: str
    content: str
    thinking: str|None=None

class Conversation(BaseModel):
    messages: list[Message]


class NewConversation(BaseModel):
    title: str
