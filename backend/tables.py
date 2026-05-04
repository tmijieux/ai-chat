from sqlalchemy import Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship, mapped_column as column

# Base class for models
from database import Base 

class Conversation(Base):
    __tablename__ = "conversations"
    id = column(String, primary_key=True, index=True)
    title = column(String, index=True, nullable=False)
    # System prompt/settings can be stored here
    settings = column(Text, nullable=True) 
    
    # Relationship to all messages within this conversation
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    id = column(String, primary_key=True, index=True)
    conversation_id = column(String, ForeignKey("conversations.id"), index=True)
    role = column(String, index=True)  # 'user', 'assistant', 'system'
    content = column(Text)
    thinking = column(Text)
    created_at = column(String, index=True)
    token_count = column(Integer)
    
    # Relationship back to the parent conversation
    conversation = relationship(Conversation, back_populates="messages")
