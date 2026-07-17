"""
CodeMentor AI - Conversation & Message ORM Models
==================================================
Stores chat sessions (conversations) and individual messages
so the history persists across page refreshes.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class Conversation(BaseModel):
    """
    A single chat session identified by a UUID.
    The first user message becomes the title (truncated to 80 chars).
    """
    __tablename__ = "conversations"

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Conversation",
        comment="Auto-generated title from first user message",
    )

    # One-to-many: a conversation has many messages
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="select",
    )


class Message(BaseModel):
    """
    A single message within a conversation.
    role: 'user' or 'assistant'
    """
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent conversation",
    )

    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="'user' or 'assistant'",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Message text (markdown for assistant messages)",
    )

    conversation: Mapped[Conversation] = relationship(
        "Conversation",
        back_populates="messages",
    )
