"""
CodeMentor AI - Chat Repository
===============================
Provides database access methods for Conversations and Messages.
"""

import uuid
from typing import Sequence
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.conversation import Conversation, Message


class ChatRepository:
    """
    Handles CRUD operations for Conversation and Message models.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create_conversation(self, conversation_id: uuid.UUID, title: str = "New Conversation") -> Conversation:
        """Get an existing conversation or create a new one."""
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            conversation = Conversation(id=conversation_id, title=title)
            self.session.add(conversation)
            await self.session.flush()
        return conversation

    async def update_conversation_title(self, conversation_id: uuid.UUID, title: str) -> None:
        """Update conversation title."""
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(query)
        conversation = result.scalar_one_or_none()
        if conversation:
            conversation.title = title
            await self.session.flush()

    async def get_conversations(self, limit: int = 50) -> Sequence[Conversation]:
        """Fetch latest conversations sorted by creation date."""
        query = select(Conversation).order_by(desc(Conversation.created_at)).limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_messages(self, conversation_id: uuid.UUID) -> Sequence[Message]:
        """Fetch all messages for a given conversation sorted by creation date."""
        query = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def add_message(self, conversation_id: uuid.UUID, role: str, content: str) -> Message:
        """Add a single message to a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def delete_conversation(self, conversation_id: uuid.UUID) -> bool:
        """
        Delete a conversation and all its messages.
        Returns True if deleted, False if not found.
        """
        from sqlalchemy import delete as sql_delete

        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            return False

        # Delete all messages first (respects FK constraint)
        await self.session.execute(
            sql_delete(Message).where(Message.conversation_id == conversation_id)
        )
        # Delete the conversation row
        await self.session.delete(conversation)
        await self.session.flush()
        return True
