"""
CodeMentor AI - Chat Router
===========================
Defines the /chat endpoints for the AI assistant.

Endpoints:
- POST /api/v1/chat: Send a query to the agentic RAG system.
"""

from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.core.database import get_db
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import (
    ChatRequest, 
    ChatResponse, 
    ConversationHistoryItem, 
    ConversationDetailResponse,
    MessageHistoryItem
)
from app.services.chat_service import ChatService

router = APIRouter(tags=["Chat"])

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the CodeMentor AI Assistant",
)
async def chat_with_agent(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
) -> ChatResponse:
    """Send a query to the Agentic RAG assistant."""
    chat_service = ChatService(db)
    return await chat_service.chat(request)


@router.get(
    "/conversations",
    response_model=list[ConversationHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="List recent conversations",
)
async def list_recent_conversations(
    db: AsyncSession = Depends(get_db)
) -> list[ConversationHistoryItem]:
    """Retrieve the recent conversations list for the sidebar."""
    repo = ChatRepository(db)
    conversations = await repo.get_conversations()
    return [
        ConversationHistoryItem(
            id=str(c.id),
            title=c.title,
            created_at=c.created_at.isoformat()
        )
        for c in conversations
    ]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve full conversation details and messages",
)
async def get_conversation_details(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
) -> ConversationDetailResponse:
    """Retrieve full conversation details and messages by ID."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format."
        )

    repo = ChatRepository(db)
    conversation = await repo.get_or_create_conversation(conv_uuid)
    messages = await repo.get_messages(conv_uuid)

    return ConversationDetailResponse(
        id=str(conversation.id),
        title=conversation.title,
        messages=[
            MessageHistoryItem(role=msg.role, content=msg.content)
            for msg in messages
        ]
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation and all its messages",
)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Permanently delete a conversation and all its messages by ID."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format."
        )

    repo = ChatRepository(db)
    deleted = await repo.delete_conversation(conv_uuid)
    await db.commit()

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found."
        )

    return {"success": True, "message": "Conversation deleted."}
