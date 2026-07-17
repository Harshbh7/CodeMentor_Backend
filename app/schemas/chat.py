"""
CodeMentor AI - Chat Pydantic Schemas
=======================================
Request and response models for the /chat endpoint.

Design Rationale:
- Pydantic v2 with Field validators ensures clean API contract.
- ChatResponse includes tool_calls_used so clients can understand
  HOW the answer was derived (transparency).
- SourceReference lets the frontend display citations from RAG.
- query length is capped at 4000 chars to prevent abuse.
"""

from typing import Any

from pydantic import Field, field_validator

from pydantic import BaseModel as BaseSchema


class ChatRequest(BaseSchema):
    """
    Request body for POST /api/v1/chat.

    Fields:
        query:           The user's question or code snippet.
        conversation_id: Optional. For future multi-turn support.
        collection:      Optional. Force retrieval from a specific KB collection.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question or programming request.",
        examples=["What is the time complexity of quicksort?"],
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for multi-turn sessions (future use).",
    )
    collection: str | None = Field(
        default=None,
        description=(
            "Optional: force search in a specific knowledge base collection. "
            "Leave empty for automatic routing."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be blank or whitespace only.")
        return v.strip()


class ToolCallInfo(BaseSchema):
    """
    Information about a single tool invocation during agent execution.
    Included in ChatResponse so clients can see how the answer was built.
    """
    tool_name: str = Field(description="Name of the tool that was called.")
    tool_input: dict[str, Any] = Field(description="Arguments passed to the tool.")
    tool_output_preview: str = Field(
        description="First 500 chars of the tool's output (truncated for brevity)."
    )


class SourceReference(BaseSchema):
    """
    A reference to a retrieved knowledge base chunk.
    Allows the frontend to display citations.
    """
    content_preview: str = Field(description="First 200 chars of the retrieved chunk.")
    collection: str = Field(description="ChromaDB collection the chunk came from.")
    relevance_score: float = Field(description="Similarity score (0.0 - 1.0).")
    page_number: int | None = Field(default=None, description="Source page number if applicable.")


class ChatResponse(BaseSchema):
    """
    Response from POST /api/v1/chat.

    Fields:
        answer:            The agent's final answer (Markdown formatted).
        tool_calls_used:   List of tools the agent invoked to produce the answer.
        sources:           RAG source references (if knowledge base was used).
        conversation_id:   Echo back the conversation_id for multi-turn use.
        iterations:        Number of think→tool loops the agent took.
        processing_time:   Total time taken in seconds.
        success:           Whether the agent completed successfully.
        error:             Error message if success is False.
    """
    answer: str = Field(description="The agent's final markdown-formatted answer.")
    tool_calls_used: list[ToolCallInfo] = Field(
        default_factory=list,
        description="Tools invoked during agent execution.",
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Knowledge base sources cited in the answer.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID for session continuity.",
    )
    iterations: int = Field(
        default=0,
        description="Number of think→tool loops taken.",
    )
    processing_time: float = Field(
        default=0.0,
        description="Total processing time in seconds.",
    )
    success: bool = Field(default=True)
    error: str | None = Field(default=None)
    cached: bool = Field(
        default=False,
        description="True if this response was served from Valkey cache (not re-computed).",
    )


class ConversationHistoryItem(BaseSchema):
    """Schema for listing past conversations in the history sidebar."""
    id: str = Field(description="UUID of the conversation as a string")
    title: str = Field(description="Auto-generated title")
    created_at: str = Field(description="ISO-formatted creation timestamp")


class MessageHistoryItem(BaseSchema):
    """Schema for individual message payload in conversation history."""
    role: str = Field(description="Sender role: user or assistant")
    content: str = Field(description="Message body text")


class ConversationDetailResponse(BaseSchema):
    """Schema for the full detail of a past conversation including all messages."""
    id: str = Field(description="UUID of the conversation")
    title: str = Field(description="Conversation title")
    messages: list[MessageHistoryItem] = Field(default_factory=list, description="Ordered conversation messages")
