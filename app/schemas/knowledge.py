"""
CodeMentor AI - Knowledge Base Pydantic Schemas
=================================================
Request/response schemas for the Knowledge Base API endpoints.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.rag.collections import VALID_COLLECTION_NAMES


# ==============================================================
# Request Schemas
# ==============================================================

class KnowledgeUploadRequest(BaseModel):
    """
    Metadata accompanying a file upload.
    The actual file is sent as multipart/form-data.
    """

    collection_name: str = Field(
        description="Target ChromaDB collection to store this document.",
        examples=["python_docs", "dsa_notes", "fastapi_docs"],
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional description of this document.",
    )

    @field_validator("collection_name")
    @classmethod
    def validate_collection(cls, v: str) -> str:
        if v not in VALID_COLLECTION_NAMES:
            raise ValueError(
                f"Invalid collection '{v}'. "
                f"Valid collections: {VALID_COLLECTION_NAMES}"
            )
        return v


class KnowledgeSearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(
        min_length=3,
        max_length=2000,
        description="Natural language search query.",
        examples=["How does binary search work?", "FastAPI dependency injection"],
    )
    collection_name: str | None = Field(
        default=None,
        description="Specific collection to search. If None, auto-routes based on query.",
        examples=["python_docs", "dsa_notes"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1-20).",
    )

    @field_validator("collection_name")
    @classmethod
    def validate_collection(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_COLLECTION_NAMES:
            raise ValueError(
                f"Invalid collection '{v}'. "
                f"Valid: {VALID_COLLECTION_NAMES}"
            )
        return v


# ==============================================================
# Response Schemas
# ==============================================================

class IngestionResultResponse(BaseModel):
    """Response returned after a file upload and ingestion."""

    source_id: str = Field(description="Unique ID for this document (use to delete)")
    filename: str
    collection_name: str
    status: str
    chunks_created: int
    chunks_stored: int
    page_count: int
    processing_time: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class SearchResultItem(BaseModel):
    """A single search result chunk."""

    content: str = Field(description="The matched text chunk")
    score: float = Field(description="Similarity score (0-1, higher = more relevant)")
    collection: str = Field(description="Collection this chunk came from")
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchResponse(BaseModel):
    """Response for a semantic search request."""

    query: str
    collection_searched: str | None
    total_results: int
    results: list[SearchResultItem]
    context_string: str = Field(
        description="Pre-formatted context string ready for LLM injection"
    )


class KnowledgeDocumentResponse(BaseModel):
    """Response schema for a single document record."""

    id: UUID
    filename: str
    collection_name: str
    source_id: str
    status: str
    page_count: int
    chunks_created: int
    chunks_stored: int
    file_size_bytes: int
    file_type: str
    document_title: str | None
    document_author: str | None
    description: str | None
    error_message: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CollectionStatsResponse(BaseModel):
    """Knowledge base statistics per collection."""

    collections: dict[str, int] = Field(
        description="Map of collection name → chunk count"
    )
    total_chunks: int
    total_collections: int


class CollectionListResponse(BaseModel):
    """List of available ChromaDB collections."""

    collections: list[dict[str, str]] = Field(
        description="Available collections with name and description"
    )
    total: int
