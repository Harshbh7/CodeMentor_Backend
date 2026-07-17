"""
CodeMentor AI - Knowledge Base ORM Model
==========================================
SQLAlchemy model for tracking uploaded knowledge base documents.

Design Rationale:
- We store document metadata in PostgreSQL, but the actual text chunks
  and vectors live in ChromaDB. PostgreSQL is the "source of truth" for
  document records (who uploaded what, when, to which collection).
- The `source_id` is the bridge between PostgreSQL and ChromaDB —
  it allows us to delete all ChromaDB chunks for a given record.
- `status` tracks ingestion progress for async upload workflows.
"""

from enum import Enum as PyEnum

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class DocumentStatus(str, PyEnum):
    """Lifecycle status of a knowledge base document."""

    PENDING = "pending"         # Uploaded but not yet processed
    PROCESSING = "processing"   # Currently being chunked/embedded
    COMPLETED = "completed"     # Successfully stored in ChromaDB
    FAILED = "failed"           # Ingestion failed (see error_message)


class KnowledgeDocument(BaseModel):
    """
    Tracks every file uploaded to the RAG knowledge base.

    Table: knowledge_documents

    Relationships:
    - In future phases: ForeignKey to User model (who uploaded it)
    """

    __tablename__ = "knowledge_documents"

    # Original file name as uploaded by the user
    filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Original uploaded filename",
    )

    # ChromaDB collection where this document's chunks are stored
    collection_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Target ChromaDB collection name",
    )

    # The UUID linking this record to its ChromaDB chunks
    source_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
        comment="UUID used as ChromaDB source identifier for deletion",
    )

    # Document processing status
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True,
        comment="Ingestion lifecycle status",
    )

    # Document content statistics
    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of pages (or 1 for text files)",
    )

    chunks_created: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of text chunks generated from this document",
    )

    chunks_stored: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of chunks successfully stored in ChromaDB",
    )

    # File metadata
    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Original file size in bytes",
    )

    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pdf",
        comment="File extension (pdf, txt, md)",
    )

    # Processing metadata
    processing_time_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=True,
        comment="Time taken to process this document (seconds)",
    )

    # Document-level metadata extracted during processing
    document_title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="PDF title extracted from metadata",
    )

    document_author: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="PDF author extracted from metadata",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="User-provided description of this document",
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error details if status=FAILED",
    )

    # Soft delete flag (don't physically delete — just mark inactive)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="False when document is deleted (soft delete)",
    )

    def __repr__(self) -> str:
        return (
            f"<KnowledgeDocument id={self.id} "
            f"file='{self.filename}' "
            f"collection='{self.collection_name}' "
            f"status='{self.status}'>"
        )
