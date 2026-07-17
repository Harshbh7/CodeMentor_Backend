"""
CodeMentor AI - RAG Ingestion Pipeline
=========================================
Orchestrates the complete RAG ingestion flow:
PDF → Extract → Chunk → Embed → Store in ChromaDB

This is the main entry point for adding knowledge to the system.

Design Rationale:
- Pipeline pattern: each stage (extract, chunk, embed, store) is independent
  and can be tested/replaced separately.
- Async-first: all I/O-heavy stages are awaited to not block the FastAPI event loop.
- Progress tracking: yields status updates for streaming upload progress (Phase 7).
- Transaction-like behavior: if embedding or storage fails, we don't store partial data.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.exceptions import AIServiceError, ValidationError, VectorDBError
from app.core.logging import get_logger
from app.rag.chunker import TextChunk, TextChunker
from app.rag.collections import VALID_COLLECTION_NAMES, TOPIC_TO_COLLECTION
from app.rag.embedder import GeminiEmbedder
from app.rag.pdf_processor import PDFProcessor
from app.rag.retriever import Retriever, RetrievalResult
from app.rag.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """
    Result of a RAG ingestion operation.

    Attributes:
        source_id:        Unique ID assigned to this document.
        filename:         Original file name.
        collection_name:  Target ChromaDB collection.
        chunks_created:   Number of text chunks generated.
        chunks_stored:    Number of chunks successfully embedded and stored.
        page_count:       Number of PDF pages processed.
        processing_time:  Total time taken in seconds.
        success:          Whether the ingestion completed successfully.
        error:            Error message if ingestion failed.
    """

    source_id: str
    filename: str
    collection_name: str
    chunks_created: int = 0
    chunks_stored: int = 0
    page_count: int = 0
    processing_time: float = 0.0
    success: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    Orchestrates the end-to-end RAG knowledge base ingestion pipeline.

    Flow:
        File bytes
            → PDFProcessor.extract_from_bytes()     [Extract text]
            → TextChunker.chunk_pages()             [Split into chunks]
            → GeminiEmbedder.embed_chunks_async()   [Generate vectors]
            → VectorStore.add_documents()           [Store in ChromaDB]
            → IngestionResult                       [Return summary]

    Retrieval Flow (used by agent):
        User query
            → Retriever.retrieve_by_topic()         [Semantic search]
            → RetrievalResult                       [Ranked results]
            → LLM context string                    [For Gemini prompt]
    """

    def __init__(self) -> None:
        self._pdf_processor = PDFProcessor()
        self._chunker = TextChunker()
        self._embedder = GeminiEmbedder()
        self._vector_store = VectorStore()
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
        )
        logger.info("RAGPipeline initialized.")

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        collection_name: str,
        additional_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """
        Ingest a file into the knowledge base.

        Full pipeline: extract → chunk → embed → store.

        Args:
            file_bytes:          Raw bytes of the uploaded file.
            filename:            Original file name (e.g., "python_docs.pdf").
            collection_name:     Target ChromaDB collection.
            additional_metadata: Extra metadata to attach to all chunks.

        Returns:
            IngestionResult with statistics and status.
        """
        start_time = datetime.now(timezone.utc)
        source_id = str(uuid.uuid4())

        result = IngestionResult(
            source_id=source_id,
            filename=filename,
            collection_name=collection_name,
        )

        # Validate collection
        if collection_name not in VALID_COLLECTION_NAMES:
            result.error = (
                f"Invalid collection '{collection_name}'. "
                f"Valid: {VALID_COLLECTION_NAMES}"
            )
            logger.error(result.error)
            return result

        logger.info(
            "Starting ingestion: file='%s' collection='%s' source_id='%s'",
            filename, collection_name, source_id
        )

        try:
            # ---- Stage 1: Extract Text ----
            logger.info("[1/4] Extracting text from '%s'...", filename)
            document = self._pdf_processor.extract_from_bytes(file_bytes, filename)
            result.page_count = document.page_count
            logger.info(
                "[1/4] ✓ Extracted %d pages (%d chars total).",
                document.page_count, len(document.full_text)
            )

            # ---- Stage 2: Chunk ----
            logger.info("[2/4] Chunking text...")
            base_metadata: dict[str, Any] = {
                "source_id": source_id,
                "source_file": filename,
                "collection": collection_name,
                "ingestion_time": start_time.isoformat(),
                **(additional_metadata or {}),
            }
            chunks: list[TextChunk] = self._chunker.chunk_pages(
                pages=document.pages,
                metadata=base_metadata,
            )
            result.chunks_created = len(chunks)
            logger.info("[2/4] ✓ Created %d chunks.", len(chunks))

            if not chunks:
                result.error = "No text chunks were created. File may be empty."
                return result

            # ---- Stage 3: Embed ----
            logger.info("[3/4] Generating embeddings for %d chunks...", len(chunks))
            embeddings = await self._embedder.embed_chunks_async(chunks)
            logger.info("[3/4] ✓ Generated %d embedding vectors.", len(embeddings))

            # ---- Stage 4: Store ----
            logger.info("[4/4] Storing in ChromaDB collection '%s'...", collection_name)
            stored_count = self._vector_store.add_documents(
                collection_name=collection_name,
                chunks=chunks,
                embeddings=embeddings,
                source_id=source_id,
            )
            result.chunks_stored = stored_count
            logger.info("[4/4] ✓ Stored %d chunks.", stored_count)

            # Success
            end_time = datetime.now(timezone.utc)
            result.processing_time = (end_time - start_time).total_seconds()
            result.success = True
            result.metadata = {
                "title": document.metadata.get("title", filename),
                "author": document.metadata.get("author", "Unknown"),
                "page_count": document.page_count,
            }

            logger.info(
                "Ingestion complete: file='%s' chunks=%d time=%.2fs",
                filename, stored_count, result.processing_time
            )

        except ValidationError as exc:
            result.error = exc.message
            logger.error("Ingestion validation error: %s", exc.message)
        except AIServiceError as exc:
            result.error = f"AI embedding failed: {exc.message}"
            logger.error("Ingestion AI error: %s", exc.message)
        except VectorDBError as exc:
            result.error = f"Vector store error: {exc.message}"
            logger.error("Ingestion vector DB error: %s", exc.message)
        except Exception as exc:
            result.error = f"Unexpected error: {exc}"
            logger.exception("Unexpected ingestion error: %s", exc)

        if not result.success:
            end_time = datetime.now(timezone.utc)
            result.processing_time = (end_time - start_time).total_seconds()

        return result

    async def ingest_text(
        self,
        text: str,
        collection_name: str,
        source_name: str = "manual_input",
        additional_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """
        Ingest raw text directly (without a file upload).

        Useful for seeding the knowledge base programmatically
        (e.g., from hardcoded documentation strings or scraped content).

        Args:
            text:             Raw text content to ingest.
            collection_name:  Target collection.
            source_name:      Descriptive name for the source.
            additional_metadata: Extra metadata for chunks.

        Returns:
            IngestionResult.
        """
        source_id = str(uuid.uuid4())
        result = IngestionResult(
            source_id=source_id,
            filename=source_name,
            collection_name=collection_name,
        )

        start_time = datetime.now(timezone.utc)

        try:
            base_metadata: dict[str, Any] = {
                "source_id": source_id,
                "source_file": source_name,
                "collection": collection_name,
                **(additional_metadata or {}),
            }
            chunks = self._chunker.chunk_text(text, metadata=base_metadata)
            result.chunks_created = len(chunks)

            if not chunks:
                result.error = "No chunks generated from provided text."
                return result

            embeddings = await self._embedder.embed_chunks_async(chunks)
            stored_count = self._vector_store.add_documents(
                collection_name=collection_name,
                chunks=chunks,
                embeddings=embeddings,
                source_id=source_id,
            )
            result.chunks_stored = stored_count
            result.success = True

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Text ingestion error: %s", exc)

        end_time = datetime.now(timezone.utc)
        result.processing_time = (end_time - start_time).total_seconds()
        return result

    async def search(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Search the knowledge base.

        If collection_name is None, automatically resolves the best collection(s).
        If collection_name is specified, searches that collection directly.

        Args:
            query:           Natural language search query.
            collection_name: Optional specific collection to search.
            top_k:           Number of results to return.

        Returns:
            RetrievalResult with ranked chunks and context string.
        """
        if collection_name:
            return await self._retriever.retrieve_from_collection(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
            )
        else:
            return await self._retriever.retrieve_by_topic(
                query=query,
                top_k=top_k,
            )

    async def delete_document(
        self,
        collection_name: str,
        source_id: str,
    ) -> dict[str, Any]:
        """
        Remove a document and all its chunks from the knowledge base.

        Args:
            collection_name: Collection containing the document.
            source_id:       Document's source ID (returned at ingestion).

        Returns:
            Dict with deletion result.
        """
        try:
            deleted = self._vector_store.delete_document(
                collection_name=collection_name,
                source_id=source_id,
            )
            return {"success": True, "chunks_deleted": deleted, "source_id": source_id}
        except VectorDBError as exc:
            return {"success": False, "error": exc.message, "source_id": source_id}

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """
        Return statistics for all collections.

        Returns:
            Dict with per-collection document counts and totals.
        """
        stats = self._vector_store.get_collection_stats()
        total = sum(v for v in stats.values() if v >= 0)
        return {
            "collections": stats,
            "total_chunks": total,
            "total_collections": len(stats),
        }

    def check_health(self) -> bool:
        """Check if the RAG pipeline dependencies are healthy."""
        return self._vector_store.check_health()


# ==============================================================
# Singleton Instance
# ==============================================================

_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Returns the singleton RAGPipeline instance.
    Used as a FastAPI dependency.

    Usage:
        from app.rag.pipeline import get_rag_pipeline
        pipeline = Depends(get_rag_pipeline)
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
