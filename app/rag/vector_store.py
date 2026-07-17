"""
CodeMentor AI - ChromaDB Vector Store Manager
================================================
Manages all interactions with ChromaDB: collection creation, document
storage, similarity search, and deletion.

Design Rationale:
- We use ChromaDB's HTTP client (not in-process) so the vector DB runs
  as a separate Docker service — more scalable, survives API restarts.
- Each knowledge domain gets its own collection — enables targeted retrieval.
- EmbeddingFunction is set to None (we supply our own vectors) — gives us
  full control over the embedding model and task type.
- Metadata filtering at search time allows the agent to restrict search
  to specific languages or topics.
"""

import uuid
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.exceptions import VectorDBError
from app.core.logging import get_logger
from app.rag.chunker import TextChunk
from app.rag.collections import (
    ALL_COLLECTIONS,
    COLLECTION_MAP,
    CollectionConfig,
)

logger = get_logger(__name__)
settings = get_settings()


# Module-level global client cache to avoid re-initializing connection on every request
_global_client: Any = None


class VectorStore:
    """
    Manages ChromaDB collections for the CodeMentor AI knowledge base.

    Key Operations:
    - initialize_collections(): Create all domain collections on startup.
    - add_documents():          Store chunks + embeddings in a collection.
    - search():                 Semantic similarity search with metadata filtering.
    - delete_document():        Remove all chunks from a specific source document.
    - get_collection_stats():   Count documents per collection.
    """

    def __init__(self) -> None:
        self._collections: dict[str, Collection] = {}

    def _get_client(self) -> Any:
        """
        Lazily initialize and return the ChromaDB client.
        Supports both local HTTP client and ChromaDB Cloud client.
        """
        global _global_client
        if _global_client is None:
            try:
                if settings.chroma_api_key:
                    logger.info("Initializing ChromaDB Cloud client...")
                    _global_client = chromadb.CloudClient(
                        api_key=settings.chroma_api_key,
                        tenant=settings.chroma_tenant or chromadb.DEFAULT_TENANT,
                        database=settings.chroma_database or chromadb.DEFAULT_DATABASE,
                    )
                    logger.info(
                        "ChromaDB Cloud client initialized (database=%s, tenant=%s)",
                        settings.chroma_database,
                        settings.chroma_tenant,
                    )
                else:
                    _global_client = chromadb.HttpClient(
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                        ssl=settings.chroma_ssl,
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                            allow_reset=True,  # Enable in dev for clean resets
                        ),
                    )
                    logger.info(
                        "Local ChromaDB client connected: %s:%s",
                        settings.chroma_host,
                        settings.chroma_port,
                    )
            except Exception as exc:
                logger.error("Failed to connect to ChromaDB: %s", exc)
                if settings.chroma_api_key:
                    raise VectorDBError(f"Cannot connect to ChromaDB Cloud: {exc}") from exc
                else:
                    raise VectorDBError(
                        f"Cannot connect to ChromaDB at {settings.chroma_host}:{settings.chroma_port}. "
                        "Ensure ChromaDB is running (docker compose up chromadb)."
                    ) from exc
        return _global_client

    def initialize_collections(self) -> None:
        """
        Ensure all defined collections exist in ChromaDB.
        Called at application startup.

        Uses `get_or_create_collection` — idempotent, safe to call multiple times.
        """
        client = self._get_client()
        created = 0
        existing = 0

        for config in ALL_COLLECTIONS:
            try:
                collection = client.get_or_create_collection(
                    name=config.name,
                    metadata=config.metadata,
                    # embedding_function=None means we supply precomputed vectors
                )
                self._collections[config.name] = collection

                count = collection.count()
                if count > 0:
                    existing += 1
                    logger.debug(
                        "Collection '%s' exists with %d documents.", config.name, count
                    )
                else:
                    created += 1

            except Exception as exc:
                logger.error(
                    "Failed to create/get collection '%s': %s", config.name, exc
                )
                raise VectorDBError(
                    f"Failed to initialize collection '{config.name}': {exc}"
                ) from exc

        logger.info(
            "ChromaDB initialized: %d collections (%d new, %d existing).",
            len(ALL_COLLECTIONS), created, existing,
        )

    def _get_collection(self, collection_name: str) -> Collection:
        """
        Retrieve a collection by name, with lazy initialization.

        Args:
            collection_name: Must be one of the defined collection names.

        Returns:
            ChromaDB Collection object.

        Raises:
            VectorDBError: If collection not found or ChromaDB unavailable.
        """
        if collection_name not in COLLECTION_MAP:
            raise VectorDBError(
                f"Unknown collection: '{collection_name}'. "
                f"Valid collections: {list(COLLECTION_MAP.keys())}"
            )

        if collection_name not in self._collections:
            client = self._get_client()
            try:
                self._collections[collection_name] = client.get_or_create_collection(
                    name=collection_name,
                    metadata=COLLECTION_MAP[collection_name].metadata,
                )
            except Exception as exc:
                raise VectorDBError(
                    f"Cannot access collection '{collection_name}': {exc}"
                ) from exc

        return self._collections[collection_name]

    def add_documents(
        self,
        collection_name: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        source_id: str,
    ) -> int:
        """
        Store text chunks with their embeddings in a ChromaDB collection.

        Args:
            collection_name: Target collection (e.g., "python_docs").
            chunks:          Text chunks from the chunker.
            embeddings:      Corresponding embedding vectors from the embedder.
            source_id:       Unique identifier for the source document (for deletion).

        Returns:
            Number of chunks successfully stored.

        Raises:
            VectorDBError: If storage fails.
        """
        if len(chunks) != len(embeddings):
            raise VectorDBError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )

        collection = self._get_collection(collection_name)

        # Prepare ChromaDB-compatible data
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embedding_list: list[list[float]] = []

        for chunk, embedding in zip(chunks, embeddings):
            # Unique ID: source_id + chunk hash (handles re-uploads gracefully)
            doc_id = f"{source_id}_{chunk.chunk_id}"

            # ChromaDB requires metadata values to be str/int/float/bool
            safe_metadata: dict[str, Any] = {
                "source_id": source_id,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.metadata.get("char_count", len(chunk.content)),
                "is_code": str(chunk.metadata.get("is_code", False)),
                "collection": collection_name,
            }

            # Add optional metadata fields if present
            if "page_number" in chunk.metadata:
                safe_metadata["page_number"] = chunk.metadata["page_number"]
            if "source_file" in chunk.metadata:
                safe_metadata["source_file"] = str(chunk.metadata["source_file"])
            if "topic" in chunk.metadata:
                safe_metadata["topic"] = str(chunk.metadata["topic"])
            if "detected_language" in chunk.metadata:
                safe_metadata["detected_language"] = str(chunk.metadata["detected_language"])

            ids.append(doc_id)
            documents.append(chunk.content)
            metadatas.append(safe_metadata)
            embedding_list.append(embedding)

        try:
            # upsert: insert or update existing (idempotent for re-uploads)
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_list,
            )
            logger.info(
                "Stored %d chunks in collection '%s' (source_id=%s).",
                len(chunks), collection_name, source_id
            )
            return len(chunks)

        except Exception as exc:
            logger.error(
                "Failed to store documents in '%s': %s", collection_name, exc
            )
            raise VectorDBError(
                f"Failed to store documents in '{collection_name}': {exc}"
            ) from exc

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic similarity search in a collection.

        Args:
            collection_name: Collection to search.
            query_embedding: Query vector from embedder.embed_query().
            n_results:       Number of top results to return.
            where:           Optional metadata filter (ChromaDB `where` clause).
                             Example: {"detected_language": "python"}

        Returns:
            List of result dicts, each containing:
            - content:   The matched text chunk.
            - score:     Similarity score (lower distance = more similar).
            - metadata:  Chunk metadata.
            - id:        ChromaDB document ID.
        """
        collection = self._get_collection(collection_name)

        # Check collection has documents
        count = collection.count()
        if count == 0:
            logger.warning("Search on empty collection '%s'.", collection_name)
            return []

        # Ensure n_results doesn't exceed collection size
        n_results = min(n_results, count)

        try:
            query_params: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_params["where"] = where

            results = collection.query(**query_params)

        except Exception as exc:
            logger.error(
                "Search failed in collection '%s': %s", collection_name, exc
            )
            raise VectorDBError(
                f"Search failed in '{collection_name}': {exc}"
            ) from exc

        # Parse ChromaDB response into clean dicts
        formatted: list[dict[str, Any]] = []
        if not results or not results.get("documents"):
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
            # Convert cosine distance to similarity score (0-1, higher = more similar)
            similarity = 1 - dist

            formatted.append({
                "content": doc,
                "score": round(similarity, 4),
                "distance": round(dist, 4),
                "metadata": meta,
                "id": doc_id,
                "collection": collection_name,
            })

        logger.debug(
            "Search in '%s': query returned %d results.", collection_name, len(formatted)
        )
        return formatted

    def search_multiple_collections(
        self,
        collection_names: list[str],
        query_embedding: list[float],
        n_results_per_collection: int = 3,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search across multiple collections and merge results by score.

        Used by the agent when the topic spans multiple domains
        (e.g., "Python FastAPI with SQLAlchemy" → search python_docs + fastapi_docs + sqlalchemy_docs).

        Args:
            collection_names:        Collections to search.
            query_embedding:         Query embedding vector.
            n_results_per_collection: Results per collection before merging.
            where:                   Optional metadata filter.

        Returns:
            Merged and sorted results (highest similarity first).
        """
        all_results: list[dict[str, Any]] = []

        for col_name in collection_names:
            try:
                results = self.search(
                    collection_name=col_name,
                    query_embedding=query_embedding,
                    n_results=n_results_per_collection,
                    where=where,
                )
                all_results.extend(results)
            except VectorDBError as exc:
                logger.warning("Skipping collection '%s': %s", col_name, exc)

        # Sort by similarity score (highest first)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

    def delete_document(
        self,
        collection_name: str,
        source_id: str,
    ) -> int:
        """
        Delete all chunks belonging to a source document.

        Args:
            collection_name: Collection containing the document.
            source_id:       Source document identifier.

        Returns:
            Number of chunks deleted.
        """
        collection = self._get_collection(collection_name)

        try:
            # Find all chunk IDs belonging to this source
            results = collection.get(
                where={"source_id": source_id},
                include=["documents"],
            )
            ids_to_delete = results.get("ids", [])

            if not ids_to_delete:
                logger.warning(
                    "No chunks found for source_id='%s' in '%s'.",
                    source_id, collection_name
                )
                return 0

            collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted %d chunks for source_id='%s' from '%s'.",
                len(ids_to_delete), source_id, collection_name
            )
            return len(ids_to_delete)

        except Exception as exc:
            logger.error(
                "Delete failed for source_id='%s' in '%s': %s",
                source_id, collection_name, exc
            )
            raise VectorDBError(f"Delete operation failed: {exc}") from exc

    def get_collection_stats(self) -> dict[str, int]:
        """
        Return document count for each collection.
        Used in admin/dashboard endpoints.

        Returns:
            Dict mapping collection name → document count.
        """
        stats: dict[str, int] = {}
        for col_name in COLLECTION_MAP:
            try:
                collection = self._get_collection(col_name)
                stats[col_name] = collection.count()
            except VectorDBError:
                stats[col_name] = -1  # -1 indicates unavailable
        return stats

    def check_health(self) -> bool:
        """
        Check if ChromaDB is reachable.
        Returns True if healthy, False otherwise.
        """
        try:
            client = self._get_client()
            client.heartbeat()
            return True
        except Exception as exc:
            logger.warning("ChromaDB health check failed: %s", exc)
            return False
