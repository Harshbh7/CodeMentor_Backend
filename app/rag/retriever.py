"""
CodeMentor AI - Semantic Retriever
=====================================
High-level retrieval interface that combines the embedder + vector store
into a clean API for the agent and other services.

Design Rationale:
- The Retriever is the only component the agent talks to for retrieval.
  It doesn't need to know about ChromaDB or Gemini internals.
- Automatic collection routing: given a topic keyword, it picks the
  right collection without hardcoded if/else chains.
- Re-ranking: results are post-processed to filter low-confidence matches
  below a similarity threshold.
- Context window management: limits total context length for LLM input.
"""

from typing import Any

from app.core.exceptions import AIServiceError, VectorDBError
from app.core.logging import get_logger
from app.rag.collections import (
    TOPIC_TO_COLLECTION,
    VALID_COLLECTION_NAMES,
    CollectionConfig,
)
from app.rag.embedder import GeminiEmbedder
from app.rag.vector_store import VectorStore

logger = get_logger(__name__)

# Minimum similarity score to include a result (0.0 - 1.0)
MIN_SIMILARITY_THRESHOLD = 0.35

# Maximum total characters of retrieved context sent to LLM
MAX_CONTEXT_CHARS = 8000


class RetrievalResult:
    """
    Wraps retrieval results with helper methods for LLM context formatting.
    """

    def __init__(self, results: list[dict[str, Any]], query: str) -> None:
        self.results = results
        self.query = query

    def __len__(self) -> int:
        return len(self.results)

    def __bool__(self) -> bool:
        return bool(self.results)

    def to_context_string(self, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        Format retrieved chunks into a context string for LLM prompting.

        Each chunk is formatted with its source and similarity score.
        Total length is capped to avoid exceeding LLM context windows.

        Args:
            max_chars: Maximum total characters of context.

        Returns:
            Formatted context string ready for LLM prompt injection.
        """
        if not self.results:
            return "No relevant documentation found for this query."

        context_parts: list[str] = []
        total_chars = 0

        for i, result in enumerate(self.results, start=1):
            collection = result.get("collection", "knowledge_base")
            score = result.get("score", 0)
            content = result.get("content", "")
            page = result.get("metadata", {}).get("page_number", "")
            page_info = f" (Page {page})" if page else ""

            chunk_text = (
                f"[Source {i}: {collection}{page_info} | Relevance: {score:.0%}]\n"
                f"{content}\n"
            )

            if total_chars + len(chunk_text) > max_chars:
                logger.debug(
                    "Context limit reached at chunk %d/%d.", i, len(self.results)
                )
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n---\n".join(context_parts)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Return results as a clean list of dicts for API responses."""
        return [
            {
                "content": r["content"],
                "score": r["score"],
                "collection": r["collection"],
                "metadata": r.get("metadata", {}),
            }
            for r in self.results
        ]


class Retriever:
    """
    High-level semantic retrieval interface for the CodeMentor AI agent.

    Usage:
        retriever = Retriever()

        # Search by topic keyword
        results = await retriever.retrieve_by_topic("python sorting algorithms", top_k=5)

        # Search in a specific collection
        results = await retriever.retrieve_from_collection(
            query="binary search implementation",
            collection_name="dsa_notes",
        )

        # Cross-collection search
        results = await retriever.retrieve_cross_collection(
            query="FastAPI async SQLAlchemy",
            collection_names=["fastapi_docs", "sqlalchemy_docs", "python_docs"],
        )
    """

    def __init__(
        self,
        embedder: GeminiEmbedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._embedder = embedder or GeminiEmbedder()
        self._vector_store = vector_store or VectorStore()

    def _resolve_collections(self, topic: str) -> list[str]:
        """
        Map a topic string to relevant collection names.

        Tries exact match first, then substring matching for compound topics.
        Falls back to top 3 general collections if no match found.

        Args:
            topic: Topic or query string to map.

        Returns:
            List of collection names to search.
        """
        topic_lower = topic.lower()

        # Direct lookup
        if topic_lower in TOPIC_TO_COLLECTION:
            return [TOPIC_TO_COLLECTION[topic_lower]]

        # Substring matching (handles "python binary search" → python_docs + dsa_notes)
        matched = []
        for keyword, col_name in TOPIC_TO_COLLECTION.items():
            if keyword in topic_lower and col_name not in matched:
                matched.append(col_name)

        if matched:
            logger.debug("Resolved topic '%s' to collections: %s", topic, matched)
            return matched[:4]  # Cap at 4 collections

        # Fallback: search broad collections
        logger.debug(
            "No specific collection matched for topic '%s'. Using defaults.", topic
        )
        return ["dsa_notes", "best_practices", "python_docs"]

    def _filter_by_threshold(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Remove results below the minimum similarity threshold.
        Prevents irrelevant chunks from polluting LLM context.
        """
        filtered = [r for r in results if r.get("score", 0) >= MIN_SIMILARITY_THRESHOLD]
        removed = len(results) - len(filtered)
        if removed:
            logger.debug(
                "Filtered %d results below threshold %.2f.", removed, MIN_SIMILARITY_THRESHOLD
            )
        return filtered

    async def retrieve_from_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> RetrievalResult:
        """
        Retrieve semantically similar chunks from a specific collection.

        Args:
            query:           User's search query.
            collection_name: ChromaDB collection to search.
            top_k:           Maximum number of results.
            where:           Optional metadata filter.

        Returns:
            RetrievalResult wrapping the search results.
        """
        if collection_name not in VALID_COLLECTION_NAMES:
            raise VectorDBError(
                f"Invalid collection: '{collection_name}'. "
                f"Valid: {VALID_COLLECTION_NAMES}"
            )

        logger.info(
            "Retrieving from collection='%s' query='%s' top_k=%d",
            collection_name, query[:80], top_k,
        )

        # Embed query
        query_embedding = await self._embedder.embed_query_async(query)

        # Search
        raw_results = self._vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        # Filter low-confidence results
        filtered = self._filter_by_threshold(raw_results)

        logger.info(
            "Retrieved %d results (%d after threshold filtering).",
            len(raw_results), len(filtered)
        )
        return RetrievalResult(results=filtered, query=query)

    async def retrieve_by_topic(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Automatically resolve the topic to collections and retrieve.

        This is the primary entry point for the LangGraph agent.
        The agent doesn't need to know which collection to use —
        the retriever figures it out from the query content.

        Args:
            query:  User's query or question.
            top_k:  Total results to return (across collections).

        Returns:
            RetrievalResult with merged, ranked results.
        """
        collections = self._resolve_collections(query)
        logger.info(
            "retrieve_by_topic: query='%s' → collections=%s", query[:80], collections
        )

        query_embedding = await self._embedder.embed_query_async(query)

        raw_results = self._vector_store.search_multiple_collections(
            collection_names=collections,
            query_embedding=query_embedding,
            n_results_per_collection=max(3, top_k // len(collections)),
        )

        filtered = self._filter_by_threshold(raw_results)

        # Sort by score, keep top_k
        filtered.sort(key=lambda x: x["score"], reverse=True)
        filtered = filtered[:top_k]

        logger.info(
            "Topic retrieval complete: %d results from %d collections.",
            len(filtered), len(collections)
        )
        return RetrievalResult(results=filtered, query=query)

    async def retrieve_cross_collection(
        self,
        query: str,
        collection_names: list[str],
        top_k: int = 8,
    ) -> RetrievalResult:
        """
        Explicitly search across a list of collections and merge results.

        Used when the agent knows which domains are relevant.

        Args:
            query:            User's search query.
            collection_names: Explicit list of collections to search.
            top_k:            Total results after merging.

        Returns:
            Merged and ranked RetrievalResult.
        """
        logger.info(
            "Cross-collection retrieval: collections=%s query='%s'",
            collection_names, query[:80]
        )

        query_embedding = await self._embedder.embed_query_async(query)

        raw_results = self._vector_store.search_multiple_collections(
            collection_names=collection_names,
            query_embedding=query_embedding,
            n_results_per_collection=max(2, top_k // max(len(collection_names), 1)),
        )

        filtered = self._filter_by_threshold(raw_results)
        filtered.sort(key=lambda x: x["score"], reverse=True)
        filtered = filtered[:top_k]

        return RetrievalResult(results=filtered, query=query)
