"""
CodeMentor AI - Gemini Embedding Engine
=========================================
Generates vector embeddings using Google Gemini's text-embedding-004 model.

Design Rationale:
- text-embedding-004 produces 768-dimensional dense vectors, strong for code+prose.
- Task types: RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for search queries.
  Using the correct task type significantly improves retrieval accuracy.
- Batch processing: Gemini allows up to 100 texts per batch API call.
  We batch to reduce API round trips and stay within rate limits.
- Exponential backoff (tenacity): Gemini API has rate limits — retries are essential
  for production reliability.
- Caching: We cache embeddings by content hash to avoid re-embedding identical text.
"""

import asyncio
import hashlib
import time
from typing import Any

from google import genai
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger
from app.rag.chunker import TextChunk

logger = get_logger(__name__)
settings = get_settings()

# Gemini embedding dimensions for text-embedding-004
EMBEDDING_DIMENSIONS = 768

# Maximum texts per API batch call
BATCH_SIZE = 100


class EmbeddingCache:
    """
    In-memory LRU-style embedding cache.
    Avoids re-embedding identical text chunks (e.g., re-uploading same PDF).
    Keyed by SHA-256 hash of text content.
    """

    def __init__(self, max_size: int = 5000) -> None:
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> list[float] | None:
        key = hashlib.sha256(text.encode()).hexdigest()
        embedding = self._cache.get(key)
        if embedding:
            self._hits += 1
        else:
            self._misses += 1
        return embedding

    def set(self, text: str, embedding: list[float]) -> None:
        if len(self._cache) >= self._max_size:
            # Evict oldest entry (simple FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        key = hashlib.sha256(text.encode()).hexdigest()
        self._cache[key] = embedding

    @property
    def stats(self) -> dict[str, int]:
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
        }


class GeminiEmbedder:
    """
    Generates text embeddings using Google Gemini API.

    Task types:
    - RETRIEVAL_DOCUMENT: for embedding documents being stored in ChromaDB
    - RETRIEVAL_QUERY:    for embedding user search queries at retrieval time
    - SEMANTIC_SIMILARITY: for comparing two pieces of text
    """

    TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    TASK_QUERY = "RETRIEVAL_QUERY"
    TASK_SIMILARITY = "SEMANTIC_SIMILARITY"

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model
        self._cache = EmbeddingCache()
        logger.info("GeminiEmbedder initialized with model: %s", self._model)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _embed_batch_sync(
        self,
        texts: list[str],
        task_type: str,
    ) -> list[list[float]]:
        """
        Embed a batch of texts synchronously with retry logic.

        Args:
            texts:     List of text strings to embed.
            task_type: Gemini task type string.

        Returns:
            List of embedding vectors, one per input text.
        """
        try:
            result = self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            )
            # The new SDK returns an object with .embeddings as a list of ContentEmbedding
            embeddings = [list(e.values) for e in result.embeddings]
            return embeddings

        except Exception as exc:
            logger.error("Gemini embedding API error: %s", exc)
            raise AIServiceError(f"Embedding generation failed: {exc}") from exc

    def embed_texts(
        self,
        texts: list[str],
        task_type: str = TASK_DOCUMENT,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts, with caching and batching.

        Args:
            texts:     Texts to embed.
            task_type: Gemini task type (DOCUMENT for storage, QUERY for search).

        Returns:
            List of embedding vectors (each vector is 768 floats).
        """
        if not texts:
            return []

        embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                embeddings[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.info(
                "Generating embeddings: %d total, %d cached, %d to embed.",
                len(texts), len(texts) - len(uncached_texts), len(uncached_texts)
            )

            # Process in batches
            all_new_embeddings: list[list[float]] = []
            for batch_start in range(0, len(uncached_texts), BATCH_SIZE):
                batch = uncached_texts[batch_start: batch_start + BATCH_SIZE]
                logger.debug(
                    "Embedding batch %d-%d...",
                    batch_start, batch_start + len(batch)
                )
                batch_embeddings = self._embed_batch_sync(batch, task_type)
                all_new_embeddings.extend(batch_embeddings)

                # Rate limiting: small sleep between batches
                if batch_start + BATCH_SIZE < len(uncached_texts):
                    time.sleep(0.5)

            # Store in cache and fill results
            for idx, (text, embedding) in enumerate(
                zip(uncached_texts, all_new_embeddings)
            ):
                self._cache.set(text, embedding)
                embeddings[uncached_indices[idx]] = embedding

        logger.debug("Cache stats: %s", self._cache.stats)
        return embeddings  # type: ignore[return-value]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query using RETRIEVAL_QUERY task type.

        Always use this for user search queries (not document storage).
        The RETRIEVAL_QUERY task type is optimized for asymmetric search.

        Args:
            query: User's search query string.

        Returns:
            Single embedding vector (768 floats).
        """
        if not query or not query.strip():
            raise AIServiceError("Cannot embed an empty query.")

        result = self.embed_texts([query.strip()], task_type=self.TASK_QUERY)
        return result[0]

    def embed_chunks(self, chunks: list[TextChunk]) -> list[list[float]]:
        """
        Embed a list of TextChunk objects for document storage.

        Args:
            chunks: List of text chunks from the chunker.

        Returns:
            List of embedding vectors corresponding to each chunk.
        """
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts, task_type=self.TASK_DOCUMENT)

    async def embed_query_async(self, query: str) -> list[float]:
        """
        Async wrapper for embed_query — for use in async FastAPI route handlers.
        Runs the blocking Gemini call in a thread pool to not block the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, query)

    async def embed_chunks_async(
        self, chunks: list[TextChunk]
    ) -> list[list[float]]:
        """
        Async wrapper for embed_chunks.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_chunks, chunks)
