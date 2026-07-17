"""
CodeMentor AI - RAG Knowledge Base Search Tool
================================================
Wraps RAGPipeline.search() as an agent-callable tool.

The agent uses this when it determines the user's query
requires retrieving information from the programming knowledge base.
The tool handles collection routing automatically.

Design Rationale:
- Stateless wrapper: the RAGPipeline singleton is fetched on each call.
- Returns a formatted context string (not raw dicts) so the agent's
  LLM can directly read and use the retrieved content.
- Graceful degradation: if ChromaDB is unavailable, returns a clear
  error string so the agent can still attempt a direct answer.
"""

from typing import Any

from app.core.logging import get_logger
from app.rag.pipeline import get_rag_pipeline
from app.tools.base import BaseTool

logger = get_logger(__name__)


class KnowledgeBaseSearchTool(BaseTool):
    """
    Search the programming knowledge base using semantic retrieval.

    The agent calls this when it needs factual information about:
    - Algorithms and data structures
    - Programming language syntax/features
    - Framework documentation (FastAPI, SQLAlchemy, etc.)
    - System design concepts
    - Interview preparation topics
    - Debugging patterns and common errors
    """

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "Search the programming knowledge base for relevant documentation, "
            "algorithms, data structures, framework guides, system design concepts, "
            "and interview preparation material. Use this when the user asks about "
            "specific programming topics, algorithms, language features, or needs "
            "conceptual explanations backed by documentation. "
            "Optionally specify a collection to narrow the search domain."
        )

    @property
    def args_schema(self) -> dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Be specific and descriptive. "
                    "Example: 'quicksort algorithm time complexity python implementation'"
                ),
            },
            "collection": {
                "type": "string",
                "description": (
                    "Optional. Specific collection to search. "
                    "Valid values: python_docs, java_docs, cpp_docs, javascript_docs, "
                    "sql_docs, fastapi_docs, sqlalchemy_docs, docker_docs, git_docs, "
                    "dsa_notes, algorithms, system_design, os_notes, dbms_notes, "
                    "networks_notes, interview_qna, best_practices, common_errors. "
                    "Leave empty to auto-route based on query content."
                ),
            },
        }

    async def run(self, query: str, collection: str = "") -> str:
        """
        Execute semantic search against the knowledge base.

        Args:
            query:      Search query string.
            collection: Optional specific collection name.

        Returns:
            Formatted context string with retrieved chunks, or error message.
        """
        logger.info(
            "KnowledgeBaseSearchTool.run: query='%s' collection='%s'",
            query[:80], collection or "auto"
        )

        try:
            pipeline = get_rag_pipeline()
            result = await pipeline.search(
                query=query,
                collection_name=collection if collection else None,
                top_k=5,
            )

            if not result:
                return (
                    f"No relevant documentation found for: '{query}'. "
                    "The knowledge base may not contain information on this topic yet. "
                    "Try rephrasing or use general knowledge to answer."
                )

            context = result.to_context_string()
            logger.info(
                "KnowledgeBaseSearchTool: retrieved %d chunks for query='%s'",
                len(result), query[:60]
            )
            return f"Retrieved knowledge base results for '{query}':\n\n{context}"

        except Exception as exc:
            logger.error("KnowledgeBaseSearchTool error: %s", exc)
            return (
                f"Knowledge base search failed: {exc}. "
                "The vector database may be unavailable. "
                "Please answer using your general programming knowledge."
            )
