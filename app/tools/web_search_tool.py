"""
CodeMentor AI - Web Search Tool (DuckDuckGo)
=============================================
Fetches current information from the web using the
DuckDuckGo Instant Answer API.

The agent uses this when the user asks about:
- Latest library versions ("What's the latest FastAPI version?")
- Recent programming news or CVEs
- Topics not covered in the knowledge base

Design Rationale:
- DuckDuckGo Instant Answer API requires NO API key.
- Returns instant answers + related topics (not full web pages).
- This is intentionally limited: we supplement RAG, not replace it.
- httpx is used (already a project dependency) for async HTTP calls.
- Falls back to a "search the web manually" message on failure.
"""

from typing import Any

import httpx

from app.core.logging import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)

DDGO_API_URL = "https://api.duckduckgo.com/"
REQUEST_TIMEOUT = 8.0  # seconds


class WebSearchTool(BaseTool):
    """
    Fetch instant answers from DuckDuckGo for current information.

    This tool supplements the knowledge base for time-sensitive queries
    (e.g., version numbers, recent releases, CVEs, current best practices).
    It does NOT do full web scraping — only DuckDuckGo Instant Answers.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information not available in the knowledge base. "
            "Use this for: latest library versions, recent releases, current CVEs, "
            "up-to-date documentation links, or any question requiring real-time data. "
            "Do NOT use this for algorithmic explanations or programming concepts — "
            "use search_knowledge_base instead. "
            "Returns a brief instant answer from DuckDuckGo."
        )

    @property
    def args_schema(self) -> dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Be concise and specific. "
                    "Examples: 'FastAPI latest version 2024', "
                    "'Python 3.12 new features', "
                    "'SQLAlchemy 2.0 async changes'"
                ),
            },
        }

    async def run(self, query: str) -> str:
        """
        Call DuckDuckGo Instant Answer API and return formatted results.

        Args:
            query: Search query string.

        Returns:
            Formatted string with instant answer and related topics.
        """
        logger.info("WebSearchTool.run: query='%s'", query[:80])

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    DDGO_API_URL,
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1",
                    },
                    headers={"User-Agent": "CodeMentor-AI/1.0"},
                )
                response.raise_for_status()
                data = response.json()

        except httpx.TimeoutException:
            return (
                f"Web search timed out for query: '{query}'. "
                "Please try a more specific query or answer from general knowledge."
            )
        except httpx.HTTPError as exc:
            logger.warning("WebSearchTool HTTP error: %s", exc)
            return (
                f"Web search failed (HTTP error): {exc}. "
                "Please answer from your general knowledge."
            )
        except Exception as exc:
            logger.error("WebSearchTool unexpected error: %s", exc)
            return f"Web search failed: {exc}"

        # Parse DuckDuckGo response
        results: list[str] = []

        # Primary instant answer
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            source = data.get("AbstractSource", "")
            results.append(f"**{source}**: {abstract}" if source else abstract)

        # Answer (short one-liners)
        answer = data.get("Answer", "").strip()
        if answer and answer != abstract:
            results.append(f"**Quick Answer**: {answer}")

        # Related topics (top 3)
        related = data.get("RelatedTopics", [])
        topic_texts: list[str] = []
        for topic in related[:3]:
            if isinstance(topic, dict) and "Text" in topic:
                text = topic["Text"].strip()
                if text and len(text) < 300:
                    topic_texts.append(f"• {text}")

        if topic_texts:
            results.append("**Related**:\n" + "\n".join(topic_texts))

        # Definition
        definition = data.get("Definition", "").strip()
        if definition:
            def_source = data.get("DefinitionSource", "")
            results.append(
                f"**Definition ({def_source})**: {definition}"
                if def_source else f"**Definition**: {definition}"
            )

        if not results:
            logger.info("WebSearchTool: no instant answer found for '%s'", query)
            return (
                f"No instant answer found for: '{query}'. "
                "The knowledge base may not have this information either. "
                "Please answer from your general programming knowledge, "
                "and note that your training data has a knowledge cutoff."
            )

        output = f"Web search results for '{query}':\n\n" + "\n\n".join(results)
        logger.info("WebSearchTool: found %d result sections", len(results))
        return output
