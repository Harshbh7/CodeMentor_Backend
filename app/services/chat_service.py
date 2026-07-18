"""
CodeMentor AI - Chat Service
==============================
Business logic layer between the /chat API endpoint and the LangGraph agent.

Responsibilities:
- Accepts ChatRequest, builds the initial agent state
- Calls run_agent() and waits for the final state
- Parses AgentState into a clean ChatResponse
- Handles errors gracefully (AI unavailable, timeout, etc.)

Design Rationale:
- Service layer keeps the API endpoint thin — it just validates and delegates.
- Source references are extracted from the retrieved_context string.
  This is simple and doesn't require the retriever to be called again.
- Processing time is measured here (wall-clock) for transparency.
"""

import time
import uuid
from typing import Any

from app.agent import run_agent
from app.agent.state import AgentState
from app.core.cache import ValkeyCache
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse, SourceReference, ToolCallInfo

logger = get_logger(__name__)


from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.chat_repository import ChatRepository

class ChatService:
    """
    Orchestrates the chat flow: request → agent → response.
    Saves history to PostgreSQL.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = ChatRepository(db)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request through the agentic RAG pipeline.

        Args:
            request: Validated ChatRequest from the API endpoint.

        Returns:
            ChatResponse with answer, sources, and tool usage info.
        """
        start_time = time.perf_counter()
        
        # Resolve or generate conversation ID
        conv_id_str = request.conversation_id or str(uuid.uuid4())
        conv_uuid = uuid.UUID(conv_id_str)

        logger.info(
            "ChatService.chat: query='%s' conversation_id='%s'",
            request.query[:80], conv_id_str
        )

        try:
            # 1. Check Valkey cache — only for standalone (first-turn) queries
            #    Multi-turn conversations are NOT cached since context differs per session.
            cache_key = ValkeyCache.generate_key(request.query, prefix="chat")
            is_new_conversation = not request.conversation_id

            if is_new_conversation:
                cached_data = ValkeyCache.get(cache_key)
                if cached_data:
                    processing_time = time.perf_counter() - start_time
                    logger.info(
                        "Cache HIT for query='%s' — skipping agent. Returned in %.3fs",
                        request.query[:60], processing_time,
                    )
                    # Reconstruct response from cached data
                    return ChatResponse(
                        answer=cached_data["answer"],
                        tool_calls_used=[
                            ToolCallInfo(**tc) for tc in cached_data.get("tool_calls_used", [])
                        ],
                        sources=[
                            SourceReference(**s) for s in cached_data.get("sources", [])
                        ],
                        conversation_id=conv_id_str,
                        iterations=cached_data.get("iterations", 0),
                        processing_time=round(processing_time, 3),
                        success=True,
                        error=None,
                        cached=True,
                    )

            # 2. Retrieve conversation & existing messages for multi-turn history context
            conversation = await self.repo.get_or_create_conversation(
                conversation_id=conv_uuid,
                title=request.query[:50] + ("..." if len(request.query) > 50 else "")
            )
            past_messages = await self.repo.get_messages(conv_uuid)

            # Format past messages for the agentic RAG pipeline
            history_context = []
            for msg in past_messages:
                history_context.append({"role": msg.role, "content": msg.content})

            # 3. Save the new user message to DB
            await self.repo.add_message(
                conversation_id=conv_uuid,
                role="user",
                content=request.query
            )

            # Update title on first message if title was default
            if len(past_messages) == 0:
                short_title = request.query[:42] + ("..." if len(request.query) > 42 else "")
                await self.repo.update_conversation_title(conv_uuid, short_title)

            # 4. Run the agentic RAG pipeline
            final_state: AgentState = await run_agent(
                query=request.query,
                conversation_history=history_context,
            )

            processing_time = time.perf_counter() - start_time

            # Extract final answer
            answer = final_state.get("final_answer") or ""
            if not answer:
                answer = (
                    "I was unable to generate a response. "
                    "Please try again or rephrase your question."
                )

            # 4. Save the assistant response to DB
            await self.repo.add_message(
                conversation_id=conv_uuid,
                role="assistant",
                content=answer
            )

            # Commit both messages and title updates
            await self.db.commit()

            # Build tool call info list
            tool_calls_used = [
                ToolCallInfo(
                    tool_name=log["tool_name"],
                    tool_input=log["tool_input"],
                    tool_output_preview=log["tool_output"],
                )
                for log in final_state.get("tool_calls_log", [])
            ]

            # Build source references from tool calls
            sources = self._extract_sources(final_state)
            error = final_state.get("error")

            logger.info(
                "ChatService.chat complete: time=%.2fs tools=%d sources=%d",
                processing_time, len(tool_calls_used), len(sources)
            )

            # 5. Store response in Valkey cache (only for fresh single-turn queries)
            if is_new_conversation and not error:
                ValkeyCache.set(
                    key=cache_key,
                    value={
                        "answer": answer,
                        "tool_calls_used": [
                            {
                                "tool_name": tc.tool_name,
                                "tool_input": tc.tool_input,
                                "tool_output_preview": tc.tool_output_preview,
                            }
                            for tc in tool_calls_used
                        ],
                        "sources": [
                            {
                                "content_preview": s.content_preview,
                                "collection": s.collection,
                                "relevance_score": s.relevance_score,
                                "page_number": s.page_number,
                            }
                            for s in sources
                        ],
                        "iterations": final_state.get("iterations", 0),
                    },
                    expire_seconds=3600,  # Cache for 1 hour
                )

            return ChatResponse(
                answer=answer,
                tool_calls_used=tool_calls_used,
                sources=sources,
                conversation_id=conv_id_str,
                iterations=final_state.get("iterations", 0),
                processing_time=round(processing_time, 3),
                success=True,
                error=error,
                cached=False,
            )

        except Exception as exc:
            processing_time = time.perf_counter() - start_time
            logger.exception("ChatService.chat error: %s", exc)

            return ChatResponse(
                answer=(
                    "An unexpected error occurred while processing your request. "
                    "Please check that your Gemini API key is configured correctly "
                    "and try again."
                ),
                tool_calls_used=[],
                sources=[],
                conversation_id=conv_id_str,
                iterations=0,
                processing_time=round(processing_time, 3),
                success=False,
                error=str(exc),
            )

    def _extract_sources(self, state: AgentState) -> list[SourceReference]:
        """
        Extract source references from RAG tool call logs.

        Parses the tool output strings from knowledge base searches
        to build SourceReference objects for the API response.

        Args:
            state: Final AgentState from the agent.

        Returns:
            List of SourceReference objects.
        """
        sources: list[SourceReference] = []

        for log in state.get("tool_calls_log", []):
            if log["tool_name"] != "search_knowledge_base":
                continue

            # The tool output is a formatted string — we parse it for display
            output = log.get("tool_output", "")
            tool_input = log.get("tool_input", {})
            collection = tool_input.get("collection", "knowledge_base")

            # Extract collection from output if auto-routed
            if "[Source" in output and "|" in output:
                # Parse lines like: [Source 1: dsa_notes (Page 5) | Relevance: 92%]
                import re
                source_blocks = re.findall(
                    r"\[Source \d+: (\w+)(?:\s*\(Page (\d+)\))?\s*\|\s*Relevance:\s*([\d]+)%\]\n(.*?)(?=\n---|$)",
                    output,
                    re.DOTALL,
                )
                for block in source_blocks[:5]:  # Max 5 sources
                    col_name, page, score_str, content = block
                    sources.append(SourceReference(
                        content_preview=content.strip()[:200],
                        collection=col_name or collection or "knowledge_base",
                        relevance_score=int(score_str) / 100,
                        page_number=int(page) if page else None,
                    ))
            elif output and not output.startswith("No relevant"):
                # Fallback: just return a simple source reference
                sources.append(SourceReference(
                    content_preview=output[:200],
                    collection=collection or "knowledge_base",
                    relevance_score=0.0,
                    page_number=None,
                ))

        return sources
