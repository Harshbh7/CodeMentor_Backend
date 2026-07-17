"""
CodeMentor AI - LangGraph Agent Graph
========================================
Assembles the StateGraph and compiles it into the runnable agent.

Graph Structure:
    START
      │
      ▼
  think_node  ◄──────────────────────┐
      │                              │
      ▼                              │
  should_continue ─── "tools" ─► tool_executor_node
      │
      └─── "end" ──► END

Design Rationale:
- `StateGraph(AgentState)` means LangGraph tracks the full state
  TypedDict across all nodes automatically.
- `tool_executor_node` is async (it awaits tool.run()), so LangGraph
  must be invoked with `await graph.ainvoke(...)`.
- The compiled graph is cached as a module-level singleton so it's
  only built once at startup (compilation involves schema validation).
- `run_agent()` is the single public entry point used by ChatService.
"""

from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import should_continue, think_node, tool_executor_node
from app.agent.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_agent_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph for the CodeMentor AI agent.

    Returns:
        Compiled LangGraph runnable (CompiledGraph).
    """
    logger.info("Building CodeMentor AI agent graph...")

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("think", think_node)
    graph.add_node("tools", tool_executor_node)

    # Entry point
    graph.add_edge(START, "think")

    # Conditional edge from think: either call tools or end
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools run, always go back to think for synthesis
    graph.add_edge("tools", "think")

    compiled = graph.compile()
    logger.info("Agent graph compiled successfully.")
    return compiled


# Module-level singleton — compiled once at import time
_agent_graph = None


def get_agent_graph() -> Any:
    """
    Return the singleton compiled agent graph.
    Builds it on first call; returns cached instance thereafter.
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


async def run_agent(query: str, conversation_history: list | None = None) -> AgentState:
    """
    Run the agentic RAG pipeline for a user query.

    This is the primary public entry point. ChatService calls this.

    Args:
        query:                The user's question or request.
        conversation_history: Optional prior messages (list of LangChain
                              message objects) for multi-turn context.
                              If None, starts a fresh conversation.

    Returns:
        Final AgentState after the graph completes, containing:
        - final_answer: The agent's response text
        - tool_calls_log: List of tools used
        - retrieved_context: Accumulated RAG results
        - messages: Full conversation history
    """
    logger.info("run_agent: query='%s'", query[:100])

    # Build initial state
    messages = list(conversation_history or [])
    messages.append(HumanMessage(content=query))

    initial_state: AgentState = {
        "messages": messages,
        "query": query,
        "tool_calls_log": [],
        "retrieved_context": "",
        "iterations": 0,
        "final_answer": None,
        "error": None,
    }

    # Run the graph
    graph = get_agent_graph()
    final_state: AgentState = await graph.ainvoke(initial_state)

    logger.info(
        "run_agent complete: tools_used=%d iterations=%d answer_len=%d",
        len(final_state.get("tool_calls_log", [])),
        final_state.get("iterations", 0),
        len(final_state.get("final_answer") or ""),
    )

    return final_state
