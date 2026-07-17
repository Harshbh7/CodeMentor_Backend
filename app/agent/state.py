"""
CodeMentor AI - LangGraph Agent State
=======================================
Defines the shared state object passed between all nodes in the graph.

Design Rationale:
- TypedDict is used (not dataclass) because LangGraph expects a plain
  dict-like structure that it can safely copy and merge across nodes.
- `messages` holds the full conversation (HumanMessage + AIMessage +
  ToolMessage instances) — this is the canonical conversation history.
- `tool_calls_log` captures what tools were invoked and their results
  for returning in the API response (so clients know how the answer
  was constructed).
- `iterations` is a safety guard: if the LLM loops more than
  MAX_ITERATIONS times, we force it to produce a final answer.
"""

from typing import Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

# Maximum number of think→tool→think loops before forcing a final answer
MAX_AGENT_ITERATIONS = 6


class ToolCallLog(TypedDict):
    """Record of a single tool invocation during agent execution."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str


class AgentState(TypedDict):
    """
    Shared state for the CodeMentor AI LangGraph agent.

    Flows through every node in the graph. Each node receives this
    state, makes updates, and returns the modified state.

    Fields:
        messages:          Full conversation history as LangChain messages.
                           Includes HumanMessage, AIMessage, ToolMessage.
        query:             The original user query (immutable after start).
        tool_calls_log:    Log of all tool invocations this turn.
        retrieved_context: Accumulated RAG context (optional, informational).
        iterations:        Number of think→tool loops so far (safety counter).
        final_answer:      Set by think_node when the agent is done.
                           None while still processing.
        error:             Set if an unrecoverable error occurred.
    """
    messages: list[BaseMessage]
    query: str
    tool_calls_log: list[ToolCallLog]
    retrieved_context: str
    iterations: int
    final_answer: str | None
    error: str | None
