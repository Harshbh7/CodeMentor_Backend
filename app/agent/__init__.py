"""
CodeMentor AI - Agent Package
================================
Public interface for the agent module.

Primary entry point:
    from app.agent import run_agent

    state = await run_agent(query="What is quicksort?")
    print(state["final_answer"])
"""

from app.agent.graph import get_agent_graph, run_agent
from app.agent.state import AgentState, MAX_AGENT_ITERATIONS

__all__ = [
    "run_agent",
    "get_agent_graph",
    "AgentState",
    "MAX_AGENT_ITERATIONS",
]
