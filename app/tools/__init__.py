"""
CodeMentor AI - Tool Registry
==============================
Central registry: instantiates all tools and provides lookup functions.

Usage:
    from app.tools import get_all_tools, get_tool_by_name

    tools = get_all_tools()                        # All tools list
    tool  = get_tool_by_name("calculate")          # Single tool lookup
    funcs = [t.to_gemini_function() for t in tools]  # Gemini declarations
"""

from app.tools.base import BaseTool
from app.tools.calculator_tool import CalculatorTool
from app.tools.code_analyzer_tool import CodeAnalyzerTool
from app.tools.rag_tool import KnowledgeBaseSearchTool
from app.tools.web_search_tool import WebSearchTool

# Singleton tool instances
_TOOLS: list[BaseTool] = [
    KnowledgeBaseSearchTool(),
    CalculatorTool(),
    CodeAnalyzerTool(),
    WebSearchTool(),
]

# Name → tool lookup map
_TOOL_MAP: dict[str, BaseTool] = {tool.name: tool for tool in _TOOLS}


def get_all_tools() -> list[BaseTool]:
    """Return all registered tool instances."""
    return _TOOLS


def get_tool_by_name(name: str) -> BaseTool | None:
    """Look up a tool by its name. Returns None if not found."""
    return _TOOL_MAP.get(name)


def get_gemini_tool_declarations() -> list[dict]:
    """
    Return all tools as Gemini function declarations.
    Pass this directly to GenerativeModel(tools=[...]).
    """
    return [tool.to_gemini_function() for tool in _TOOLS]


__all__ = [
    "BaseTool",
    "KnowledgeBaseSearchTool",
    "CalculatorTool",
    "CodeAnalyzerTool",
    "WebSearchTool",
    "get_all_tools",
    "get_tool_by_name",
    "get_gemini_tool_declarations",
]
