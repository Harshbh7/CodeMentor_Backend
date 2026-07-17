"""
CodeMentor AI - Abstract Tool Base Class
=========================================
Defines the interface that all agent tools must implement.

Design Rationale:
- All tools share a consistent interface (name, description, run).
- `to_gemini_function()` converts each tool's schema into the exact
  format Gemini's function-calling API expects — so the LLM knows
  what tools are available and what arguments they take.
- Tools are pure functions: given inputs → return a string result.
  The agent decides how to use that result.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Abstract base class for all CodeMentor AI agent tools.

    Each tool must provide:
    - name: unique identifier used in LLM function calling
    - description: natural language description for the LLM to understand
      when and why to use this tool
    - args_schema: dict describing the tool's parameters (JSON Schema format)
    - run(): async coroutine that executes the tool
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier. Used as the function name in Gemini calls."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Clear description of what this tool does.
        This is sent directly to the LLM — be precise about when to use it.
        """
        ...

    @property
    @abstractmethod
    def args_schema(self) -> dict[str, Any]:
        """
        JSON Schema for the tool's input parameters.
        Used to construct the Gemini function declaration.

        Example:
            {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            }
        """
        ...

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments matching args_schema.

        Returns:
            String result to be fed back into the agent's conversation.
            Always returns a string — even on error (error message string).
        """
        ...

    def to_gemini_function(self) -> dict[str, Any]:
        """
        Convert this tool to Gemini's function declaration format.

        Returns a dict that can be passed directly to:
            genai.GenerativeModel(tools=[tool.to_gemini_function()])

        Format follows the Gemini API function_declarations schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.args_schema,
                "required": list(self.args_schema.keys()),
            },
        }

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
