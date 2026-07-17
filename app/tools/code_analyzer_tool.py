"""
CodeMentor AI - Code Analyzer Tool
=====================================
Analyzes a code snippet using Gemini for time/space complexity,
bugs, and improvement suggestions.

The agent calls this when the user explicitly provides code and wants:
- Big-O complexity analysis
- Bug detection
- Code review / refactoring suggestions
- Language-specific best practices

Design Rationale:
- Delegates the heavy lifting to Gemini with a targeted prompt.
  We don't re-implement static analysis — we leverage the LLM's
  code understanding capabilities with a structured prompt.
- Separates code analysis from the main conversation so the agent
  can present analysis results distinctly from general answers.
- Returns structured text (not JSON) for easy reading in the chat response.
"""

from typing import Any

from google import genai
from google.genai import types as genai_types

from app.core.config import get_settings
from app.core.logging import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)

_CODE_ANALYSIS_PROMPT = """You are an expert code reviewer. Analyze the following {language} code and provide:

1. **Time Complexity**: Big-O notation with explanation
2. **Space Complexity**: Big-O notation with explanation
3. **Bugs / Issues**: List any bugs, edge cases not handled, or logic errors
4. **Improvements**: Concrete suggestions to make the code cleaner, faster, or more Pythonic
5. **Overall Assessment**: One sentence summary

Be concise but thorough. Format your response clearly with these exact sections.

Code to analyze:
```{language}
{code}
```"""


class CodeAnalyzerTool(BaseTool):
    """
    Analyzes code snippets for complexity, bugs, and improvements.

    Uses Gemini directly with a structured code-review prompt.
    The agent calls this when the user provides code and asks for analysis.
    """

    @property
    def name(self) -> str:
        return "analyze_code"

    @property
    def description(self) -> str:
        return (
            "Analyze a code snippet for time complexity, space complexity, "
            "potential bugs, and improvement suggestions. "
            "Use this when the user provides actual code and asks for: "
            "code review, Big-O analysis, bug detection, optimization advice, "
            "or best practice recommendations. "
            "Do NOT use this for general programming questions — use search_knowledge_base instead."
        )

    @property
    def args_schema(self) -> dict[str, Any]:
        return {
            "code": {
                "type": "string",
                "description": "The code snippet to analyze. Include the full function or class.",
            },
            "language": {
                "type": "string",
                "description": (
                    "Programming language of the code. "
                    "Examples: python, java, cpp, javascript, typescript, sql"
                ),
            },
        }

    async def run(self, code: str, language: str = "python") -> str:
        """
        Analyze code using Gemini with a structured review prompt.

        Args:
            code:     The code snippet to analyze.
            language: Programming language.

        Returns:
            Structured analysis with complexity, bugs, and suggestions.
        """
        logger.info(
            "CodeAnalyzerTool.run: language='%s' code_length=%d",
            language, len(code)
        )

        if not code.strip():
            return "No code provided. Please include a code snippet to analyze."

        if len(code) > 10_000:
            return (
                "Code snippet is too long (>10,000 chars). "
                "Please provide a smaller excerpt for analysis."
            )

        try:
            settings = get_settings()
            client = genai.Client(api_key=settings.gemini_api_key)

            prompt = _CODE_ANALYSIS_PROMPT.format(
                language=language,
                code=code,
            )

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            analysis = response.text.strip()

            logger.info("CodeAnalyzerTool: analysis complete for %d chars of code", len(code))
            return f"Code Analysis Result:\n\n{analysis}"

        except Exception as exc:
            logger.error("CodeAnalyzerTool error: %s", exc)
            return (
                f"Code analysis failed: {exc}. "
                "Please ensure the Gemini API key is configured correctly."
            )
