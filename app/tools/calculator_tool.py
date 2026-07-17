"""
CodeMentor AI - Safe Calculator Tool
======================================
Evaluates mathematical expressions safely without using eval().

The agent uses this for:
- Big-O complexity calculations (e.g., n*log(n) for n=1000)
- Time/space tradeoff comparisons
- Numeric answers to algorithm analysis questions

Design Rationale:
- Uses Python's ast module to parse and walk the expression tree.
  Only safe node types (numbers, operators, math functions) are allowed.
  This prevents arbitrary code execution unlike raw eval().
- Supports common math functions (log, sqrt, etc.) from the math module.
- Returns clear error messages so the agent can explain to the user
  why a calculation failed.
"""

import ast
import math
import operator
from typing import Any

from app.core.logging import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)

# Allowed binary operators
ALLOWED_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions
ALLOWED_FUNCTIONS: dict[str, Any] = {
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "factorial": math.factorial,
}

# Allowed constants
ALLOWED_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "n": 1,        # placeholder — users can use 'n' in expressions
}


class _SafeEvaluator(ast.NodeVisitor):
    """
    AST-based safe expression evaluator.
    Only allows numeric literals, binary/unary ops, and whitelisted functions.
    """

    def __init__(self, variables: dict[str, float] | None = None) -> None:
        self._vars = {**ALLOWED_NAMES, **(variables or {})}

    def evaluate(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
            return self.visit(tree.body)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(str(exc)) from exc
        except Exception as exc:
            raise ValueError(f"Cannot evaluate expression: {exc}") from exc

    def visit_Expression(self, node: ast.Expression) -> float:  # noqa: N802
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:  # noqa: N802
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Name(self, node: ast.Name) -> float:  # noqa: N802
        if node.id in self._vars:
            return float(self._vars[node.id])
        raise ValueError(f"Unknown variable: '{node.id}'")

    def visit_BinOp(self, node: ast.BinOp) -> float:  # noqa: N802
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return ALLOWED_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:  # noqa: N802
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        operand = self.visit(node.operand)
        return ALLOWED_OPERATORS[op_type](operand)

    def visit_Call(self, node: ast.Call) -> float:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function not allowed: '{func_name}'")
        args = [self.visit(arg) for arg in node.args]
        return float(ALLOWED_FUNCTIONS[func_name](*args))

    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """
    Safe mathematical expression evaluator.

    Supports standard arithmetic, math functions (log, sqrt, etc.),
    and variables like 'n' for algorithm complexity analysis.

    Example inputs:
        "n * log2(n)"              with n=1000 → 9965.78
        "n**2 / 2"                 with n=100  → 5000.0
        "sqrt(n)"                  with n=256  → 16.0
        "(1 + 1/n)**n"             with n=1000 → ~e
    """

    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression safely. "
            "Useful for computing Big-O complexity values (e.g., n*log2(n) for n=1000), "
            "space/time tradeoffs, numeric comparisons between algorithms, "
            "or any arithmetic needed to answer the user's question. "
            "Supports: +, -, *, /, **, %, log, log2, log10, sqrt, ceil, floor, "
            "abs, factorial, pi, e, and the variable 'n'."
        )

    @property
    def args_schema(self) -> dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "description": (
                    "A safe mathematical expression to evaluate. "
                    "Use 'n' as a variable. "
                    "Examples: 'n * log2(n)', 'n**2', 'sqrt(n) + log10(n)'"
                ),
            },
            "n": {
                "type": "number",
                "description": (
                    "Optional value for the variable 'n' in the expression. "
                    "Default is 1 if not provided."
                ),
            },
        }

    async def run(self, expression: str, n: float = 1.0) -> str:
        """
        Safely evaluate the mathematical expression.

        Args:
            expression: Math expression string.
            n:          Value for variable 'n'.

        Returns:
            Formatted result string or error description.
        """
        logger.info("CalculatorTool.run: expression='%s' n=%s", expression, n)

        try:
            evaluator = _SafeEvaluator(variables={"n": float(n)})
            result = evaluator.evaluate(expression)

            # Format result nicely
            if result == int(result) and abs(result) < 1e15:
                formatted = str(int(result))
            else:
                formatted = f"{result:.4f}"

            logger.info("CalculatorTool result: %s = %s", expression, formatted)
            return (
                f"Calculation result:\n"
                f"  Expression: {expression}\n"
                f"  n = {n}\n"
                f"  Result = {formatted}"
            )

        except ValueError as exc:
            return f"Calculation error: {exc}. Please check the expression syntax."
        except ZeroDivisionError:
            return "Calculation error: Division by zero."
        except OverflowError:
            return "Calculation error: Result is too large to compute."
        except Exception as exc:
            logger.error("CalculatorTool unexpected error: %s", exc)
            return f"Unexpected calculation error: {exc}"
