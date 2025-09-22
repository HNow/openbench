# Copyright Sierra

from typing import Any
from inspect_ai.tool import tool


@tool
def calculate():
    async def execute(expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'.
                The expression can contain numbers, operators (+, -, *, /),
                parentheses, and spaces.

        Returns:
            A string representing the result rounded to 2 decimal places, or an
            error message starting with 'Error:' if the expression is invalid.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            return "Error: invalid characters in expression"
        try:
            # Evaluate the mathematical expression safely
            return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        except Exception as e:
            return f"Error: {e}"

    return execute
