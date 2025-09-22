# Inspect AI version of the "Calculate" tool

from inspect_ai.tool import tool

@tool
async def calculate(expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'.
                The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            A string with the numeric result rounded to 2 decimals, or an error message.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            return "Error: invalid characters in expression"
        try:
            return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        except Exception as e:
            return f"Error: {e}"
