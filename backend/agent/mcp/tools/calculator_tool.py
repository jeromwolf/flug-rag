"""Simple calculator tool for numeric computations."""

import ast
import operator

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)

# Safe operators for evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval(expr: str) -> float:
    """Safely evaluate a mathematical expression."""
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric values allowed")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](_eval_node(node.operand))
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Safe mathematical expression evaluator."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="수식을 계산합니다. 사칙연산, 괄호, 지수 등 지원",
            category="utility",
            help_text=(
                "수학 수식을 안전하게 평가합니다.\n"
                "지원 연산: +, -, *, /, ** (거듭제곱), % (나머지), 괄호\n"
                "예시: '(100 + 200) * 1.1', '2 ** 10', '355 / 113'\n"
                "주의: 변수, 함수 호출은 지원하지 않습니다."
            ),
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ToolParamType.STRING,
                    description="계산할 수식 (예: '(100 + 200) * 1.1')",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        expression = kwargs.get("expression", "")
        if not expression:
            return ToolResult(success=False, error="expression is required")

        try:
            result = safe_eval(expression)
            return ToolResult(
                success=True,
                data={"expression": expression, "result": result},
            )
        except (ValueError, ZeroDivisionError, SyntaxError) as e:
            return ToolResult(success=False, error=f"Calculation error: {e}")
