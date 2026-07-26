"""Shared safe numeric helpers for deterministic rules."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from ...core.models import SolveDecision

NUMBER = r"[-+]?\d+(?:\.\d+)?"
EXPRESSION = rf"{NUMBER}(?:[+\-*/]\d+(?:\.\d+)?)+"
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def parse_number(value: str) -> Decimal | None:
    cleaned = value.strip().rstrip(".。")
    if re.fullmatch(NUMBER, cleaned) is None:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def evaluate_expression(text: str) -> Decimal | None:
    expression = text.strip().rstrip(".。")
    if re.fullmatch(EXPRESSION, expression) is None:
        return None
    try:
        return evaluate_ast(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ArithmeticError, InvalidOperation):
        return None


def evaluate_numeric_form(text: str) -> Decimal | None:
    parsed = parse_number(text)
    return parsed if parsed is not None else evaluate_expression(text)


def evaluate_ast(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            evaluate_ast(node.left),
            evaluate_ast(node.right),
        )
    raise ValueError("unsupported expression")


def match_unique(
    target: Decimal,
    options: tuple[Decimal | None, ...],
    reason: str,
) -> SolveDecision | None:
    matches = [index for index, value in enumerate(options) if value == target]
    if len(matches) != 1:
        return None
    return SolveDecision(matches[0], "rule", f"{reason}: {target}")
