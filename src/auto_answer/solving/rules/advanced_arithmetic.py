"""Safe rules for upper-primary arithmetic choice questions."""

from __future__ import annotations

import ast
import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, evaluate_ast, match_unique, parse_number

_PRODUCT_CLOSEST = re.compile(
    rf"与(?P<first>{NUMBER})\*(?P<second>{NUMBER})的积最接近"
)
_REMOVE_PARENTHESES = re.compile(r"去掉括号不改变(?:计算结果|运算顺序)")
_VARIABLE_PRODUCT = re.compile(
    rf"若A\*(?P<first_factor>{NUMBER})=(?P<first_product>{NUMBER})"
    rf"[,，]则A\*(?P<second_factor>{NUMBER})="
)
_VERTICAL_TENS_PARTIAL = re.compile(
    rf"用竖式计算(?P<multiplicand>{NUMBER})\*(?P<multiplier>\d{{2}})时[,，]"
    rf"乘数(?P=multiplier)十位上的(?P<tens>\d)乘"
    rf"(?P=multiplicand)得"
)
_EXPLICIT_EXPRESSION = re.compile(
    r"^(?:计算)?(?P<expression>[\d+\-*/().]+)="
)
_EQUAL_PRODUCT_EXPRESSION = re.compile(
    rf"与(?P<target>{NUMBER}\*{NUMBER})的积一样的算式"
)


def solve_advanced_arithmetic(question: Question) -> SolveDecision | None:
    values = tuple(parse_number(option) for option in question.options)

    explicit = _EXPLICIT_EXPRESSION.search(question.text)
    if explicit:
        target = _safe_expression_value(explicit.group("expression"))
        if target is not None:
            return match_unique(target, values, "explicit arithmetic expression")

    equal_product = _EQUAL_PRODUCT_EXPRESSION.search(question.text)
    if equal_product:
        target = _safe_expression_value(equal_product.group("target"))
        option_values = tuple(
            _safe_expression_value(option) for option in question.options
        )
        if target is not None:
            return match_unique(target, option_values, "equivalent product")

    closest = _PRODUCT_CLOSEST.search(question.text)
    if closest:
        product = Decimal(closest.group("first")) * Decimal(closest.group("second"))
        candidates = [
            (index, value, abs(value - product))
            for index, value in enumerate(values)
            if value is not None
        ]
        if len(candidates) != len(question.options):
            return None
        minimum = min(candidate[2] for candidate in candidates)
        matches = [candidate[0] for candidate in candidates if candidate[2] == minimum]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"closest option to product: {product}",
            )
        return None

    if _REMOVE_PARENTHESES.search(question.text):
        matches = []
        for index, option in enumerate(question.options):
            original = _safe_expression_tree(option)
            without = _safe_expression_tree(
                option.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            )
            if original is not None and original == without:
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "parentheses do not change result")
        return None

    variable = _VARIABLE_PRODUCT.search(question.text)
    if variable:
        first_factor = Decimal(variable.group("first_factor"))
        if first_factor == 0:
            return None
        target = (
            Decimal(variable.group("first_product"))
            / first_factor
            * Decimal(variable.group("second_factor"))
        )
        return match_unique(target, values, "scaled variable product")

    vertical = _VERTICAL_TENS_PARTIAL.search(question.text)
    if vertical:
        multiplier = vertical.group("multiplier")
        tens = int(vertical.group("tens"))
        if int(multiplier[0]) != tens:
            return None
        target = Decimal(vertical.group("multiplicand")) * tens * 10
        return match_unique(target, values, "vertical multiplication tens partial")
    return None


def _safe_expression_tree(text: str) -> str | None:
    normalized = text.replace("[", "(").replace("]", ")")
    if re.fullmatch(r"[\d+\-*/(). ]+", normalized) is None:
        return None
    try:
        return ast.dump(ast.parse(normalized, mode="eval").body, include_attributes=False)
    except SyntaxError:
        return None


def _safe_expression_value(text: str) -> Decimal | None:
    normalized = text.replace("[", "(").replace("]", ")")
    if re.fullmatch(r"[\d+\-*/(). ]+", normalized) is None:
        return None
    try:
        return evaluate_ast(ast.parse(normalized, mode="eval").body)
    except (SyntaxError, ValueError, ArithmeticError):
        return None
