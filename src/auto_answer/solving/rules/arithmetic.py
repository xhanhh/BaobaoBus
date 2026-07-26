"""Rules for explicit arithmetic, comparisons, sequences, and inequalities."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from ...core.models import Question, SolveDecision
from .common import (
    EXPRESSION,
    NUMBER,
    evaluate_ast,
    evaluate_expression,
    evaluate_numeric_form,
    match_unique,
    parse_number,
)

_BLANK = r"(?:\(\)|__|□)"
_BLANK_LEFT_INEQUALITY = re.compile(
    rf"(?P<blank>{_BLANK})(?P<operator>[+\-*/])(?P<number>{NUMBER})"
    rf"(?P<comparator>[<>])(?P<bound>{NUMBER})"
)
_BLANK_RIGHT_INEQUALITY = re.compile(
    rf"(?P<number>{NUMBER})(?P<operator>[+\-*/])(?P<blank>{_BLANK})"
    rf"(?P<comparator>[<>])(?P<bound>{NUMBER})"
)
_SYMBOL_OPERATORS: dict[str, Callable[[Decimal, Decimal], Decimal]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}
_DIRECT_EXTREME_QUESTION = re.compile(
    r"(?:下面|下列|这些|哪个|哪一个|四个选项).*?"
    r"(?:最大|最小|最多|最少|最高|最低)"
    r"|(?:最大|最小|最高|最低)(?:的)?(?:数|数字)(?:是|为|有)"
)
_EQUIVALENT_EXPRESSION = re.compile(
    rf"与(?P<target>{EXPRESSION})(?:的)?(?:结果|得数)(?:相等|相同)"
)
_EXPRESSION_NEAR_THRESHOLD = re.compile(
    rf"得数比?(?P<threshold>{NUMBER})(?P<direction>小一些|小一点|大一些|大一点)"
)
_EXPRESSION_THRESHOLD = re.compile(
    rf"得数(?P<direction>小于|大于)(?P<threshold>{NUMBER})的算式"
)
_COMPARISON_BLANK = re.compile(
    rf"(?P<left>{EXPRESSION}|{NUMBER})(?:□|口|\(\)|__)"
    rf"(?P<right>{EXPRESSION}|{NUMBER})"
)
_FOUR_TERM_ARITHMETIC_SEQUENCE = re.compile(
    rf"(?P<first>{NUMBER})[、,](?P<second>{NUMBER})[、,]"
    rf"(?:{_BLANK}|口)[、,](?P<fourth>{NUMBER})"
)
_ADDEND_CHANGES = re.compile(
    rf"一个加数(?P<first_direction>增加|减少)(?P<first>{NUMBER})[,，]"
    rf"另一个加数(?P<second_direction>增加|减少)(?P<second>{NUMBER})[,，]和"
)
_SUBTRACTION_CHANGES = re.compile(
    rf"被减数(?P<first_direction>增加|减少)(?P<first>{NUMBER})[,，]"
    rf"减数(?P<second_direction>增加|减少)(?P<second>{NUMBER})[,，]差"
)
_SUBTRAHEND_FROM_DIFFERENCE = re.compile(
    rf"被减数是(?P<minuend>{NUMBER})[,，]"
    rf"差是(?P<difference>{NUMBER})[,，]"
    rf"减数是(?:\(\)|多少|几)"
)
_SUBTRAHEND_FROM_REVERSED_FACTS = re.compile(
    rf"差是(?P<difference>{NUMBER})[,，]"
    rf"被减数是(?P<minuend>{NUMBER})[,，]"
    rf"减数是(?:\(\)|多少|几)"
)
_TRAILING_ARITHMETIC_SEQUENCE = re.compile(
    rf"(?:找规律[:：]?)?(?P<first>{NUMBER})[、,](?P<second>{NUMBER})[、,]"
    rf"(?P<third>{NUMBER})[、,](?P<fourth>{NUMBER})[、,](?:{_BLANK}|口)"
)


def solve_arithmetic_sequence(
    question: Question,
    option_values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    match = _FOUR_TERM_ARITHMETIC_SEQUENCE.search(question.text)
    if match is not None:
        first = Decimal(match.group("first"))
        second = Decimal(match.group("second"))
        fourth = Decimal(match.group("fourth"))
        difference = second - first
        missing = second + difference
        if fourth != missing + difference:
            return None
        return match_unique(missing, option_values, "arithmetic sequence")

    trailing = _TRAILING_ARITHMETIC_SEQUENCE.search(question.text)
    if trailing is None:
        return None
    terms = tuple(
        Decimal(trailing.group(name))
        for name in ("first", "second", "third", "fourth")
    )
    differences = tuple(
        right - left for left, right in zip(terms, terms[1:], strict=False)
    )
    if len(set(differences)) != 1:
        return None
    return match_unique(
        terms[-1] + differences[0],
        option_values,
        "arithmetic sequence",
    )


def solve_operation_change(question: Question) -> SolveDecision | None:
    missing_subtrahend = _SUBTRAHEND_FROM_DIFFERENCE.search(question.text)
    if missing_subtrahend is None:
        missing_subtrahend = _SUBTRAHEND_FROM_REVERSED_FACTS.search(question.text)
    if missing_subtrahend:
        minuend = Decimal(missing_subtrahend.group("minuend"))
        difference = Decimal(missing_subtrahend.group("difference"))
        target = minuend - difference
        values = tuple(parse_number(option) for option in question.options)
        return match_unique(target, values, "subtrahend from difference")

    match = _ADDEND_CHANGES.search(question.text)
    if match:
        change = _signed_change(
            match.group("first_direction"),
            Decimal(match.group("first")),
        ) + _signed_change(
            match.group("second_direction"),
            Decimal(match.group("second")),
        )
    else:
        match = _SUBTRACTION_CHANGES.search(question.text)
        if match is None:
            return None
        change = _signed_change(
            match.group("first_direction"),
            Decimal(match.group("first")),
        ) - _signed_change(
            match.group("second_direction"),
            Decimal(match.group("second")),
        )

    if change == 0:
        expected = "不变"
    elif change > 0:
        expected = f"增加{change}"
    else:
        expected = f"减少{abs(change)}"
    matches = [
        index for index, option in enumerate(question.options) if option == expected
    ]
    if len(matches) != 1:
        return None
    return SolveDecision(matches[0], "rule", f"operation change: {expected}")


def _signed_change(direction: str, value: Decimal) -> Decimal:
    return value if direction == "增加" else -value


def solve_comparison_symbol(question: Question) -> SolveDecision | None:
    if "比较大小" not in question.text and "比大小" not in question.text:
        return None
    match = _COMPARISON_BLANK.search(question.text)
    if match is None:
        return None
    left = evaluate_numeric_form(match.group("left"))
    right = evaluate_numeric_form(match.group("right"))
    if left is None or right is None:
        return None
    target = "<" if left < right else ">" if left > right else "="
    matches = [
        index for index, option in enumerate(question.options) if option == target
    ]
    if len(matches) != 1:
        return None
    return SolveDecision(
        matches[0],
        "rule",
        f"comparison expression: {left}{target}{right}",
    )


def solve_option_expression(question: Question) -> SolveDecision | None:
    values = tuple(evaluate_expression(option) for option in question.options)
    if any(value is None for value in values):
        return None
    numeric_values = tuple(value for value in values if value is not None)

    choose_max = "得数最大" in question.text
    choose_min = "得数最小" in question.text
    if choose_max != choose_min:
        target = max(numeric_values) if choose_max else min(numeric_values)
        matches = [
            index for index, value in enumerate(numeric_values) if value == target
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"option expression extreme: {target}",
            )
        return None

    equivalent = _EQUIVALENT_EXPRESSION.search(question.text)
    if equivalent:
        target = evaluate_expression(equivalent.group("target"))
        if target is None:
            return None
        matches = [
            index for index, value in enumerate(numeric_values) if value == target
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"equivalent option expression: {target}",
            )
        return None

    near = _EXPRESSION_NEAR_THRESHOLD.search(question.text)
    if near:
        threshold = Decimal(near.group("threshold"))
        below = near.group("direction").startswith("小")
        candidates = [
            (index, value)
            for index, value in enumerate(numeric_values)
            if (value < threshold if below else value > threshold)
        ]
        if not candidates:
            return None
        target = (
            max(value for _, value in candidates)
            if below
            else min(value for _, value in candidates)
        )
        matches = [index for index, value in candidates if value == target]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"nearest option expression: {target}",
            )
        return None

    threshold_match = _EXPRESSION_THRESHOLD.search(question.text)
    if threshold_match:
        threshold = Decimal(threshold_match.group("threshold"))
        predicate = (
            operator.lt
            if threshold_match.group("direction") == "小于"
            else operator.gt
        )
        matches = [
            index
            for index, value in enumerate(numeric_values)
            if predicate(value, threshold)
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                "unique option expression threshold",
            )
    return None


def solve_arithmetic_expression(text: str) -> Decimal | None:
    normalized = (
        text.replace("加", "+")
        .replace("减", "-")
        .replace("乘以", "*")
        .replace("乘", "*")
        .replace("除以", "/")
        .replace("除", "/")
    )
    expression_pattern = (
        r"(?<![\d.])[-+]?\d+(?:\.\d+)?"
        r"(?:[+\-*/]\d+(?:\.\d+)?)+(?![\d.])"
    )
    matches = re.findall(expression_pattern, normalized)
    if len(matches) != 1:
        return None
    try:
        return evaluate_ast(ast.parse(matches[0], mode="eval").body)
    except (SyntaxError, ValueError, ArithmeticError, InvalidOperation):
        return None


def solve_extreme(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    if _DIRECT_EXTREME_QUESTION.search(text) is None or any(
        value is None for value in values
    ):
        return None
    numeric = tuple(value for value in values if value is not None)
    choose_max = any(token in text for token in ("最大", "最多", "最高"))
    choose_min = any(token in text for token in ("最小", "最少", "最低"))
    if choose_max == choose_min:
        return None
    target = max(numeric) if choose_max else min(numeric)
    return match_unique(target, values, "numeric comparison")


def solve_inequality_blank(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    choose_max = "最大" in text
    choose_min = "最小" in text
    if choose_max == choose_min or any(value is None for value in values):
        return None

    match = _BLANK_LEFT_INEQUALITY.search(text)
    blank_on_left = match is not None
    if match is None:
        match = _BLANK_RIGHT_INEQUALITY.search(text)
    if match is None:
        return None

    operation = _SYMBOL_OPERATORS[match.group("operator")]
    fixed = Decimal(match.group("number"))
    bound = Decimal(match.group("bound"))
    comparator = operator.lt if match.group("comparator") == "<" else operator.gt

    matches: list[tuple[int, Decimal]] = []
    for index, value in enumerate(values):
        assert value is not None
        try:
            result = operation(value, fixed) if blank_on_left else operation(fixed, value)
        except (ArithmeticError, InvalidOperation):
            continue
        if comparator(result, bound):
            matches.append((index, value))
    if not matches:
        return None

    target = (
        max(value for _, value in matches)
        if choose_max
        else min(value for _, value in matches)
    )
    target_indexes = [index for index, value in matches if value == target]
    if len(target_indexes) != 1:
        return None
    return SolveDecision(
        target_indexes[0],
        "rule",
        f"inequality {'maximum' if choose_max else 'minimum'}: {target}",
    )


def solve_threshold(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    match = re.search(rf"(大于|小于)({NUMBER})(?:的数)?", text)
    if not match or any(value is None for value in values):
        return None
    threshold = Decimal(match.group(2))
    predicate = operator.gt if match.group(1) == "大于" else operator.lt
    matches = [index for index, value in enumerate(values) if predicate(value, threshold)]
    if len(matches) != 1:
        return None
    return SolveDecision(matches[0], "rule", "unique threshold comparison")


def solve_between(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    match = re.search(rf"比({NUMBER})大.*?比({NUMBER})小", text)
    if match:
        lower = Decimal(match.group(1))
        upper = Decimal(match.group(2))
    else:
        reverse = re.search(
            rf"比({NUMBER})少(?:一些|一点).*?比({NUMBER})多(?:一些|一点)",
            text,
        )
        if reverse is None:
            return None
        upper = Decimal(reverse.group(1))
        lower = Decimal(reverse.group(2))
    if any(value is None for value in values):
        return None
    if lower >= upper:
        return None
    matches = [
        index
        for index, value in enumerate(values)
        if value is not None and lower < value < upper
    ]
    if len(matches) != 1:
        return None
    return SolveDecision(matches[0], "rule", "unique value between bounds")
