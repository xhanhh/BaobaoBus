"""Deterministic decimal place-value, rounding, and zero rules."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_DECIMAL_PLACE = re.compile(
    rf"(?P<number>{NUMBER})中[,，]?"
    rf"(?P<place>十分位|百分位)上的数是"
)
_ROUND_DECIMAL = re.compile(
    rf"把(?P<number>{NUMBER})精确到(?P<place>十分之一|百分之一)"
)
_REMOVABLE_ZERO = re.compile(r"不改变数的大小.*?可以去掉[“\"']?0[”\"']?")


def solve_decimal_concept(question: Question) -> SolveDecision | None:
    place = _DECIMAL_PLACE.search(question.text)
    if place:
        whole, dot, fraction = place.group("number").partition(".")
        del whole
        position = 0 if place.group("place") == "十分位" else 1
        if not dot or len(fraction) <= position:
            digit = 0
        else:
            digit = int(fraction[position])
        values = tuple(parse_number(option) for option in question.options)
        return match_unique(Decimal(digit), values, f"decimal {place.group('place')}")

    rounding = _ROUND_DECIMAL.search(question.text)
    if rounding:
        places = 1 if rounding.group("place") == "十分之一" else 2
        quantum = Decimal(1).scaleb(-places)
        try:
            rounded = Decimal(rounding.group("number")).quantize(
                quantum,
                rounding=ROUND_HALF_UP,
            )
        except InvalidOperation:
            return None
        expected = f"{rounded:.{places}f}"
        exact_matches = [
            index
            for index, option in enumerate(question.options)
            if option == expected
        ]
        if len(exact_matches) == 1:
            return SolveDecision(
                exact_matches[0],
                "rule",
                f"decimal rounding: {expected}",
            )
        values = tuple(parse_number(option) for option in question.options)
        return match_unique(rounded, values, f"decimal rounding: {expected}")

    if _REMOVABLE_ZERO.search(question.text):
        matches = [
            index
            for index, option in enumerate(question.options)
            if _has_removable_decimal_zero(option)
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "removable decimal trailing zero")
    return None


def _has_removable_decimal_zero(option: str) -> bool:
    if "." not in option or not option.endswith("0"):
        return False
    shortened = option.rstrip("0").rstrip(".")
    try:
        return Decimal(option) == Decimal(shortened)
    except InvalidOperation:
        return False
