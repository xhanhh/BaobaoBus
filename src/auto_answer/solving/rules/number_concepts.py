"""Rules for number neighbors, place value, and counters."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import SolveDecision
from .common import NUMBER, match_unique

_NUMBER_NEIGHBORS = re.compile(
    rf"邻居(?:分别)?是(?P<first>{NUMBER})(?:和|、|,)(?P<second>{NUMBER})"
)
_COUNTER_TWO_DIGIT_EXTREME = re.compile(
    rf"计数器上.*?用(?P<count>{NUMBER})[颗棵个]珠子.*?"
    r"(?P<direction>最大|最小)(?:的)?两位数"
)
_PLACE_VALUE_COMPARE = re.compile(
    rf"(?:一个两位数|一个数).*?十位上(?:的数字)?是"
    rf"(?P<tens>最大的一位数|{NUMBER}).*?"
    rf"个位上的数字比十位上的数字(?P<direction>少|小|多|大)(?P<delta>{NUMBER}).*?"
    rf"这个数"
)
_PLACE_VALUE_OPERATION = re.compile(
    rf"一个两位数.*?十位上的数字是(?P<tens>最大的一位数|{NUMBER}).*?"
    rf"个位上的数字是十位上的数字(?P<operation>加上|减去)(?P<delta>{NUMBER}).*?"
    rf"这个数"
)
_ONES_TO_TENS_COMPARE = re.compile(
    rf"(?:一个数|一个两位数).*?个位上(?:的数(?:字)?)?是(?P<ones>{NUMBER}).*?"
    rf"十位上(?:的数(?:字)?)?比个位上(?:的数(?:字)?)?"
    rf"(?P<direction>少|小|多|大)(?P<delta>{NUMBER}).*?这个数"
)
_COUNTED_PLACE_UNITS = re.compile(
    rf"(?P<count>{NUMBER})个(?P<place>一|十|百|千)是(?:\(\)|多少|几)"
)
_SAME_DIGITS_WITH_LOWER_BOUND = re.compile(
    rf"个位和十位上的数字相同.*?比(?P<lower>{NUMBER})大"
)
_BOUNDED_PARITY = re.compile(
    rf"比(?P<lower>{NUMBER})大.*?比(?P<upper>{NUMBER})小.*?"
    rf"(?:并且|而且|,|，).*?是(?P<parity>偶数|双数|奇数|单数)"
)
_EXTREME_DIGIT_SUM = re.compile(r"最大的一位数和最小的两位数的和")


def solve_number_neighbor(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    match = _NUMBER_NEIGHBORS.search(text)
    if match is None:
        return None
    first = Decimal(match.group("first"))
    second = Decimal(match.group("second"))
    if abs(first - second) != 2:
        return None
    target = (first + second) / 2
    return match_unique(target, values, "number between neighbors")


def solve_counter_two_digit_extreme(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    match = _COUNTER_TWO_DIGIT_EXTREME.search(text)
    if match is None:
        return None
    bead_count = Decimal(match.group("count"))
    if bead_count != bead_count.to_integral_value() or not 1 <= bead_count <= 18:
        return None

    if match.group("direction") == "最大":
        tens = min(Decimal(9), bead_count)
    else:
        tens = max(Decimal(1), bead_count - 9)
    ones = bead_count - tens
    target = tens * 10 + ones
    return match_unique(
        target,
        values,
        f"counter {match.group('direction')} two-digit number",
    )


def solve_place_value_number(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    if _EXTREME_DIGIT_SUM.search(text):
        return match_unique(Decimal(19), values, "extreme digit sum")

    same_digits = _SAME_DIGITS_WITH_LOWER_BOUND.search(text)
    if same_digits and all(value is not None for value in values):
        lower = Decimal(same_digits.group("lower"))
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None
            and value == value.to_integral_value()
            and 10 <= value <= 99
            and value > lower
            and int(value) // 10 == int(value) % 10
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "same tens and ones digit")
        return None

    bounded_parity = _BOUNDED_PARITY.search(text)
    if bounded_parity and all(value is not None for value in values):
        lower = Decimal(bounded_parity.group("lower"))
        upper = Decimal(bounded_parity.group("upper"))
        wants_even = bounded_parity.group("parity") in {"偶数", "双数"}
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None
            and value == value.to_integral_value()
            and lower < value < upper
            and (int(value) % 2 == 0) == wants_even
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "bounded parity")
        return None

    match = _PLACE_VALUE_COMPARE.search(text)
    if match is not None:
        tens = _tens_digit(match.group("tens"))
        delta = Decimal(match.group("delta"))
        ones = (
            tens - delta
            if match.group("direction") in {"少", "小"}
            else tens + delta
        )
    else:
        match = _PLACE_VALUE_OPERATION.search(text)
        if match is not None:
            tens = _tens_digit(match.group("tens"))
            delta = Decimal(match.group("delta"))
            ones = (
                tens + delta
                if match.group("operation") == "加上"
                else tens - delta
            )
        else:
            reverse = _ONES_TO_TENS_COMPARE.search(text)
            if reverse is None:
                counted_units = _COUNTED_PLACE_UNITS.search(text)
                if counted_units is None:
                    return None
                count = Decimal(counted_units.group("count"))
                multiplier = {
                    "一": Decimal(1),
                    "十": Decimal(10),
                    "百": Decimal(100),
                    "千": Decimal(1000),
                }[counted_units.group("place")]
                return match_unique(
                    count * multiplier,
                    values,
                    "counted place units",
                )
            ones = Decimal(reverse.group("ones"))
            delta = Decimal(reverse.group("delta"))
            tens = (
                ones - delta
                if reverse.group("direction") in {"少", "小"}
                else ones + delta
            )

    if (
        tens != tens.to_integral_value()
        or ones != ones.to_integral_value()
        or not 1 <= tens <= 9
        or not 0 <= ones <= 9
    ):
        return None
    target = tens * 10 + ones
    return match_unique(target, values, "two-digit place value")


def _tens_digit(value: str) -> Decimal:
    return Decimal(9) if value == "最大的一位数" else Decimal(value)
