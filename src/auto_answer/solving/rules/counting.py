"""Conservative rules for inclusive counting and queue positions."""

from __future__ import annotations

import re
from decimal import Decimal
from itertools import combinations, permutations

from .common import NUMBER

_STRICT_BETWEEN_COUNT = re.compile(
    rf"(?:比|大于)(?P<lower>{NUMBER})(?:大|)[,，、]?"
    rf"(?:比|小于)(?P<upper>{NUMBER})(?:小|)的数有(?:\(\)|多少|几)"
)
_INCLUSIVE_COUNT = re.compile(
    rf"(?:从)?第?(?P<start>{NUMBER})(?:页)?(?:数|读|看)?到"
    rf"第?(?P<end>{NUMBER})(?:页)?.*?(?:一共|今天|她|他).*?"
    rf"(?:数|读|看)了?(?:\(\)|多少|几)"
)
_FRONT_AND_BEHIND_TOTAL = re.compile(
    rf"前面有(?P<front>{NUMBER})人[^,，。]*[,，]"
    rf"后面有(?P<behind>{NUMBER})人.*?(?:一共|共有|总共)(?:有)?(?:几|多少)人"
)
_TWO_POSITIONS_BETWEEN = re.compile(
    rf"(?:排第|是第)(?P<first>{NUMBER})(?:个)?[^,，。]*[,，]"
    rf"[^,，。]*?(?:排第|是第)(?P<second>{NUMBER})(?:个)?.*?"
    rf"他们之间有(?:\(\)|多少|几)人"
)
_POSITION_FROM_BOTH_ENDS = re.compile(
    rf"从(?:前面|左)数[^,，。]*?第(?P<front>{NUMBER})(?:个)?[^,，。]*[,，]"
    rf"从(?:后面|右)数[^,，。]*?第(?P<back>{NUMBER})(?:个)?.*?"
    rf"(?:一队|这一队|这一排|全队).*?(?:共有|有)(?:\(\)|多少|几)人"
)
_DIGITS_FORM_TWO_DIGIT_NUMBERS = re.compile(
    r"用(?P<digits>\d(?:[,，、]\d)+).*?组成(?:\(\)|多少|几)个不同的两位数"
)
_PAIR_SUM_POSSIBILITIES = re.compile(
    r"用(?P<numbers>\d+(?:[,，、]\d+)+).*?任意两个数求和[,，]"
    r"得数有(?:\(\)|多少|几)种可能"
)
_THREE_STUDENTS_QUEUE = re.compile(
    r"[\u4e00-\u9fff]+、[\u4e00-\u9fff]+、[\u4e00-\u9fff]+"
    r"三个同学排队[,，]有(?:\(\)|多少|几)种排法"
)


def solve_counting_problem(text: str) -> Decimal | None:
    """Return a count only when one well-defined counting pattern matches."""
    digits = _DIGITS_FORM_TWO_DIGIT_NUMBERS.search(text)
    if digits:
        values = tuple(int(value) for value in re.split(r"[,，、]", digits.group("digits")))
        if len(values) != len(set(values)):
            return None
        formed = {
            tens * 10 + ones
            for tens, ones in permutations(values, 2)
            if tens != 0
        }
        return Decimal(len(formed))

    pair_sums = _PAIR_SUM_POSSIBILITIES.search(text)
    if pair_sums:
        values = tuple(
            int(value) for value in re.split(r"[,，、]", pair_sums.group("numbers"))
        )
        if len(values) != len(set(values)) or len(values) < 2:
            return None
        return Decimal(len({left + right for left, right in combinations(values, 2)}))

    if _THREE_STUDENTS_QUEUE.search(text):
        return Decimal(6)

    match = _STRICT_BETWEEN_COUNT.search(text)
    if match:
        lower = _integer(match.group("lower"))
        upper = _integer(match.group("upper"))
        if lower is None or upper is None or lower >= upper:
            return None
        return max(Decimal(), upper - lower - 1)

    match = _INCLUSIVE_COUNT.search(text)
    if match:
        start = _integer(match.group("start"))
        end = _integer(match.group("end"))
        if start is None or end is None or end < start:
            return None
        return end - start + 1

    match = _FRONT_AND_BEHIND_TOTAL.search(text)
    if match:
        front = _integer(match.group("front"))
        behind = _integer(match.group("behind"))
        if front is None or behind is None or front < 0 or behind < 0:
            return None
        return front + behind + 1

    match = _TWO_POSITIONS_BETWEEN.search(text)
    if match:
        first = _integer(match.group("first"))
        second = _integer(match.group("second"))
        if first is None or second is None or first <= 0 or second <= 0:
            return None
        return max(Decimal(), abs(first - second) - 1)

    match = _POSITION_FROM_BOTH_ENDS.search(text)
    if match:
        front = _integer(match.group("front"))
        back = _integer(match.group("back"))
        if front is None or back is None or front <= 0 or back <= 0:
            return None
        return front + back - 1
    return None


def _integer(value: str) -> Decimal | None:
    number = Decimal(value)
    return number if number == number.to_integral_value() else None
