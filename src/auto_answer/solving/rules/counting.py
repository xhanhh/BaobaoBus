"""Conservative rules for inclusive counting and queue positions."""

from __future__ import annotations

import re
from decimal import Decimal

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


def solve_counting_problem(text: str) -> Decimal | None:
    """Return a count only when one well-defined counting pattern matches."""
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
