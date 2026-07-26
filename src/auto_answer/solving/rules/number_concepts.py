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
